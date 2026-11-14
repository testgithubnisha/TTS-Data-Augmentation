"""Local unified CosyVoice2 TTS + Whisper ASR demo UI.

Save this file beside fourth_generate.py in the AI project folder and run it
with the existing .venv_cosyvoice environment. The server binds to localhost only.
Models are loaded lazily and switched sequentially to reduce GPU memory pressure.
"""
from __future__ import annotations

import argparse
import base64
import gc
import json
import re
import threading
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import torch
import torch.nn as nn
import whisper

PROJECT = Path(__file__).resolve().parent
WHISPER_CHECKPOINT = PROJECT / "whisper_models" / "small.pt"
OUTPUT_DIR = PROJECT / "voice_asr_studio_output"
UPLOAD_DIR = PROJECT / "voice_asr_studio_uploads"
MAX_CHARS = 300
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_REQUEST_BYTES = 42 * 1024 * 1024
ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".mpeg"}
MODEL_NAMES = {
    "original_whisper_small": "Original Whisper-small (no ASR adapter)",
    "experiment_A_real": "Experiment A — real-only LoRA",
    "experiment_B_augmented": "Experiment B — real + synthetic LoRA",
}

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASR + TTS Research Studio</title>
<style>
:root{color-scheme:dark;--bg:#10121c;--panel:#191c2a;--line:#303447;--muted:#abb0c7;--ink:#f3f2ff;--violet:#b6a1ff;--green:#8ee0bc;--red:#ffb4b4}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 80% 0,#27213e 0,transparent 45%),var(--bg);color:var(--ink);font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif;min-height:100vh}main{max-width:1180px;margin:auto;padding:28px 22px 50px}header{display:flex;gap:16px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-bottom:36px}.brand{font-size:19px;font-weight:800;letter-spacing:-.4px}.logo{background:var(--violet);color:#211b33;border-radius:12px;padding:7px 12px;margin-right:10px}.badge{font-size:12px;color:var(--green);background:#18302b;border:1px solid #345349;padding:5px 12px;border-radius:30px}.eyebrow{color:var(--violet);font-size:12px;font-weight:800;letter-spacing:2px}h1{font-size:clamp(30px,5vw,48px);letter-spacing:-1.5px;line-height:1.12;margin:12px 0}.intro{color:var(--muted);max-width:780px;margin:0 0 26px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:23px;box-shadow:0 20px 55px #0002;min-width:0}.full{grid-column:1/-1}h2{font-size:18px;margin:0 0 5px}.hint{font-size:13px;color:var(--muted);margin:0 0 18px}.label{display:block;font-size:13px;color:#d4d5e5;margin:12px 0 6px}textarea,select,input[type=file]{width:100%;background:#10131f;border:1px solid #3a3e54;border-radius:11px;padding:12px;color:var(--ink);font:inherit}textarea{min-height:130px;resize:vertical}textarea:focus,select:focus,input:focus{outline:2px solid var(--violet);outline-offset:2px}.actions{display:flex;gap:9px;flex-wrap:wrap;margin-top:13px}button{font:inherit;font-weight:700;cursor:pointer;border:0;border-radius:11px;padding:11px 15px;background:var(--violet);color:#211831}button.secondary{background:#292d42;color:var(--ink);border:1px solid #444961}button:disabled{opacity:.55;cursor:wait}.status{min-height:25px;margin:10px 0;color:var(--muted);font-size:13px;overflow-wrap:anywhere}.status.error{color:var(--red)}.status.ok{color:var(--green)}.result{border-top:1px solid var(--line);padding-top:14px;margin-top:14px}.result h3{font-size:14px;margin:0 0 8px}.result-text{white-space:pre-wrap;overflow-wrap:anywhere;background:#10131f;border:1px solid var(--line);border-radius:11px;padding:13px;min-height:54px}.result audio{width:100%;margin:8px 0}.download{display:inline-block;color:#e2d8ff;text-decoration:none;background:#29233e;border:1px solid #685788;padding:8px 11px;border-radius:10px;font-size:13px}.compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.mini{font-size:12px;color:var(--muted)}.note{border-left:3px solid #7e6bb2;padding:9px 12px;color:var(--muted);font-size:12px;background:#171928;border-radius:0 8px 8px 0;margin-top:16px}footer{margin-top:24px;color:var(--muted);font-size:12px} [hidden]{display:none!important}@media(max-width:760px){.grid,.compare-grid{grid-template-columns:1fr}.full{grid-column:auto}main{padding:20px 13px}.panel{padding:17px}header{margin-bottom:26px}}
</style></head><body><main>
<header><div class="brand"><span class="logo" aria-hidden="true">≈</span>ASR + TTS Research Studio</div><span class="badge">nvidia GEFORCE RTX</span></header>
<div class="eyebrow">LOW-RESOURCE SPEECH RESEARCH</div><h1>Improving Code-Switching Speech Recognition with TTS Data Augmentation</h1>
<p class="intro">This local demo uses the saved CosyVoice2 TTS LoRA checkpoint and lets you choose the original Whisper-small model or either trained Whisper ASR adapter. It does not train or modify any checkpoint.</p>
<div class="grid">
<section class="panel"><h2>1. Text → speech → transcription</h2><p class="hint">Generate synthetic speech with CosyVoice2, then recognize that generated audio with the selected Whisper model.</p>
<label class="label" for="ttsText">Text to speak (up to 300 characters)</label><textarea id="ttsText" maxlength="300" placeholder="例如：你好，welcome to my project。"></textarea>
<label class="label" for="pipelineModel">Whisper model for transcription</label><select id="pipelineModel"><option value="experiment_A_real">Experiment A — real-only LoRA</option><option value="experiment_B_augmented">Experiment B — real + synthetic LoRA</option><option value="original_whisper_small">Original Whisper-small (baseline)</option></select>
<label class="label" for="pipelineLanguage">Transcription language</label><select id="pipelineLanguage"><option value="zh">Chinese / Chinese-English code-switching (zh)</option><option value="en">English (en)</option><option value="auto">Auto-detect</option></select>
<div class="actions"><button id="pipelineBtn">Generate + transcribe</button><button id="ttsOnlyBtn" class="secondary">Generate speech only</button></div><p class="status" id="pipelineStatus" role="status" aria-live="polite"></p>
<div class="result" id="pipelineAudioResult" hidden><h3>Generated audio</h3><audio id="pipelineAudio" controls></audio><a id="pipelineDownload" class="download" download>Download WAV</a><p class="mini" id="pipelineAudioMeta"></p></div>
<div class="result" id="pipelineTranscriptResult" hidden><h3>Whisper transcription output</h3><div class="result-text" id="pipelineTranscript"></div><p class="mini" id="pipelineModelMeta"></p></div>
</section>
<section class="panel"><h2>2. Upload audio → transcription</h2><p class="hint">Transcribe an existing recording with the original model or a trained adapter.</p>
<label class="label" for="audioFile">Audio file</label><input id="audioFile" type="file" accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm,.mp4">
<label class="label" for="uploadModel">Whisper model</label><select id="uploadModel"><option value="experiment_A_real">Experiment A — real-only LoRA</option><option value="experiment_B_augmented">Experiment B — real + synthetic LoRA</option><option value="original_whisper_small">Original Whisper-small (baseline)</option></select>
<label class="label" for="uploadLanguage">Transcription language</label><select id="uploadLanguage"><option value="zh">Chinese / Chinese-English code-switching (zh)</option><option value="en">English (en)</option><option value="auto">Auto-detect</option></select>
<div class="actions"><button id="transcribeBtn">Transcribe audio</button><button id="compareBtn" class="secondary">Compare A vs B</button></div><p class="status" id="uploadStatus" role="status" aria-live="polite"></p>
<div class="result" id="uploadResult" hidden><h3>Transcription</h3><div class="result-text" id="uploadTranscript"></div><p class="mini" id="uploadMeta"></p></div>
<div class="result" id="compareResult" hidden><h3>Same audio: Experiment A vs Experiment B</h3><div class="compare-grid"><div><p class="mini">Experiment A — real-only</p><div class="result-text" id="compareA"></div></div><div><p class="mini">Experiment B — real + synthetic</p><div class="result-text" id="compareB"></div></div></div></div>
<div class="note">Audio uploads are processed locally by the Python server. Whisper decoding of non-WAV formats may require FFmpeg, as in the Whisper package. Use recordings you are permitted to process.</div>
</section>
<section class="panel full"><h2>3. Research result snapshot</h2><p class="hint">Your completed evaluation summary, copied here for demonstration. These are aggregate test-set results, not predictions for the single audio above.</p>
<div class="compare-grid"><div><h3>Original Whisper-small</h3><div class="result-text">WER: 61.8635%<br>CER: 52.6707%</div></div><div><h3>Experiment A — real-only LoRA</h3><div class="result-text">WER: 19.7424%<br>CER: 14.3056%</div></div><div><h3>Experiment B — real + synthetic LoRA</h3><div class="result-text">WER: 27.0134%<br>CER: 18.5581%</div></div><div><h3>Experiment B vs A</h3><div class="result-text">WER: 36.83% relative error increase<br>CER: 29.73% relative error increase</div></div></div>
<p class="mini">Note: the negative relative improvement reported by the evaluation means Experiment B has higher error than A (so it is worse on this test). Values are shown for reference; re-run evaluation if results/settings change.</p>
</section>
</div><footer>ASR + TTS Research Studio · Local research prototype · Keep this Python terminal open while using the page. Existing training scripts and checkpoints are not modified.</footer>
</main><script>
const $=id=>document.getElementById(id);
async function post(path,payload){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});let d;try{d=await r.json()}catch{throw new Error('The server returned an unreadable response. Check the Python terminal.')}if(!r.ok)throw new Error(d.error||('Request failed ('+r.status+')'));return d}
function status(id,msg,kind=''){const el=$(id);el.textContent=msg;el.className='status '+kind}
function busy(button,yes,label){button.disabled=yes;if(yes){button.dataset.old=button.textContent;button.textContent=label||'Working…'}else{button.textContent=button.dataset.old||button.textContent}}
function showAudio(prefix,d){const audio=$(prefix+'Audio');audio.src=d.audio_url;audio.load();const link=$(prefix+'Download');link.href=d.audio_url;link.download=d.filename;$(prefix+'AudioMeta').textContent=(Number(d.duration||0).toFixed(2))+' seconds · '+((Number(d.sample_rate||0)/1000).toFixed(1))+' kHz · generated by CosyVoice2 TTS LoRA';$(prefix+'AudioResult').hidden=false}
async function readFileBase64(file){return await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(',')[1]||'');r.onerror=()=>reject(new Error('Could not read the selected file.'));r.readAsDataURL(file)})}
$('pipelineBtn').onclick=async()=>{const text=$('ttsText').value.trim();if(!text){status('pipelineStatus','Enter some text first.','error');return}const b=$('pipelineBtn');busy(b,true,'Generating and transcribing…');$('pipelineTranscriptResult').hidden=true;try{status('pipelineStatus','Loading models as needed. TTS and ASR run sequentially to reduce GPU memory use.');const d=await post('/api/pipeline',{text,model:$('pipelineModel').value,language:$('pipelineLanguage').value});showAudio('pipeline',d);$('pipelineTranscript').textContent=d.transcription;$('pipelineModelMeta').textContent='Recognized by '+d.model_label+' · language setting: '+d.language; $('pipelineTranscriptResult').hidden=false;status('pipelineStatus','Pipeline completed successfully.','ok')}catch(e){status('pipelineStatus',e.message,'error')}finally{busy(b,false)}};
$('ttsOnlyBtn').onclick=async()=>{const text=$('ttsText').value.trim();if(!text){status('pipelineStatus','Enter some text first.','error');return}const b=$('ttsOnlyBtn');busy(b,true,'Generating speech…');try{status('pipelineStatus','Generating speech with CosyVoice2 TTS LoRA…');const d=await post('/api/tts',{text});showAudio('pipeline',d);$('pipelineTranscriptResult').hidden=true;status('pipelineStatus','Speech generated successfully.','ok')}catch(e){status('pipelineStatus',e.message,'error')}finally{busy(b,false)}};
async function uploadPayload(){const f=$('audioFile').files[0];if(!f)throw new Error('Choose an audio file first.');if(f.size>30*1024*1024)throw new Error('Please choose an audio file smaller than 30 MB.');return {filename:f.name,audio_base64:await readFileBase64(f),model:$('uploadModel').value,language:$('uploadLanguage').value}}
$('transcribeBtn').onclick=async()=>{const b=$('transcribeBtn');busy(b,true,'Transcribing…');$('uploadResult').hidden=true;try{status('uploadStatus','Uploading audio to the local server and running Whisper…');const d=await post('/api/transcribe',await uploadPayload());$('uploadTranscript').textContent=d.transcription;$('uploadMeta').textContent=d.model_label+' · language setting: '+d.language;$('uploadResult').hidden=false;status('uploadStatus','Transcription completed.','ok')}catch(e){status('uploadStatus',e.message,'error')}finally{busy(b,false)}};
$('compareBtn').onclick=async()=>{const b=$('compareBtn');busy(b,true,'Comparing A and B…');$('compareResult').hidden=true;try{status('uploadStatus','Running Experiment A and B sequentially on the same audio. This may take a few minutes…');const p=await uploadPayload();const d=await post('/api/compare',p);$('compareA').textContent=d.experiment_A;$('compareB').textContent=d.experiment_B;$('compareResult').hidden=false;status('uploadStatus','Both adapters processed the same audio.','ok')}catch(e){status('uploadStatus',e.message,'error')}finally{busy(b,false)}};
</script></body></html>'''


class LoRALinear(nn.Module):
    """Same adapter module layout as train_whisper_lora.py."""
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Preserve the pretrained Whisper model's native mixed-precision behavior.
        base_output = self.base(x)
        adapter_input = x.to(dtype=self.lora_A.weight.dtype)
        adapter_output = self.lora_B(self.dropout(self.lora_A(adapter_input)))
        return base_output + adapter_output.to(dtype=base_output.dtype) * self.scale


def add_lora(model, rank=8, alpha=16, dropout=0.05):
    replaced = 0
    for group in (model.encoder.blocks, model.decoder.blocks):
        for block in group:
            attentions = [block.attn]
            if getattr(block, "cross_attn", None) is not None:
                attentions.append(block.cross_attn)
            for attn in attentions:
                for attr in ("query", "key", "value", "out"):
                    layer = getattr(attn, attr, None)
                    if isinstance(layer, nn.Linear):
                        setattr(attn, attr, LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout))
                        replaced += 1
    if replaced == 0:
        raise RuntimeError("No Whisper attention projection layers were adapted; model layout may differ.")
    return replaced


class Studio:
    def __init__(self):
        self.lock = threading.RLock()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.s4 = None
        self.tts_model = None
        self.tts_frontend = None
        self.tts_sample_rate = None
        self.asr_model = None
        self.asr_key = None
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This project is configured for the NVIDIA GPU.")
        if not WHISPER_CHECKPOINT.is_file():
            raise FileNotFoundError(f"Whisper checkpoint not found: {WHISPER_CHECKPOINT}")
        if not (PROJECT / "fourth_generate.py").is_file():
            raise FileNotFoundError("Save asr_tts_studio.py in the same folder as fourth_generate.py.")
        print("Studio server initialized. Models will load only when requested.", flush=True)
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    def _cuda_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _unload_asr(self):
        if self.asr_model is not None:
            print("Unloading Whisper model to free GPU memory...", flush=True)
        self.asr_model = None
        self.asr_key = None
        self._cuda_cleanup()

    def _unload_tts(self):
        if self.tts_model is not None:
            print("Unloading CosyVoice2 to free GPU memory...", flush=True)
        self.tts_model = None
        self.tts_frontend = None
        self.tts_sample_rate = None
        self._cuda_cleanup()

    def _ensure_tts(self):
        self._unload_asr()
        if self.tts_model is not None:
            return
        if self.s4 is None:
            import sys
            if str(PROJECT) not in sys.path:
                sys.path.insert(0, str(PROJECT))
            import fourth_generate as s4
            self.s4 = s4
        s4 = self.s4
        if not s4.CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"CosyVoice2 TTS LoRA checkpoint not found: {s4.CHECKPOINT_PATH}")
        required = [s4.MODEL_DIR / "llm.pt", s4.MODEL_DIR / "flow.pt", s4.MODEL_DIR / "hift.pt"]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError("CosyVoice2 model file(s) not found: " + ", ".join(missing))
        print("Loading CosyVoice2 and TTS LoRA. This can take a while...", flush=True)
        config = s4.load_inference_config()
        sample_rate = int(config["sample_rate"])
        frontend = s4.CosyVoiceFrontEnd(
            get_tokenizer=config["get_tokenizer"],
            feat_extractor=config["feat_extractor"],
            campplus_model=str(s4.MODEL_DIR / "campplus.onnx"),
            speech_tokenizer_model=str(s4.MODEL_DIR / "speech_tokenizer_v2.onnx"),
            allowed_special=config["allowed_special"],
        )
        model = s4.CosyVoice2Model(config["llm"], config["flow"], config["hift"], fp16=True)
        model.load(str(s4.MODEL_DIR / "llm.pt"), str(s4.MODEL_DIR / "flow.pt"), str(s4.MODEL_DIR / "hift.pt"))
        del config
        gc.collect()
        s4.apply_trained_lora(model)
        self.tts_frontend = frontend
        self.tts_model = model
        # Read from the inference config helper via the model setup the same way as voice_studio.py.
        # `sample_rate` is kept by fourth_generate.py's inference configuration.
        cfg = s4.load_inference_config()
        self.tts_sample_rate = sample_rate
        print("CosyVoice2 TTS LoRA loaded.", flush=True)

    def _generate_tts(self, text: str):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Please enter text to generate speech.")
        text = text.strip()
        if len(text) > MAX_CHARS:
            raise ValueError(f"Please use no more than {MAX_CHARS} characters.")
        if not any(c.isalnum() for c in text):
            raise ValueError("Please include words, not only punctuation.")
        self._ensure_tts()
        filename = "tts_" + uuid.uuid4().hex + ".wav"
        self.s4.TARGET_TEXT = text
        self.s4.SEED = 42
        self.s4.OUTPUT_DIR = OUTPUT_DIR
        generated = self.s4.generate_audio(self.tts_frontend, self.tts_model, self.tts_sample_rate, filename)
        path = Path(generated) if generated else OUTPUT_DIR / filename
        if not path.is_absolute():
            path = (PROJECT / path).resolve()
        if not path.is_file():
            fallback = OUTPUT_DIR / filename
            if fallback.is_file():
                path = fallback
            else:
                raise RuntimeError(f"TTS finished but output WAV was not found: {path}")
        try:
            info = self.s4.sf.info(str(path))
            duration = float(info.duration)
            sample_rate = int(info.samplerate)
        except Exception:
            duration, sample_rate = 0.0, int(self.tts_sample_rate)
        return {"audio_url": "/audio/" + path.name, "filename": path.name, "duration": duration, "sample_rate": sample_rate, "path": str(path)}

    def _load_asr(self, choice: str):
        if choice not in MODEL_NAMES:
            raise ValueError("Unknown Whisper model choice.")
        if self.asr_model is not None and self.asr_key == choice:
            return self.asr_model
        self._unload_tts()
        self._unload_asr()
        adapter_path = PROJECT / "whisper_runs" / choice / "adapter_checkpoint.pt"
        if choice != "original_whisper_small" and not adapter_path.is_file():
            raise FileNotFoundError(f"Whisper adapter checkpoint not found: {adapter_path}")
        print(f"Loading {MODEL_NAMES[choice]} from the original small.pt checkpoint...", flush=True)
        model = whisper.load_model(str(WHISPER_CHECKPOINT), device=self.device)
        if choice != "original_whisper_small":
            payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
            state = payload.get("adapter_state_dict")
            if not isinstance(state, dict) or not state:
                raise ValueError(f"No adapter_state_dict found in {adapter_path}")
            rank = int(payload.get("rank", 8))
            alpha = int(payload.get("alpha", 16))
            dropout = float(payload.get("dropout", 0.05))
            add_lora(model, rank=rank, alpha=alpha, dropout=dropout)
            keys = set(model.state_dict().keys())
            unexpected = [k for k in state if k not in keys]
            if unexpected:
                raise RuntimeError("Adapter keys do not match the loaded Whisper model: " + ", ".join(unexpected[:5]))
            incompatible = model.load_state_dict(state, strict=False)
            if incompatible.unexpected_keys:
                raise RuntimeError("Unexpected adapter weights: " + ", ".join(incompatible.unexpected_keys[:5]))
            if not any("lora_A" in k or "lora_B" in k for k in state):
                raise RuntimeError("The checkpoint does not contain LoRA adapter weights.")
            print(f"Loaded {len(state)} LoRA tensors for {choice}.", flush=True)
        model.eval()
        self.asr_model = model
        self.asr_key = choice
        print(f"Whisper model ready: {choice}", flush=True)
        return model

    def _transcribe(self, audio_path: Path, choice: str, language: str):
        if language not in {"zh", "en", "auto"}:
            raise ValueError("Language must be zh, en, or auto.")
        model = self._load_asr(choice)
        lang_arg = None if language == "auto" else language
        with torch.inference_mode():
            result = model.transcribe(
                str(audio_path),
                language=lang_arg,
                task="transcribe",
                fp16=(self.device.type == "cuda"),
                verbose=False,
                temperature=0.0,
                condition_on_previous_text=False,
            )
        return (result.get("text") or "").strip()

    def transcribe_upload(self, filename: str, audio_b64: str, choice: str, language: str):
        if not isinstance(audio_b64, str) or not audio_b64:
            raise ValueError("Please select an audio file.")
        try:
            audio_bytes = base64.b64decode(audio_b64, validate=True)
        except Exception as exc:
            raise ValueError("The uploaded audio payload is not valid base64.") from exc
        if not audio_bytes:
            raise ValueError("The selected audio file is empty.")
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError("Audio file is too large. Maximum upload size is 30 MB.")
        suffix = Path(filename or "audio.wav").suffix.lower()
        if suffix not in ALLOWED_AUDIO_SUFFIXES:
            raise ValueError("Unsupported audio extension. Use WAV, MP3, M4A, FLAC, OGG, or WebM.")
        path = UPLOAD_DIR / ("upload_" + uuid.uuid4().hex + suffix)
        path.write_bytes(audio_bytes)
        text = self._transcribe(path, choice, language)
        return {"transcription": text, "model": choice, "model_label": MODEL_NAMES[choice], "language": language, "saved_audio": str(path)}


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalASRTTSStudio/1.0"

    def log_message(self, fmt, *args):
        # Keep the terminal readable; critical exceptions are printed in do_POST.
        pass

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, status: int, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/health":
            studio = self.server.studio
            return self._json(200, {"ok": True, "gpu": torch.cuda.get_device_name(0), "active_model": studio.asr_key or ("cosyvoice2_tts" if studio.tts_model is not None else None)})
        match = re.fullmatch(r"/audio/(tts_[0-9a-f]{32}\.wav)", path)
        if match:
            file_path = OUTPUT_DIR / match.group(1)
            if file_path.is_file():
                return self._send(200, file_path.read_bytes(), "audio/wav")
        return self._json(404, {"error": "Not found."})

    def do_POST(self):
        if self.path not in {"/api/tts", "/api/transcribe", "/api/pipeline", "/api/compare"}:
            return self._json(404, {"error": "Not found."})
        origin = self.headers.get("Origin")
        expected = f"http://127.0.0.1:{self.server.server_port}"
        if origin and origin not in {expected, f"http://localhost:{self.server.server_port}"}:
            return self._json(403, {"error": "Open this studio using the local address printed in the Python terminal."})
        if self.headers.get_content_type() != "application/json":
            return self._json(400, {"error": "Expected a JSON request."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("Invalid request size. Audio uploads are limited to 30 MB.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Invalid request body.")
            studio: Studio = self.server.studio
            with studio.lock:
                if self.path == "/api/tts":
                    result = studio._generate_tts(payload.get("text", ""))
                    return self._json(200, result)
                if self.path == "/api/transcribe":
                    result = studio.transcribe_upload(payload.get("filename", "audio.wav"), payload.get("audio_base64", ""), payload.get("model", "experiment_A_real"), payload.get("language", "zh"))
                    return self._json(200, result)
                if self.path == "/api/pipeline":
                    audio = studio._generate_tts(payload.get("text", ""))
                    transcription = studio._transcribe(Path(audio["path"]), payload.get("model", "experiment_A_real"), payload.get("language", "zh"))
                    audio.update({"transcription": transcription, "model": payload.get("model", "experiment_A_real"), "model_label": MODEL_NAMES.get(payload.get("model", "experiment_A_real"), "Selected Whisper model"), "language": payload.get("language", "zh")})
                    return self._json(200, audio)
                # Compare A and B on exactly the same uploaded recording.
                audio_b64 = payload.get("audio_base64", "")
                filename = payload.get("filename", "audio.wav")
                language = payload.get("language", "zh")
                a = studio.transcribe_upload(filename, audio_b64, "experiment_A_real", language)["transcription"]
                b = studio.transcribe_upload(filename, audio_b64, "experiment_B_augmented", language)["transcription"]
                return self._json(200, {"experiment_A": a, "experiment_B": b, "language": language})
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            message = str(exc).strip() or exc.__class__.__name__
            return self._json(500, {"error": message + " Check the Python terminal for the full traceback."})


def main():
    parser = argparse.ArgumentParser(description="Local CosyVoice2 TTS LoRA + Whisper ASR LoRA studio.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not (PROJECT / "fourth_generate.py").is_file():
        raise FileNotFoundError("Place asr_tts_studio.py in the same project folder as fourth_generate.py.")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.studio = Studio()
        url = f"http://127.0.0.1:{server.server_port}"
        print("\nASR + TTS STUDIO READY:", url, flush=True)
        print("TTS LoRA checkpoint: loaded lazily from fourth_generate.py", flush=True)
        print("Whisper adapters: whisper_runs\\experiment_A_real and whisper_runs\\experiment_B_augmented", flush=True)
        print("Models switch sequentially to reduce VRAM use.", flush=True)
        print("Keep this terminal open. Press Ctrl+C to stop.", flush=True)
        print("Output folder:", OUTPUT_DIR, flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStudio stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
