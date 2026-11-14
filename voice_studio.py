"""Local CosyVoice2 studio. Run with .venv_cosyvoice beside fourth_generate.py.
Uses the existing model and best.pt. No additional web framework is needed.
"""
import argparse
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

ROOT = Path(__file__).resolve().parent
MAX_CHARS = 300
PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Voice Studio · Text to speech</title>
<style>
:root{color-scheme:dark;--bg:#10121c;--panel:#191c2a;--line:#303447;--muted:#abb0c7;--ink:#f3f2ff;--violet:#b6a1ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 80% 0,#27213e 0,transparent 45%),var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif;min-height:100vh}main{max-width:1150px;margin:auto;padding:32px 26px 50px}header{display:flex;align-items:center;justify-content:space-between;margin-bottom:56px}.brand{font-size:18px;font-weight:750;letter-spacing:-.4px}.logo{background:#b6a1ff;color:#211b33;border-radius:12px;padding:7px 12px;margin-right:10px}.badge{font-size:12px;color:#bce8d5;background:#18302b;border:1px solid #345349;padding:5px 12px;border-radius:30px}.eyebrow{color:var(--violet);font-size:12px;font-weight:700;letter-spacing:2px}h1{font-size:clamp(32px,5vw,52px);letter-spacing:-2px;line-height:1.13;margin:14px 0} .intro{color:var(--muted);max-width:610px;margin:0 0 34px}.grid{display:grid;grid-template-columns:1.2fr 1fr;gap:22px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:26px;box-shadow:0 20px 55px #0002}h2{font-size:18px;margin:0 0 5px}.hint{font-size:13px;color:var(--muted);margin:0 0 22px}.row{display:flex;justify-content:space-between;align-items:center;gap:12px}.counter{font-size:12px;color:var(--muted)}textarea{width:100%;min-height:215px;resize:vertical;background:#10131f;border:1px solid #3a3e54;border-radius:13px;padding:17px;color:var(--ink);font:17px/1.7 inherit;font-family:inherit;font-size:17px;line-height:1.7;margin:14px 0 7px}textarea:focus{outline:2px solid var(--violet);outline-offset:2px}button,a{touch-action:manipulation}button{font:inherit;cursor:pointer}button:focus-visible,a:focus-visible{outline:2px solid #d7caff;outline-offset:4px}.examples{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 24px}.chip{background:#24283a;color:#dadbed;border:1px solid #3c4159;border-radius:30px;padding:7px 12px;font-size:12px}.chip:hover{background:#34394f}.generate{width:100%;background:var(--violet);color:#211831;border:0;border-radius:12px;padding:14px;font-weight:750;font-size:15px}.generate:hover{background:#cabaff}button:disabled{opacity:.55;cursor:wait}.subnote{text-align:center;font-size:12px;color:var(--muted);margin:12px 0 0}.visual{display:flex;align-items:center;justify-content:center;gap:7px;height:145px;border-radius:16px;background:linear-gradient(140deg,#22233b,#161a29);margin:20px 0}.bar{width:6px;border-radius:10px;background:linear-gradient(180deg, #ff0000, #ff7f00, #ffff00, #00ff00, #0000ff, #4b0082, #8b00ff);height:var(--h)}.busy .bar{animation:pulse .85s ease-in-out infinite alternate;animation-delay:var(--d)}@keyframes pulse{to{transform:scaleY(.35);opacity:.4}}.status{min-height:48px;font-size:14px;color:var(--muted);margin:0 0 12px}.status.error{color:#ffb4b4}audio{width:100%;margin:10px 0 18px}.download{display:block;text-align:center;text-decoration:none;color:#e2d8ff;background:#29233e;border:1px solid #685788;padding:10px;border-radius:12px;font-size:14px}.metadata{color:var(--muted);font-size:12px}.empty{font-size:13px;color:var(--muted);padding:16px 0}footer{margin-top:28px;color:var(--muted);font-size:12px}.mini{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}.mini span{font-size:11px;border:1px solid var(--line);padding:3px 9px;border-radius:6px}details{margin-top:24px;font-size:13px;color:var(--muted)}summary{cursor:pointer}#spoken{overflow-wrap:anywhere;color:#d8d8e9;font-size:14px} [hidden]{display:none!important}@media(max-width:760px){.grid{grid-template-columns:1fr}header{margin-bottom:36px}.panel{padding:20px}main{padding:22px 16px}h1{letter-spacing:-1px}}@media(prefers-reduced-motion:reduce){.busy .bar{animation:none}}
</style></head><body><main>
<header><div class="brand"><span class="logo" aria-hidden="true">≈</span>Voice Studio</div><span class="badge">NVIDIA GEFORCE RTX</span></header>
<div class="eyebrow">YOUR WORDS, SPOKEN</div><h1>Give your text a voice.</h1>
<p class="intro">Improving Code-Switching Speech Recognition with TTS Data Augmentation</p>
<div class="grid"><section class="panel" aria-labelledby="compose"><div class="row"><h2 id="compose">"Give me something to say!"</h2><span class="counter" id="count">0 / 300</span></div>
<p class="hint">Mandarin · English · Mixed-language text</p>
<label for="text" class="metadata">Your text</label><textarea id="text" maxlength="300" placeholder="你好，welcome to my project。" spellcheck="false"></textarea>
<div class="examples"><button class="chip" data-text="चलो, आज देखते हैं ये स्मार्ट Qwen2.5 मॉडल आपके लिखे टेक्स्ट को कैसे मक्खन जैसी आवाज़ में बदलता है!"。(Create by Nisha)">Try mixed speech</button><button class="chip" data-text="Hello, welcome to my voice studio.">Try English</button><button class="chip" data-text="你好，很高兴认识你。">Try Mandarin</button></div>
<button class="generate" id="generate">Generate speech →</button><p calass="subnote">Up to 300 characters · Click  to generate</p>
<details><summary>About this voice</summary><p>Uses your saved CosyVoice2 LoRA checkpoint and the same repository reference voice used in your comparisons. Short sentences work best. This interface is for Mandarin and English; Hindi is not supported by this setup.</p></details></section>
<section class="panel" aria-labelledby="listen"><h2 id="listen">Listen to your creation</h2><p class="hint">Your latest generated audio appears here.</p>
<div class="visual" id="visual" aria-hidden="true"><span class="bar" style="--h:22px;--d:0s"></span><span class="bar" style="--h:43px;--d:.1s"></span><span class="bar" style="--h:70px;--d:.2s"></span><span class="bar" style="--h:38px;--d:.3s"></span><span class="bar" style="--h:88px;--d:.4s"></span><span class="bar" style="--h:60px;--d:.5s"></span><span class="bar" style="--h:95px;--d:.1s"></span><span class="bar" style="--h:54px;--d:.2s"></span><span class="bar" style="--h:73px;--d:.3s"></span><span class="bar" style="--h:39px;--d:.4s"></span><span class="bar" style="--h:20px;--d:.5s"></span></div>
<p class="status" id="status" role="status" aria-live="polite">"Keyboard uthaiye, kuch shabad yahan chhodiye, aur baaki kaam hum par chhodiye!".</p>
<div id="result" hidden><div class="metadata" id="meta"></div><audio id="audio" controls preload="metadata">Your browser does not support audio playback.</audio><a id="download" class="download" download>Download WAV ↓</a><p id="spoken"></p></div>
<div id="empty" class="empty">Your audio player will appear after generation.</div><div class="mini"><span>Fine-tuned CosyVoice2</span><span>WAV output</span><span>On-device inference</span></div>
</section></div><footer>Voice Studio · Research prototype · Audio is saved on your computer. Keep the Python terminal running while using this page.</footer>
</main><script>
const text=document.getElementById('text'),btn=document.getElementById('generate'),statusEl=document.getElementById('status');let running=false;
text.addEventListener('input',()=>document.getElementById('count').textContent=Array.from(text.value).length+' / 300');
document.querySelectorAll('.chip').forEach(b=>b.onclick=()=>{text.value=b.dataset.text;text.dispatchEvent(new Event('input'));text.focus()});
async function generate(){if(running)return;const value=text.value.trim();if(!value){statusEl.textContent='Please enter some text first.';statusEl.className='status error';text.focus();return}running=true;btn.disabled=true;btn.textContent='Generating…';document.getElementById('visual').classList.add('busy');statusEl.className='status';statusEl.textContent='Creating your speech on the GPU. Please wait…';document.getElementById('result').hidden=true;document.getElementById('empty').hidden=true;document.getElementById('audio').pause();
try{const res=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:value})});const data=await res.json();if(!res.ok)throw Error(data.error||'Generation failed.');const audio=document.getElementById('audio');audio.src=data.url;audio.load();const link=document.getElementById('download');link.href=data.url;link.download=data.filename;document.getElementById('meta').textContent=data.duration.toFixed(2)+' seconds · '+(data.sample_rate/1000)+' kHz · WAV';document.getElementById('spoken').textContent=value;document.getElementById('result').hidden=false;statusEl.textContent='Your speech is ready. Press play to listen.'}
catch(e){statusEl.className='status error';statusEl.textContent=e.message==='Failed to fetch'?'Connection lost. Check that the Python terminal is still running.':e.message;document.getElementById('empty').hidden=false}
finally{running=false;btn.disabled=false;btn.textContent='Generate speech →';document.getElementById('visual').classList.remove('busy')}}
btn.onclick=generate;text.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();generate()}});
</script></body></html>'''


class Studio:
    def __init__(self):
        self.lock = threading.Lock()
        self.output = ROOT / "voice_studio_output"
        self.output.mkdir(exist_ok=True)
        print("Loading the saved fine-tuned model. Please wait...", flush=True)
        import fourth_generate as s4
        self.s4 = s4
        if not s4.torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use .venv_cosyvoice and check your GPU.")
        config = s4.load_inference_config()
        self.sample_rate = int(config["sample_rate"])
        self.frontend = s4.CosyVoiceFrontEnd(
            get_tokenizer=config["get_tokenizer"], feat_extractor=config["feat_extractor"],
            campplus_model=str(s4.MODEL_DIR / "campplus.onnx"),
            speech_tokenizer_model=str(s4.MODEL_DIR / "speech_tokenizer_v2.onnx"),
            allowed_special=config["allowed_special"])
        self.model = s4.CosyVoice2Model(config["llm"], config["flow"], config["hift"], fp16=True)
        self.model.load(str(s4.MODEL_DIR / "llm.pt"), str(s4.MODEL_DIR / "flow.pt"),
                        str(s4.MODEL_DIR / "hift.pt"))
        del config
        gc.collect()
        s4.apply_trained_lora(self.model)
        print("Model ready. GPU:", s4.torch.cuda.get_device_name(0), flush=True)

    def generate(self, text):
        # Handler holds this studio's lock for the entire inference operation.
        self.s4.TARGET_TEXT = text
        self.s4.SEED = 42
        self.s4.OUTPUT_DIR = self.output
        filename = "speech_" + uuid.uuid4().hex + ".wav"
        path = self.s4.generate_audio(self.frontend, self.model, self.sample_rate, filename)
        info = self.s4.sf.info(str(path))
        path.with_suffix(".json").write_text(json.dumps({
            "text": text, "checkpoint": str(self.s4.CHECKPOINT_PATH),
            "reference_wav": str(self.s4.REFERENCE_WAV), "seed": 42,
            "sample_rate": info.samplerate, "duration": info.duration,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"url": "/audio/" + filename, "filename": filename,
                "duration": info.duration, "sample_rate": info.samplerate}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_data(self, status, body, content_type):
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

    def json_response(self, status, payload):
        self.send_data(status, json.dumps(payload).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            return self.send_data(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if re.fullmatch(r"/audio/speech_[0-9a-f]{32}\.wav", path):
            file = self.server.studio.output / path.rsplit("/", 1)[1]
            if file.is_file():
                return self.send_data(200, file.read_bytes(), "audio/wav")
        self.json_response(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/generate":
            return self.json_response(404, {"error": "Not found"})
        origin = self.headers.get("Origin")
        expected = f"http://127.0.0.1:{self.server.server_port}"
        if origin and origin != expected:
            return self.json_response(403, {"error": "Open the studio using " + expected})
        if self.headers.get_content_type() != "application/json":
            return self.json_response(400, {"error": "Expected JSON input."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("Invalid request size.")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
                raise ValueError("Please enter text.")
            text = payload["text"].strip()
            if not text or len(text) > MAX_CHARS:
                raise ValueError("Please enter between 1 and 300 characters.")
            if not any(c.isalnum() for c in text):
                raise ValueError("Please include words, not only punctuation.")
        except (ValueError, UnicodeDecodeError) as exc:
            return self.json_response(400, {"error": str(exc)})
        studio = self.server.studio
        if not studio.lock.acquire(blocking=False):
            return self.json_response(409, {"error": "Another audio is generating. Please try again when it finishes."})
        try:
            result = studio.generate(text)
        except Exception:
            traceback.print_exc()
            self.json_response(500, {"error": "Generation failed. Check the Python terminal for details. Try a shorter sentence; close other GPU tasks if memory is full."})
        else:
            self.json_response(200, result)
        finally:
            studio.lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not (ROOT / "fourth_generate.py").is_file():
        raise FileNotFoundError("Save voice_studio.py in the same folder as fourth_generate.py.")
    # Reserve the local port before loading the GPU model.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.studio = Studio()
        url = f"http://127.0.0.1:{server.server_port}"
        print("\nVOICE STUDIO READY:", url, flush=True)
        print("Keep this terminal open. Press Ctrl+C to stop.", flush=True)
        print("Saved audio folder:", server.studio.output, flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nVoice Studio stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
