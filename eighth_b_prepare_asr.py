"""Step 8B: prepare auditable real-only and provisional augmented ASR data.
Run from the project venv. No model training or ASR scoring occurs here.
JSONL audio paths are relative to the output directory; keep it intact.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDED = {'synthetic_0065', 'synthetic_0097'}
CONFIRMED = {'synthetic_0081', 'synthetic_0051', 'synthetic_0092',
             'synthetic_0035', 'synthetic_0100'}
IMPLIED = {'synthetic_0019', 'synthetic_0050', 'synthetic_0061', 'synthetic_0016'}
EXPECTED_HASH = '6629c0789629e56538e345f797fcfa9c60da73beb17c803ed84c32b5ba1a2970'

def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'synthetic_pilot' / 'run_20261007_214529_267060')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest_path = run / 'manifest.jsonl'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise RuntimeError('Pilot manifest differs from the reviewed 100-sample run. Stop and check the input folder.')
    pilot = read_jsonl(manifest_path)
    ids = [r['synthetic_id'] for r in pilot]
    if len(ids) != 100 or len(set(ids)) != 100 or not EXCLUDED.issubset(ids):
        raise RuntimeError('Expected the reviewed 100 unique synthetic samples.')
    quality_dir = run / 'quality_20261007_225111_181291'
    quality = json.loads((quality_dir / 'quality_summary.json').read_text(encoding='utf-8-sig'))
    if quality['status'] != 'complete' or quality['manifest_sha256'] != EXPECTED_HASH:
        raise RuntimeError('Quality report does not match this pilot.')
    with (quality_dir / 'quality_review.csv').open(encoding='utf-8-sig', newline='') as f:
        review_rows = list(csv.DictReader(f))
    review = {r['synthetic_id']: r for r in review_rows}
    if len(review_rows) != 100 or set(review) != set(ids):
        raise RuntimeError('Quality CSV is incomplete or contains duplicate IDs.')

    print('STEP 8B: PREPARE ASR DATA', flush=True)
    print('Loading data packages; no GPU/model required...', flush=True)
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from datasets import load_dataset, Audio
    import datasets

    def export_audio(source, dest):
        x, sr = sf.read(source, dtype='float32', always_2d=True)
        if sr <= 0 or x.size == 0 or not np.isfinite(x).all():
            raise ValueError('Empty or non-finite audio')
        x = x.mean(axis=1)
        if not np.any(x):
            raise ValueError('Silent audio')
        original_sr = int(sr)
        if sr != 16000:
            g = math.gcd(int(sr), 16000)
            x = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
        peak = float(np.abs(x).max())
        if not math.isfinite(peak) or peak == 0:
            raise ValueError('Invalid resampled audio')
        # FLOAT preserves resampling peaks without PCM clipping/normalization.
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), x, 16000, subtype='FLOAT')
        return {'sample_rate': 16000, 'duration_seconds': len(x) / 16000,
                'original_sample_rate': original_sr, 'peak': peak}

    print('Loading ASCEND (uses Hugging Face cache when available)...', flush=True)
    ds = load_dataset('CAiRE/ASCEND')
    for split, expected in [('train', 9869), ('validation', 1130), ('test', 1315)]:
        if len(ds[split]) != expected:
            raise RuntimeError(f'{split} size changed; expected {expected}. Check dataset version.')
        ds[split] = ds[split].cast_column('audio', Audio(decode=False))
    # Verify every synthetic utterance maps to its original TRAIN transcript.
    train_texts = ds['train']['transcription']
    for row in pilot:
        idx = row['source_index']
        if (row.get('source_split') != 'train' or row.get('is_synthetic') is not True
                or not isinstance(idx, int) or not 0 <= idx < len(train_texts)
                or row['text'].strip() != train_texts[idx].strip()):
            raise RuntimeError(f"Source pairing mismatch: {row['synthetic_id']}")
        audio = (run / row['audio_path']).resolve()
        if not audio.is_relative_to(run) or not audio.is_file():
            raise RuntimeError(f'Invalid pilot audio path: {audio}')
        qr = review[row['synthetic_id']]
        if qr['text'] != row['text'] or qr['technical_status'] != 'pass':
            raise RuntimeError(f"Quality pairing/check failed: {row['synthetic_id']}")

    out = ROOT / 'asr_prepared' / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    status = {'status': 'incomplete', 'source_dataset': 'CAiRE/ASCEND',
              'datasets_version': datasets.__version__, 'pilot_manifest_sha256': EXPECTED_HASH,
              'source_pilot': str(run), 'path_base': 'this output folder',
              'sample_rate': 16000, 'audio_subtype': 'FLOAT',
              'policy': 'Original real splits preserved. No ASR-score filtering or transcript replacement. Two user-uncertain synthetic samples excluded.',
              'review_limitation': 'Provisional synthetic set. Only selected English portions reviewed by user; Chinese and other content not fully verified.',
              'original_pilot_asr_mer_percent': quality['corpus_mixed_error_rate_percent'],
              'fingerprints': {k: ds[k]._fingerprint for k in ('train', 'validation', 'test')}}
    write_json(out / 'preparation_summary.json', status)
    print('Output:', out, flush=True)
    real = {}
    try:
        for split in ('train', 'validation', 'test'):
            rows = []
            for i, sample in enumerate(ds[split]):
                text = sample['transcription'].strip()
                if not text:
                    raise ValueError(f'Empty real transcript: {split}/{i}; no silent skipping.')
                a = sample['audio']
                source = io.BytesIO(a['bytes']) if a.get('bytes') is not None else a.get('path')
                if source is None:
                    raise ValueError(f'Missing audio bytes/path: {split}/{i}')
                uid = f'ascend_{split}_{i:06d}'
                relative = Path('audio') / split / (uid + '.wav')
                info = export_audio(source, out / relative)
                rows.append({'id': uid, 'audio_path': relative.as_posix(), 'text': text,
                             'split': split, 'is_synthetic': False, 'source_index': i,
                             'dataset_id': str(sample['id']),
                             'original_speaker_id': sample.get('original_speaker_id'),
                             'language': sample.get('language'), **info})
                if (i + 1) % 250 == 0 or i + 1 == len(ds[split]):
                    print(f'{split}: {i + 1}/{len(ds[split])}', flush=True)
            real[split] = rows
            write_jsonl(out / f'{split}_real.jsonl', rows)
        synthetic, decisions = [], []
        for row in pilot:
            sid = row['synthetic_id']
            if sid in EXCLUDED:
                note = 'User reported listed English word slightly unclear; excluded provisionally, not confirmed incorrect.'
            elif sid in CONFIRMED:
                note = 'User explicitly reported queried English portion clear; full utterance not verified.'
            elif sid in IMPLIED:
                note = 'No issue reported when user singled out 0065/0097; NOT explicit full-utterance approval.'
            else:
                note = 'No individual human listening result recorded.'
            decisions.append({'synthetic_id': sid, 'include_in_pilot': sid not in EXCLUDED,
                              'human_review_note': note, 'full_utterance_verified': False,
                              'automatic_mer_percent': float(review[sid]['mixed_error_rate_percent'])})
            if sid in EXCLUDED:
                continue
            relative = Path('audio') / 'synthetic' / (sid + '.wav')
            info = export_audio(str(run / row['audio_path']), out / relative)
            synthetic.append({'id': sid, 'audio_path': relative.as_posix(), 'text': row['text'],
                              'split': 'train', 'is_synthetic': True,
                              'source_id': row['source_id'], 'source_index': row['source_index'],
                              'quality_status': 'provisional_not_fully_verified', **info})
        write_jsonl(out / 'synthetic_review_decisions.jsonl', decisions)
        write_jsonl(out / 'train_synthetic.jsonl', synthetic)
        write_jsonl(out / 'train_augmented.jsonl', real['train'] + synthetic)
        status.update(status='complete', counts={
            'train_real': len(real['train']), 'train_synthetic': len(synthetic),
            'train_augmented': len(real['train']) + len(synthetic),
            'validation_real': len(real['validation']), 'test_real': len(real['test'])},
            excluded_synthetic_ids=sorted(EXCLUDED),
            over_30_seconds={k: sum(r['duration_seconds'] > 30 for r in v) for k,v in real.items()},
            note='Any >30s real rows are preserved and counted; Step 9 must handle them explicitly, never silently truncate labels.')
        write_json(out / 'preparation_summary.json', status)
        print('\nSTEP 8B COMPLETE', flush=True)
        print(json.dumps(status['counts'], indent=2))
        print('Output:', out)
        print('Share preparation_summary.json. No training started; original files unchanged.')
    except Exception as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(out / 'preparation_summary.json', status)
        print('FAILED; do not train from this incomplete output:', out, flush=True)
        raise

if __name__ == '__main__':
    main()
