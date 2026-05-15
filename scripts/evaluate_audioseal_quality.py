#!/usr/bin/env python3
"""Evaluate AudioSeal watermark quality impact.

This script compares three conditions on 24 kHz speech audio:
- original
- audio resample-only roundtrip (24k -> 16k -> 24k)
- current watermark path via AudioSeal

It evaluates external clean speech from Hugging Face datasets and, when
available, locally generated *clean* KugelAudio samples that were saved before
watermarking. Public sample WAVs from the model repo are intentionally not used
as baselines because they are already likely watermarked.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import librosa
import numpy as np
import requests
import soundfile as sf
from scipy import signal

from kugelaudio_open.watermark import AudioWatermark

OUTPUT_DIR = Path("artifacts/audioseal_quality")
AUDIO_DIR = OUTPUT_DIR / "audio"
REPORT_PATH = OUTPUT_DIR / "report.md"
JSON_PATH = OUTPUT_DIR / "results.json"
CSV_PATH = OUTPUT_DIR / "results.csv"
TARGET_SR = 24000
RESAMPLE_SR = 16000
HIGH_BAND_LOW = 8000
HIGH_BAND_HIGH = 12000
HIGH_BAND_MIN_RATIO = 1e-4
HF_TIMEOUT = 60


@dataclass
class SampleResult:
    sample_id: str
    kind: str
    transcript: str
    sample_rate: int
    duration_s: float
    high_band_ratio: float
    high_band_db: float
    high_band_pass: bool
    resample_snr_db: float
    watermark_snr_db: float
    snr_gap_db: float
    resample_lsd_db: float
    watermark_lsd_db: float
    lsd_gap_db: float
    resample_high_band_lsd_db: float
    watermark_high_band_lsd_db: float
    high_band_lsd_gap_db: float
    resample_rms_delta_db: float
    watermark_rms_delta_db: float
    resample_peak_abs_delta: float
    watermark_peak_abs_delta: float


def fetch_libritts_samples(limit: int = 4) -> list[dict]:
    response = requests.get(
        "https://datasets-server.huggingface.co/first-rows",
        params={
            "dataset": "parler-tts/libritts_r_filtered",
            "config": "clean",
            "split": "test.clean",
        },
        timeout=HF_TIMEOUT,
    )
    response.raise_for_status()
    rows = response.json()["rows"][:limit]
    out = []
    for idx, row in enumerate(rows):
        audio_src = row["row"]["audio"][0]["src"]
        text = row["row"]["text_normalized"]
        speaker_id = row["row"]["speaker_id"]
        out.append(
            {
                "id": f"libritts_{idx}_{speaker_id}",
                "kind": "external",
                "url": audio_src,
                "text": text,
            }
        )
    return out


def load_audio_url(url: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    response = requests.get(url, timeout=HF_TIMEOUT)
    response.raise_for_status()
    return _load_audio_bytes(response.content, target_sr=target_sr)


def load_audio_path(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=False)
    return _normalize_loaded_audio(audio, sr, target_sr=target_sr)


def _load_audio_bytes(data: bytes, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(data), always_2d=False)
    return _normalize_loaded_audio(audio, sr, target_sr=target_sr)


def _normalize_loaded_audio(audio: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak * 0.95
    return audio.astype(np.float32), sr


def collect_local_generated_samples(generated_dir: Optional[Path]) -> list[dict]:
    if generated_dir is None or not generated_dir.exists():
        return []
    samples = []
    for path in sorted(generated_dir.glob("*.wav")):
        samples.append(
            {
                "id": f"generated_{path.stem}",
                "kind": "generated",
                "path": str(path),
                "text": path.stem.replace("_", " "),
            }
        )
    return samples


def resample_roundtrip(audio: np.ndarray, sr: int) -> np.ndarray:
    down = librosa.resample(audio, orig_sr=sr, target_sr=RESAMPLE_SR)
    up = librosa.resample(down, orig_sr=RESAMPLE_SR, target_sr=sr)
    return _align_length(audio, up)


def high_band_ratio(audio: np.ndarray, sr: int) -> tuple[float, float]:
    freqs, psd = signal.welch(audio, fs=sr, nperseg=min(4096, len(audio)))
    total = float(np.sum(psd) + 1e-12)
    mask = (freqs >= HIGH_BAND_LOW) & (freqs <= min(HIGH_BAND_HIGH, sr / 2 - 1))
    band = float(np.sum(psd[mask]) + 1e-12)
    ratio = band / total
    db = 10.0 * math.log10(ratio)
    return ratio, db


def snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference, candidate = _same_length(reference, candidate)
    noise = reference - candidate
    signal_power = float(np.mean(reference**2) + 1e-12)
    noise_power = float(np.mean(noise**2) + 1e-12)
    return 10.0 * math.log10(signal_power / noise_power)


def rms_delta_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference, candidate = _same_length(reference, candidate)
    ref_rms = math.sqrt(float(np.mean(reference**2) + 1e-12))
    cand_rms = math.sqrt(float(np.mean(candidate**2) + 1e-12))
    return 20.0 * math.log10(cand_rms / ref_rms)


def peak_abs_delta(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference, candidate = _same_length(reference, candidate)
    return float(np.max(np.abs(reference - candidate)))


def log_spectral_distance_db(reference: np.ndarray, candidate: np.ndarray, sr: int) -> float:
    reference, candidate = _same_length(reference, candidate)
    ref_stft = np.abs(librosa.stft(reference, n_fft=1024, hop_length=256))
    cand_stft = np.abs(librosa.stft(candidate, n_fft=1024, hop_length=256))
    ref_log = np.log10(np.maximum(ref_stft, 1e-7))
    cand_log = np.log10(np.maximum(cand_stft, 1e-7))
    return float(np.mean(np.sqrt(np.mean((ref_log - cand_log) ** 2, axis=0))))


def high_band_lsd_db(reference: np.ndarray, candidate: np.ndarray, sr: int) -> float:
    reference, candidate = _same_length(reference, candidate)
    ref_stft = np.abs(librosa.stft(reference, n_fft=1024, hop_length=256))
    cand_stft = np.abs(librosa.stft(candidate, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    mask = (freqs >= HIGH_BAND_LOW) & (freqs <= min(HIGH_BAND_HIGH, sr / 2 - 1))
    if not np.any(mask):
        return 0.0
    ref_log = np.log10(np.maximum(ref_stft[mask], 1e-7))
    cand_log = np.log10(np.maximum(cand_stft[mask], 1e-7))
    return float(np.mean(np.sqrt(np.mean((ref_log - cand_log) ** 2, axis=0))))


def _same_length(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    min_len = min(len(reference), len(candidate))
    return reference[:min_len], candidate[:min_len]


def _align_length(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if len(candidate) < len(reference):
        candidate = np.pad(candidate, (0, len(reference) - len(candidate)))
    return candidate[: len(reference)]


def save_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


def summarize(results: Iterable[SampleResult]) -> dict[str, dict[str, float]]:
    all_results = list(results)
    grouped: dict[str, list[SampleResult]] = {}
    for result in all_results:
        grouped.setdefault(result.kind, []).append(result)
    grouped["all"] = all_results
    grouped["high_band_pass"] = [result for result in all_results if result.high_band_pass]

    summary: dict[str, dict[str, float]] = {}
    for key, items in grouped.items():
        if not items:
            continue
        summary[key] = {
            "count": float(len(items)),
            "mean_high_band_ratio": float(np.mean([x.high_band_ratio for x in items])),
            "mean_resample_snr_db": float(np.mean([x.resample_snr_db for x in items])),
            "mean_watermark_snr_db": float(np.mean([x.watermark_snr_db for x in items])),
            "mean_snr_gap_db": float(np.mean([x.snr_gap_db for x in items])),
            "mean_resample_lsd_db": float(np.mean([x.resample_lsd_db for x in items])),
            "mean_watermark_lsd_db": float(np.mean([x.watermark_lsd_db for x in items])),
            "mean_lsd_gap_db": float(np.mean([x.lsd_gap_db for x in items])),
            "mean_resample_high_band_lsd_db": float(np.mean([x.resample_high_band_lsd_db for x in items])),
            "mean_watermark_high_band_lsd_db": float(np.mean([x.watermark_high_band_lsd_db for x in items])),
            "mean_high_band_lsd_gap_db": float(np.mean([x.high_band_lsd_gap_db for x in items])),
            "mean_resample_peak_abs_delta": float(np.mean([x.resample_peak_abs_delta for x in items])),
            "mean_watermark_peak_abs_delta": float(np.mean([x.watermark_peak_abs_delta for x in items])),
        }
    return summary


def write_csv(results: list[SampleResult]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()) if results else list(SampleResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_report(results: list[SampleResult], summary: dict[str, dict[str, float]], generated_count: int) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AudioSeal quality investigation",
        "",
        "This report compares three conditions:",
        "- original 24 kHz speech",
        "- resample-only roundtrip (24k -> 16k -> 24k)",
        "- current AudioSeal watermark path",
        "",
        "The key hypothesis under test is whether resampling hurts fidelity as much as or more than watermarking itself.",
        "",
        "## Baseline selection note",
        "",
        "Public sample WAVs from the KugelAudio model repo were intentionally excluded as clean baselines because they are already likely watermarked.",
        f"Generated clean baselines found locally: {generated_count}",
        "",
        "## High-band sanity check",
        "",
        f"A sample is marked `high_band_pass` when its 8-12 kHz energy ratio exceeds {HIGH_BAND_MIN_RATIO:.1e}. This is the band most obviously affected by a 24k -> 16k -> 24k bottleneck.",
        "",
        "## Aggregate summary",
        "",
    ]
    for group, metrics in summary.items():
        lines.extend(
            [
                f"### {group}",
                "",
                f"- count: {int(metrics['count'])}",
                f"- mean high-band ratio: {metrics['mean_high_band_ratio']:.6f}",
                f"- mean resample-only SNR (dB): {metrics['mean_resample_snr_db']:.2f}",
                f"- mean watermark SNR (dB): {metrics['mean_watermark_snr_db']:.2f}",
                f"- mean SNR gap (watermark - resample, dB): {metrics['mean_snr_gap_db']:.2f}",
                f"- mean resample-only LSD: {metrics['mean_resample_lsd_db']:.4f}",
                f"- mean watermark LSD: {metrics['mean_watermark_lsd_db']:.4f}",
                f"- mean LSD gap (watermark - resample): {metrics['mean_lsd_gap_db']:.4f}",
                f"- mean resample-only high-band LSD: {metrics['mean_resample_high_band_lsd_db']:.4f}",
                f"- mean watermark high-band LSD: {metrics['mean_watermark_high_band_lsd_db']:.4f}",
                f"- mean high-band LSD gap (watermark - resample): {metrics['mean_high_band_lsd_gap_db']:.4f}",
                f"- mean resample-only peak abs delta: {metrics['mean_resample_peak_abs_delta']:.6f}",
                f"- mean watermark peak abs delta: {metrics['mean_watermark_peak_abs_delta']:.6f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Per-sample results",
            "",
            "| sample | kind | high-band ratio | pass | resample SNR | watermark SNR | SNR gap | resample LSD | watermark LSD |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        lines.append(
            f"| {result.sample_id} | {result.kind} | {result.high_band_ratio:.6f} | {result.high_band_pass} | {result.resample_snr_db:.2f} | {result.watermark_snr_db:.2f} | {result.snr_gap_db:.2f} | {result.resample_lsd_db:.4f} | {result.watermark_lsd_db:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Subjective listening protocol",
            "",
            "Audio triplets are saved under `artifacts/audioseal_quality/audio/`.",
            "For each sample, compare:",
            "- `_original.wav`",
            "- `_resample_only.wav`",
            "- `_watermarked.wav`",
            "",
            "Focus on:",
            "- fricatives / sibilants",
            "- tail clarity on final phones",
            "- high-frequency brightness",
            "- low-level hiss or warble",
            "",
            "A useful blind test is to shuffle filenames before listening and rank perceived degradation.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path("artifacts/clean_generated_samples"),
        help="Directory containing locally generated clean KugelAudio WAVs (watermark disabled)",
    )
    parser.add_argument(
        "--external-limit",
        type=int,
        default=4,
        help="Number of external LibriTTS samples to fetch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    watermark = AudioWatermark(device="cpu")
    samples = fetch_libritts_samples(limit=args.external_limit) + collect_local_generated_samples(args.generated_dir)

    results: list[SampleResult] = []
    for sample in samples:
        if "url" in sample:
            audio, sr = load_audio_url(sample["url"])
        else:
            audio, sr = load_audio_path(Path(sample["path"]))

        resample_only = resample_roundtrip(audio, sr)
        watermarked = np.asarray(watermark.embed(audio, sample_rate=sr), dtype=np.float32)
        watermarked = _align_length(audio, watermarked)

        ratio, ratio_db = high_band_ratio(audio, sr)
        resample_snr = snr_db(audio, resample_only)
        watermark_snr = snr_db(audio, watermarked)
        resample_lsd = log_spectral_distance_db(audio, resample_only, sr)
        watermark_lsd = log_spectral_distance_db(audio, watermarked, sr)
        resample_hb_lsd = high_band_lsd_db(audio, resample_only, sr)
        watermark_hb_lsd = high_band_lsd_db(audio, watermarked, sr)

        result = SampleResult(
            sample_id=sample["id"],
            kind=sample["kind"],
            transcript=sample["text"],
            sample_rate=sr,
            duration_s=len(audio) / sr,
            high_band_ratio=ratio,
            high_band_db=ratio_db,
            high_band_pass=ratio >= HIGH_BAND_MIN_RATIO,
            resample_snr_db=resample_snr,
            watermark_snr_db=watermark_snr,
            snr_gap_db=watermark_snr - resample_snr,
            resample_lsd_db=resample_lsd,
            watermark_lsd_db=watermark_lsd,
            lsd_gap_db=watermark_lsd - resample_lsd,
            resample_high_band_lsd_db=resample_hb_lsd,
            watermark_high_band_lsd_db=watermark_hb_lsd,
            high_band_lsd_gap_db=watermark_hb_lsd - resample_hb_lsd,
            resample_rms_delta_db=rms_delta_db(audio, resample_only),
            watermark_rms_delta_db=rms_delta_db(audio, watermarked),
            resample_peak_abs_delta=peak_abs_delta(audio, resample_only),
            watermark_peak_abs_delta=peak_abs_delta(audio, watermarked),
        )
        results.append(result)

        save_audio(AUDIO_DIR / f"{sample['id']}_original.wav", audio, sr)
        save_audio(AUDIO_DIR / f"{sample['id']}_resample_only.wav", resample_only, sr)
        save_audio(AUDIO_DIR / f"{sample['id']}_watermarked.wav", watermarked, sr)

    summary = summarize(results)
    JSON_PATH.write_text(json.dumps({"results": [asdict(x) for x in results], "summary": summary}, indent=2))
    write_csv(results)
    write_report(results, summary, generated_count=len([x for x in samples if x['kind'] == 'generated']))
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
