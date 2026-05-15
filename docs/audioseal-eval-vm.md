# AudioSeal quality evaluation on a stronger VM

This runbook prepares a remote workflow for investigating whether AudioSeal post-processing degrades audio quality, and specifically whether the mandatory 24k -> 16k -> 24k path harms fidelity as much as or more than watermarking itself.

## Overview

The workflow has two stages:

1. **Generate clean KugelAudio speech** on a VM with enough RAM/VRAM, with watermarking disabled.
2. **Evaluate quality impact** by comparing:
   - original clean speech
   - resample-only roundtrip
   - full AudioSeal watermark path

The evaluation also pulls a small set of clean external speech samples from Hugging Face for comparison.

## Important baseline note

Do **not** use the public sample WAVs from the `kugelaudio/kugelaudio-0-open` model repo as clean baselines. Those files are likely already watermarked.

Use only:
- external clean speech fetched by the evaluation script
- locally generated clean KugelAudio WAVs created with `do_watermark=False`

## VM prerequisites

Recommended:
- Linux VM
- CUDA GPU with enough VRAM for `kugelaudio/kugelaudio-0-open`
- enough system RAM to load the 7B model
- network access to Hugging Face

Project setup:

```bash
uv sync --extra cuda
```

Optional syntax-aware chunking extra is not required for this evaluation.

## Step 1: Generate clean KugelAudio samples

Use the provided manifest:

```bash
python scripts/generate_clean_eval_samples.py \
  --manifest manifests/audioseal_eval_samples.json \
  --output-dir artifacts/clean_generated_samples
```

This generates clean WAVs with:
- `do_watermark=False`
- optional per-sample language hints from the manifest

Outputs:
- `artifacts/clean_generated_samples/*.wav`
- `artifacts/clean_generated_samples/manifest_outputs.json`

## Step 2: Run AudioSeal evaluation

```bash
python scripts/evaluate_audioseal_quality.py \
  --generated-dir artifacts/clean_generated_samples \
  --external-limit 4
```

This will:
- fetch external clean speech from `parler-tts/libritts_r_filtered`
- compute resample-only and full watermark-path comparisons
- run the 8-12 kHz high-band sanity check
- export listening triplets

Outputs:
- `artifacts/audioseal_quality/report.md`
- `artifacts/audioseal_quality/results.json`
- `artifacts/audioseal_quality/results.csv`
- `artifacts/audioseal_quality/audio/*_{original,resample_only,watermarked}.wav`

## What to copy back from the VM

At minimum:
- `artifacts/audioseal_quality/report.md`
- `artifacts/audioseal_quality/results.json`
- `artifacts/audioseal_quality/results.csv`

If subjective listening is needed locally, also copy:
- `artifacts/audioseal_quality/audio/`

## Subjective listening protocol

For each sample, compare:
- original
- resample-only
- watermarked

Listen specifically for:
- loss of brightness / air
- sibilant damage (`s`, `sh`, `f`)
- final phone truncation
- low-level hiss or warble

A useful blind procedure:
- randomize filenames before listening
- rank degradation severity
- note whether resample-only already explains most of the perceived change

## Interpreting the hypothesis

The user hypothesis is that the 24k -> 16k -> 24k resampling path may degrade fidelity as much as or more than watermarking itself.

The most relevant columns are:
- `resample_snr_db` vs `watermark_snr_db`
- `resample_lsd_db` vs `watermark_lsd_db`
- `resample_high_band_lsd_db` vs `watermark_high_band_lsd_db`
- `snr_gap_db`
- `lsd_gap_db`
- `high_band_lsd_gap_db`

Heuristic reading:
- if resample-only and watermark numbers are very close, resampling likely dominates
- if watermark is substantially worse than resample-only, watermark embedding adds meaningful extra harm beyond the resampling bottleneck
- pay special attention to samples with `high_band_pass=true`, because those actually contain enough 8-12 kHz signal for the bottleneck to matter perceptually
