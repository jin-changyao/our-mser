# MELD Omni Caption Workflow

This workflow creates a reusable MELD caption dataset with:

- `text`: original MELD utterance text from the CSV files
- `audio_caption`: Qwen2.5-Omni description from wav-only input
- `video_caption`: Qwen2.5-Omni description from video-only input

## 1. Check Or Prepare MELD Raw Data

Expected raw archive:

```bash
/home/pc/jcy/MELD_raw/MELD.Raw.tar.gz
```

Expected extracted directory:

```bash
/home/pc/jcy/MELD_raw/MELD.Raw
```

Check:

```bash
ls -lh /home/pc/jcy/MELD_raw
ls -lh /home/pc/jcy/MELD_raw/MELD.Raw
find /home/pc/jcy/MELD_raw/MELD.Raw -maxdepth 2 -type f \( -name "dia0_utt0.mp4" -o -name "train_sent_emo.csv" \)
```

If `MELD.Raw` is missing but the archive exists:

```bash
cd /home/pc/jcy/MELD_raw
tar -xzf MELD.Raw.tar.gz

cd /home/pc/jcy/MELD_raw/MELD.Raw
tar -xzf train.tar.gz
tar -xzf dev.tar.gz
tar -xzf test.tar.gz
find . -name "._*" -delete
```

Or let the project script check and extract whatever is missing:

```bash
cd /home/pc/jcy/Our-MSER/LLM_code
bash prepare_meld_raw_data.sh /home/pc/jcy/MELD_raw
```

## 2. Build The Manifest

```bash
cd /home/pc/jcy/Our-MSER/LLM_code

python build_meld_caption_manifest.py \
  --raw_root /home/pc/jcy/MELD_raw/MELD.Raw \
  --out ../PROCESSED_DATASET/meld/omni_caption/meld_raw_manifest.jsonl
```

A correct report should have about 13k rows and `missing_video: 0`.

## 3. Extract Wav Files

```bash
python extract_meld_audio.py \
  --manifest ../PROCESSED_DATASET/meld/omni_caption/meld_raw_manifest.jsonl \
  --num_workers 4
```

For a smoke test:

```bash
python extract_meld_audio.py \
  --manifest ../PROCESSED_DATASET/meld/omni_caption/meld_raw_manifest.jsonl \
  --limit 20 \
  --num_workers 2
```

## 4. Generate Captions

The Omni environment must support Qwen2.5-Omni. If needed, install the model-specific dependencies on the server environment:

```bash
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
pip install "qwen-omni-utils[decord]" accelerate soundfile
```

Then run a small caption smoke test first:

```bash
python generate_meld_omni_captions.py \
  --manifest ../PROCESSED_DATASET/meld/omni_caption/meld_raw_manifest.jsonl \
  --out ../PROCESSED_DATASET/meld/omni_caption/meld_omni_captions.jsonl \
  --model_path /home/pc/jcy/models/Qwen2.5-Omni-7B \
  --limit 20 \
  --caption_type both \
  --attn_implementation flash_attention_2
```

If flash attention is unavailable, omit `--attn_implementation flash_attention_2`.

For full generation:

```bash
python generate_meld_omni_captions.py \
  --manifest ../PROCESSED_DATASET/meld/omni_caption/meld_raw_manifest.jsonl \
  --out ../PROCESSED_DATASET/meld/omni_caption/meld_omni_captions.jsonl \
  --model_path /home/pc/jcy/models/Qwen2.5-Omni-7B \
  --caption_type both \
  --attn_implementation flash_attention_2
```

The script appends one JSON line per sample and skips completed rows on rerun.

## 5. Validate

```bash
python validate_meld_omni_captions.py \
  --manifest ../PROCESSED_DATASET/meld/omni_caption/meld_raw_manifest.jsonl \
  --captions ../PROCESSED_DATASET/meld/omni_caption/meld_omni_captions.jsonl
```

The first smoke test should show:

- no missing media files for the selected samples
- no error captions
- non-empty `audio_caption`
- non-empty `video_caption`

Review `label_word_hits_for_review`; this is not always fatal, but many direct label words usually mean the prompt should be tightened.

