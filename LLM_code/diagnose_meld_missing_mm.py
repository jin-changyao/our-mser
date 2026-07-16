import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path


STANDARD_LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]


def read_jsonl(path):
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv_by_filename(path):
    rows = {}
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[row.get("filename", "")] = row
    return rows


def clean_text(text):
    text = str(text or "")
    replacements = {
        "\x92": "'",
        "\u805e": "'",
        "\u806d": "",
        "\u806f": "",
        "\u8070": "",
        "\u8074": " ",
        "\u805f": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def normalize_text(text):
    text = clean_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()


def infer_split(path):
    stem = path.stem.lower()
    if "train" in stem:
        return "train"
    if "valid" in stem or "dev" in stem:
        return "valid"
    return "test"


def parse_meld_utterance_id(utterance_id):
    match = re.match(r"^(train|val|test)_dia(\d+)_utt(\d+)$", utterance_id)
    if not match:
        return None
    split_prefix, dia, utt = match.groups()
    split = "valid" if split_prefix == "val" else split_prefix
    offset = {"train": 0, "valid": 1039, "test": 1153}[split]
    return {
        "split": split,
        "split_prefix": split_prefix,
        "local_dialog_id": int(dia),
        "global_dialog_id": offset + int(dia),
        "turn_id": int(utt),
        "filename": f"dia{dia}_utt{utt}.wav",
    }


def utterance_id_from_legacy_item(item, split):
    utterance_id = item.get("utterance_id", "")
    if utterance_id:
        return utterance_id
    stem = Path(str(item.get("path", ""))).stem
    if stem.startswith(("train_", "val_", "test_")):
        return stem
    prefix = {"train": "train", "valid": "val", "test": "test"}[split]
    return f"{prefix}_{stem}" if stem else ""


def load_manifest_indexes(manifest_dir, split):
    if not manifest_dir:
        return {}, {}
    path = Path(manifest_dir) / f"meld_multimodal_{split}.jsonl"
    by_id = {}
    by_old_turn = {}
    for row in read_jsonl(path):
        by_id[row.get("utterance_id", "")] = row
        old_dialog_id = row.get("old_dialog_id", "")
        old_turn_id = row.get("old_turn_id", "")
        if old_dialog_id != "" and old_turn_id != "":
            by_old_turn[(int(old_dialog_id), int(old_turn_id))] = row
    return by_id, by_old_turn


def load_prc_split(prc_data_dir, split):
    if not prc_data_dir:
        return {}
    path = Path(prc_data_dir) / f"meld.{split}.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_old_meld(old_meld_pkl):
    path = Path(old_meld_pkl) if old_meld_pkl else None
    if path is None or not path.is_file():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def get_prc_row(prc_payload, parsed):
    conv = prc_payload.get(str(parsed["global_dialog_id"]))
    if not conv:
        return None
    turn = parsed["turn_id"]
    sentences = conv.get("sentences", [])
    labels = conv.get("labels", [])
    if turn >= len(sentences):
        return None
    return {
        "text": sentences[turn],
        "label_id": labels[turn] if turn < len(labels) else "",
        "label": STANDARD_LABELS[labels[turn]] if turn < len(labels) and isinstance(labels[turn], int) else "",
        "dialog_len": len(sentences),
    }


def get_old_row(old_data, parsed):
    if old_data is None:
        return None
    dialog_id = parsed["global_dialog_id"]
    turn = parsed["turn_id"]
    if dialog_id not in old_data[3] or turn >= len(old_data[3][dialog_id]):
        return None
    label_id = int(old_data[1][dialog_id][turn])
    return {
        "text": old_data[3][dialog_id][turn],
        "label_id": label_id,
        "label": STANDARD_LABELS[label_id],
        "dialog_len": len(old_data[3][dialog_id]),
    }


def feature_exists(row, feature_dirs):
    missing = []
    for feature_dir in feature_dirs:
        path = row.get(f"feature_{feature_dir}", "") if row else ""
        if not path or not Path(path).is_file():
            missing.append(feature_dir)
    return missing


def lookup_manifest_row(manifest_by_id, manifest_by_old_turn, utterance_id, parsed):
    row = manifest_by_id.get(utterance_id)
    if row is not None:
        return row, "utterance_id"
    old_key = (parsed["global_dialog_id"], parsed["turn_id"])
    row = manifest_by_old_turn.get(old_key)
    if row is not None:
        return row, "old_dialog_id/old_turn_id"
    return None, ""


def diagnose(args):
    legacy_dir = Path(args.legacy_data_dir)
    split = args.split
    legacy_rows = read_jsonl(legacy_dir / f"{split}.json")
    manifest, manifest_by_old_turn = load_manifest_indexes(args.manifest_dir, split)
    prc = load_prc_split(args.prc_data_dir, split)
    old = load_old_meld(args.old_meld_pkl)
    speech_rows = read_csv_by_filename(Path(args.speech_feature_dir) / "meld_processed_5class_test_audio_features.csv")
    if split == "train":
        speech_rows = read_csv_by_filename(Path(args.speech_feature_dir) / "meld_processed_5class_train_audio_features.csv")
    elif split == "valid":
        speech_rows = read_csv_by_filename(Path(args.speech_feature_dir) / "meld_processed_5class_dev_audio_features.csv")
    feature_dirs = [item.strip() for item in args.feature_dirs.split(",") if item.strip()]

    missing = []
    for item in legacy_rows:
        utterance_id = utterance_id_from_legacy_item(item, split)
        parsed = parse_meld_utterance_id(utterance_id)
        if not parsed:
            continue
        manifest_row, match_source = lookup_manifest_row(manifest, manifest_by_old_turn, utterance_id, parsed)
        missing_features = feature_exists(manifest_row, feature_dirs) if manifest_row else feature_dirs
        if manifest_row and not missing_features:
            continue
        prc_row = get_prc_row(prc, parsed)
        old_row = get_old_row(old, parsed)
        speech_row = speech_rows.get(parsed["filename"], {})
        target_text = item.get("target_utterance", "")
        missing.append(
            {
                "utterance_id": utterance_id,
                "filename": parsed["filename"],
                "legacy_target": item.get("target", ""),
                "manifest_status": "missing_row" if manifest_row is None else "missing_features",
                "manifest_match_source": match_source,
                "manifest_utterance_id": manifest_row.get("utterance_id", "") if manifest_row else "",
                "missing_feature_dirs": missing_features,
                "legacy_target_utterance": target_text,
                "prc_text": prc_row["text"] if prc_row else "",
                "prc_label": prc_row["label"] if prc_row else "",
                "old_text": old_row["text"] if old_row else "",
                "old_label": old_row["label"] if old_row else "",
                "speech_csv_exists": bool(speech_row),
                "speech_csv_utterance": speech_row.get("Utterance", ""),
                "speech_csv_emotion": speech_row.get("Emotion", ""),
                "speech_csv_has_description": bool(speech_row.get("description", "")),
                "legacy_vs_prc_text": normalize_text(target_text) == normalize_text(prc_row["text"] if prc_row else ""),
                "legacy_vs_old_text": normalize_text(target_text) == normalize_text(old_row["text"] if old_row else ""),
            }
        )

    empty_speech = [
        {
            "filename": name,
            "description": row.get("description", "")[:120],
        }
        for name, row in speech_rows.items()
        if not row.get("Utterance", "").strip()
    ]
    report = {
        "legacy_data_dir": str(legacy_dir),
        "manifest_dir": args.manifest_dir,
        "split": split,
        "legacy_rows": len(legacy_rows),
        "manifest_rows": len(manifest),
        "manifest_old_turn_index_rows": len(manifest_by_old_turn),
        "old_turn_fallback_matches": sum(
            1
            for item in legacy_rows
            if parse_meld_utterance_id(utterance_id_from_legacy_item(item, split))
            and utterance_id_from_legacy_item(item, split) not in manifest
            and (
                (
                    parse_meld_utterance_id(utterance_id_from_legacy_item(item, split))["global_dialog_id"],
                    parse_meld_utterance_id(utterance_id_from_legacy_item(item, split))["turn_id"],
                )
                in manifest_by_old_turn
            )
        ),
        "missing_or_incomplete_mm_rows": len(missing),
        "missing_or_incomplete_examples": missing[: args.sample_limit],
        "speech_csv_empty_utterance_rows": len(empty_speech),
        "speech_csv_empty_utterance_examples": empty_speech[: args.sample_limit],
        "interpretation": (
            "Rows present in PRC/old MELD but missing from manifest or feature files are not absent from the "
            "classification dataset. If speech_csv_utterance is empty for the same ids, the MELD feature/manifest "
            "source likely dropped or failed to annotate these utterances; fix the manifest/features when possible. "
            "Only use SKIP_MISSING_MM=True as an explicitly reported filtered-evaluation fallback."
        ),
    }
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Diagnose MELD legacy rows missing multimodal manifest/features.")
    parser.add_argument("--legacy_data_dir", default="../PROCESSED_DATASET/meld/window/True_False_qwen_chat_prc_featuretext")
    parser.add_argument("--manifest_dir", default="")
    parser.add_argument("--speech_feature_dir", default="../speech_features")
    parser.add_argument("--prc_data_dir", default="../../PRC-Emo-code/data")
    parser.add_argument("--old_meld_pkl", default="../original_data/meld/meld.pkl")
    parser.add_argument("--split", choices=["train", "test", "valid"], default="test")
    parser.add_argument("--feature_dirs", default="chinese-hubert-large-UTT,clip-vit-large-patch14-UTT")
    parser.add_argument("--sample_limit", type=int, default=20)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    report = diagnose(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
