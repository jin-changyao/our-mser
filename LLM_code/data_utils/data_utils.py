import pandas as pd
import logging
import os
from os import truncate
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import json
from dataclasses import dataclass, asdict
from multiprocessing import Pool
import multiprocessing
import math
from random import sample
from pathlib import Path
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    Wav2Vec2FeatureExtractor
)
import numpy as np
import opensmile
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)
AUDIO_MAX_LEN = 16000*6

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_split_from_file_name(file_name):
    stem = Path(file_name).stem.lower()
    if "train" in stem:
        return "train"
    if "valid" in stem or "dev" in stem:
        return "valid"
    if "test" in stem:
        return "test"
    return ""


def manifest_key_from_path(dataset, split, path):
    stem = Path(str(path)).stem
    if stem.startswith(("train_", "val_", "test_", "Ses")):
        return stem
    if dataset == "iemocap":
        return stem
    if dataset == "meld":
        split_prefix = {"train": "train", "valid": "val", "dev": "val", "test": "test"}.get(split, split)
        return f"{split_prefix}_{stem}"
    return stem


def load_multimodal_manifest(manifest_dir, dataset, split):
    if not manifest_dir:
        return {}
    split_name = "valid" if split == "dev" else split
    manifest_path = Path(manifest_dir) / f"{dataset}_multimodal_{split_name}.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Multimodal manifest not found: {manifest_path}")
    rows = read_jsonl(manifest_path)
    return {row["utterance_id"]: row for row in rows}


def infer_feature_root_from_manifest_dir(manifest_dir):
    if not manifest_dir:
        return None
    path = Path(manifest_dir)
    if path.name.startswith("splits_"):
        return path.parent
    return path


def fallback_feature_path(manifest_dir, feature_dir, utterance_id):
    feature_root = infer_feature_root_from_manifest_dir(manifest_dir)
    if feature_root is None:
        return ""
    path = feature_root / feature_dir / f"{utterance_id}.npy"
    return str(path) if path.is_file() else ""


def read_data(file_name, percent, random_seed, args=None):
    f = open(file_name, 'r', encoding='utf-8').readlines()
    data = [json.loads(d) for d in f]
    use_mm_prefix = bool(args is not None and getattr(args, "use_mm_prefix", False))
    skip_missing_mm = bool(args is not None and getattr(args, "skip_missing_mm", False))
    split = infer_split_from_file_name(file_name)
    manifest_by_id = {}
    if use_mm_prefix:
        manifest_by_id = load_multimodal_manifest(
            getattr(args, "multimodal_manifest_dir", ""),
            getattr(args, "dataset", ""),
            split,
        )

    inputs = []
    targets = []
    paths = []
    target_utterances = []
    audio_feature_paths = []
    video_feature_paths = []
    missing_manifest = []
    skipped_missing_mm = []
    for index, d in enumerate(data):
        if pd.isnull(d['target']) or pd.isna(d['target']):
            continue
        audio_path = ""
        video_path = ""
        if use_mm_prefix:
            utterance_id = d.get("utterance_id") or manifest_key_from_path(getattr(args, "dataset", ""), split, d["path"])
            manifest_row = manifest_by_id.get(utterance_id)
            if manifest_row is None:
                audio_path = fallback_feature_path(
                    args.multimodal_manifest_dir,
                    args.mm_audio_feature_dir,
                    utterance_id,
                )
                video_path = fallback_feature_path(
                    args.multimodal_manifest_dir,
                    args.mm_video_feature_dir,
                    utterance_id,
                )
            else:
                audio_path = manifest_row.get(f"feature_{args.mm_audio_feature_dir}", "")
                video_path = manifest_row.get(f"feature_{args.mm_video_feature_dir}", "")
            if not audio_path or not video_path:
                missing_manifest.append(utterance_id)
                if skip_missing_mm:
                    skipped_missing_mm.append(utterance_id)
                    continue
        inputs.append(d['input'])
        targets.append(d['target'])
        paths.append(d['path'])
        target_utterances.append(d.get("target_utterance") or extract_target_utterance(d.get("input", "")))
        if use_mm_prefix:
            audio_feature_paths.append(audio_path)
            video_feature_paths.append(video_path)
    if missing_manifest:
        if skip_missing_mm:
            print(
                f"Skipped {len(skipped_missing_mm)} rows with missing multimodal features for {file_name}. "
                f"Examples: {skipped_missing_mm[:10]}"
            )
        else:
            raise ValueError(
                f"Missing {len(missing_manifest)} multimodal manifest rows/features for {file_name}. "
                f"Examples: {missing_manifest[:10]}. "
                "Set SKIP_MISSING_MM=True to drop these rows for multimodal-prefix runs."
            )
    dict_ = {'input': inputs, 'output': targets, 'path': paths, "target_utterance": target_utterances}
    if use_mm_prefix:
        dict_["audio_feature_path"] = audio_feature_paths
        dict_["video_feature_path"] = video_feature_paths
    df_data = pd.DataFrame(dict_)
    df_data.dropna(axis=0, how='any')

    # randomly extract *percent of the data
    num_samples = int(len(df_data)*percent)
    print(f'the number of num_samples is {len(df_data)}')
    df_data = df_data.sample(n=num_samples, random_state=random_seed)
    print(f'the number of num_samples is {len(df_data)}')

    return df_data


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False
    
def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

class Seq2SeqDataset(Dataset):
    def __init__(self, args, data, mode):
        inputs = list(data["input"])
        outputs = list(data['output'])
        paths = list(data['path'])
        target_utterances = list(data.get("target_utterance", [""] * len(inputs)))
        self.use_mm_prefix = getattr(args, "use_mm_prefix", False)
        if self.use_mm_prefix:
            audio_feature_paths = list(data["audio_feature_path"])
            video_feature_paths = list(data["video_feature_path"])
            self.examples = [
                [i, o, p, t, a, v]
                for i, o, p, t, a, v in zip(inputs, outputs, paths, target_utterances, audio_feature_paths, video_feature_paths)
            ]
        else:
            self.examples = [[i, o, p, t] for i, o, p, t in zip(inputs, outputs, paths, target_utterances)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def render_chat_input(example_input, tokenizer):
    if isinstance(example_input, list):
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                example_input,
                tokenize=False,
                add_generation_prompt=True,
            )
        rendered = []
        for message in example_input:
            rendered.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
        rendered.append('assistant:')
        return '\n'.join(rendered)
    return example_input


def render_chat_inputs(example_inputs, tokenizer):
    return [render_chat_input(example_input, tokenizer) for example_input in example_inputs]


def extract_target_utterance(example_input):
    if isinstance(example_input, list):
        for message in example_input:
            content = message.get("content", "")
            extracted = extract_target_utterance(content)
            if extracted:
                return extracted
        return ""
    text = str(example_input)
    marker = "Target utterance:\n"
    if marker in text:
        target = text.split(marker, 1)[1].split("\n\n", 1)[0].strip()
        return target
    import re
    match = re.search(r"emotional label of <(.+?)>", text)
    if match:
        return match.group(1).strip()
    return ""


class Seq2SeqCollator(object):
    def __init__(self, args, tokenizer, mode="train"):
        self.tokenizer = tokenizer
        self.args = args    
        self.mode = mode
        self.feature = args.feature
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)                                               

    def __call__(self, batch):
        if self.mode == "dev":
            inputs = render_chat_inputs([d[0] for d in batch], self.tokenizer)
            inputs = self.tokenizer(inputs, max_length=self.args.max_length, truncation=True, padding=True, return_tensors='pt')
        else:
            inputs = preprocess_data_batch(batch, self.tokenizer, self.args)
        
        if self.feature == 'text':
            if getattr(self.args, "use_mm_prefix", False):
                inputs["mm_audio_features"] = self.load_numpy_features([d[4] for d in batch])
                inputs["mm_video_features"] = self.load_numpy_features([d[5] for d in batch])
                if getattr(self.args, "text_guided_mm", False):
                    target_text = self.tokenizer(
                        [d[3] for d in batch],
                        max_length=getattr(self.args, "target_text_max_length", 128),
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs["target_text_input_ids"] = target_text["input_ids"]
                    inputs["target_text_attention_mask"] = target_text["attention_mask"]
            return inputs
        
        paths = [d[2] for d in batch]
        audio_features = []
        audio_masks = []
        for path in paths:
            # load audio
            try:
                sound, _ = torchaudio.load(path)
                soundData = torch.mean(sound, dim=0, keepdim=False)
                # extract audio features
                features = self.feature_extractor(soundData, sampling_rate=16000, return_tensors="pt", padding="max_length",
                                                max_length=AUDIO_MAX_LEN, return_attention_mask=True, truncation=True)
                audio_feature = features['input_values']
                audio_mask = features['attention_mask']
                audio_features.append(audio_feature)
                audio_masks.append(audio_mask)
            except:
                print(path)
        # inputs = {}
        inputs['audio_features'] = torch.cat(audio_features, dim=0)
        inputs['audio_masks'] = torch.cat(audio_masks, dim=0)
        
        if inputs['audio_features'].dim() == 1:
            inputs['audio_features'] = inputs['audio_features'].unsqueeze(0)
        if inputs['audio_masks'].dim() == 1:
            inputs['audio_masks'] = inputs['audio_masks'].unsqueeze(0)
        
        return inputs

    def load_numpy_features(self, paths):
        features = []
        for path in paths:
            if not path:
                raise FileNotFoundError("Empty multimodal feature path.")
            array = np.load(path)
            if array.ndim > 1:
                array = array.mean(axis=0)
            features.append(torch.tensor(array, dtype=torch.float32))
        return torch.stack(features, dim=0)


def preprocess_data_batch(data, tokenizer, args):
    
    inputs = render_chat_inputs([d[0] for d in data], tokenizer)
    inputs_pred = None
    targets = [d[1] for d in data]

    if args.model_type == "decoder":
        if args.mode == "pretrain":
            inputs = tokenizer(
                inputs,
                max_length=args.max_seq_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            labels = inputs['input_ids'].clone().contiguous()
            labels[labels[:, :] == tokenizer.pad_token_id] = -100
            type_token_ids = inputs['attention_mask'].long()
            inputs['labels'] = labels
            inputs["type_token_ids"] = type_token_ids
            return inputs
            
        # decoder-only model
        inputs = tokenizer(
            inputs,
            max_length=args.max_length - 1,
            truncation=True
        )

        targets = tokenizer(
            targets,
            add_special_tokens=False,
        )
        input_ids = inputs['input_ids']
        target_ids = targets['input_ids']
        concat_input = [input_ids[i] + target_ids[i] for i in range(len(input_ids))]
        concat_input = [c_[: args.max_length] for c_ in concat_input]
        if not args.open_ended:
            concat_input = [c_ids + [tokenizer.eos_token_id] for c_ids in concat_input]

        type_token_ids = [[0] * len(input_ids[i]) + [1] * (len(concat_input[i]) - len(input_ids[i])) for i in range(len(input_ids))]
        attention_mask = [[1] * len(concat_input[i]) for i in range(len(input_ids))]
        
        max_batch_length = 0
        for i in range(len(input_ids)):
            max_batch_length = max(max_batch_length, len(type_token_ids[i]))

        type_token_ids = [[0] * (max_batch_length - len(ids)) + ids for ids in type_token_ids]
        attention_mask = [[0] * (max_batch_length - len(ids)) + ids for ids in attention_mask]
        concat_input = [[tokenizer.pad_token_id] * (max_batch_length - len(ids)) + ids for ids in concat_input]
        type_token_ids = torch.Tensor(type_token_ids).long()
        attention_mask = torch.Tensor(attention_mask).long()
        concat_input = torch.Tensor(concat_input).long()
        labels = concat_input.clone().contiguous()
        labels[type_token_ids[:, :] == 0] = -100
                   
        return {
            "input_ids": concat_input,
            "attention_mask": attention_mask,
            "type_token_ids": type_token_ids,
            "labels": labels,
            "tradoff": args.beta
        }
    
@dataclass
class ModelArgs:
    model_type: str = "decoder"
    model_name_or_path: str = "YOUR_MODEL_PATH"
    checkpoint_dir: str = None
    output_dir: str = "YOUR_OUTPUT_DIR_PATH"
    data_dir: str = "DATASET_PATH"
    deepspeed_config = "./deepspeed_config.json"
    do_train: bool = True
    do_eval: bool = False
    num_train_epochs = 10
    warmup_ratio: float = 0.1
    warmup_steps: int = None
    save_steps: int = 500
    weight_decay: float = 0.0
    max_seq_length: int = 96
    max_length: int = 32
    max_new_tokens: int = 10
    num_beams: int = 1
    do_sample: bool = False
    top_k: int = None
    top_p: float = None
    learning_rate: float = 3e-5
    preprocess_inputs: bool = True
    clip_norm: float = 1.0
    open_ended: bool = False
    batch_size: int = 32
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora: bool = True
    lora_dim: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_module_name: str = 'q_proj,k_proj,v_proj,query_key_value'
    seed: int = 42
    offload_optimizer: bool = False
    deepspeed_config: str = None
    zero_shot: bool = False
    mode: str = "sft"
    gradient_checkpointing: bool = False
    use_mm_prefix: bool = False
    multimodal_manifest_dir: str = ""
    mm_audio_feature_dir: str = "chinese-hubert-large-UTT"
    mm_video_feature_dir: str = "clip-vit-large-patch14-UTT"
    mm_audio_dim: int = 1024
    mm_video_dim: int = 768
    mm_audio_tokens: int = 4
    mm_video_tokens: int = 4
    mm_projector_dropout: float = 0.05
    skip_missing_mm: bool = False
    mm_hidden_size: int = 3584
    text_guided_mm: bool = False
    text_guide_source: str = "target_text"
    text_guided_mode: str = "film_gate"
    text_guided_audio: bool = True
    text_guided_video: bool = True
    log_mm_gates: bool = False
    target_text_max_length: int = 128

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            f.write(json.dumps(asdict(self), indent=5))

    def update(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))
