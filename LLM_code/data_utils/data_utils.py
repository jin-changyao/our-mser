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

def read_data(file_name, percent, random_seed):
    f = open(file_name, 'r', encoding='utf-8').readlines()
    data = [json.loads(d) for d in f]

    inputs = []
    targets = []
    paths = []
    for index, d in enumerate(data):
        if pd.isnull(d['target']) or pd.isna(d['target']):
            continue
        inputs.append(d['input'])
        targets.append(d['target'])
        paths.append(d['path'])
    dict_ = {'input': inputs, 'output': targets, 'path': paths}
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
        self.examples = [[i, o, p] for i, o, p in zip(inputs, outputs, paths)]       

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

class Seq2SeqCollator(object):
    def __init__(self, args, tokenizer, mode="train"):
        self.tokenizer = tokenizer
        self.args = args    
        self.mode = mode
        self.feature = args.feature
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)                                               

    def __call__(self, batch):
        if self.mode == "dev":
            inputs = [d[0] for d in batch]     
            inputs = self.tokenizer(inputs, max_length=self.args.max_length, truncation=True, padding=True, return_tensors='pt')
        else:
            inputs = preprocess_data_batch(batch, self.tokenizer, self.args)
        
        if self.feature == 'text':
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


def preprocess_data_batch(data, tokenizer, args):
    
    inputs = [d[0] for d in data]
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
    lora_dim: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_module_name: str = 'q_proj,k_proj,v_proj,query_key_value'
    seed: int = 42
    offload_optimizer: bool = False
    deepspeed_config: str = None
    zero_shot: bool = False
    mode: str = "sft"
    gradient_checkpointing: bool = False

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