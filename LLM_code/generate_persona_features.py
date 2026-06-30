import argparse
import json
import os
import pickle

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def clean_text(text):
    text = text.replace('\u806e', "'")
    text = text.replace('\u806d', "")
    text = text.replace('\u806f', "")
    text = text.replace('\u8070', "")
    text = text.replace('\u8074', " ")
    text = text.replace('\u805f', " ")
    return text.strip()


def load_conversation_speakers(project_root, dataset):
    data_path = os.path.join(project_root, 'original_data', dataset, f'{dataset}.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    if dataset == 'iemocap':
        all_conv_ids = data[3] + data[4] + data[5]
        sentence_dict = data[2]
        speaker_source = data[0]

        def speaker_id(raw_speaker):
            return 0 if raw_speaker == 'M' else 1

    elif dataset == 'meld':
        all_conv_ids = data[4] + data[5] + data[6]
        sentence_dict = data[3]
        speaker_source = data[0]

        def speaker_id(raw_speaker):
            return raw_speaker.index(1)

    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    conversations = {}
    for conv_id in all_conv_ids:
        speakers = {}
        for raw_speaker, sentence in zip(speaker_source[conv_id], sentence_dict[conv_id]):
            sid = speaker_id(raw_speaker)
            speakers.setdefault(f'Speaker_{sid}', []).append(clean_text(sentence))
        conversations[str(conv_id)] = speakers

    return conversations


def build_prompt(dataset, conv_id, speaker_name, utterances):
    utterance_text = '\n'.join(f'- {utterance}' for utterance in utterances if utterance)
    return (
        'You are creating speaker persona features for emotion recognition in conversation.\n'
        f'Dataset: {dataset}\n'
        f'Conversation id: {conv_id}\n'
        f'Speaker: {speaker_name}\n\n'
        'All utterances from this speaker in this conversation:\n'
        f'{utterance_text}\n\n'
        'Infer the speaker persona using only the utterances above. Focus on stable traits that may help '
        'emotion recognition: role in the conversation, communication style, attitude, relationship cues, '
        'and recurring behavior. Avoid copying long phrases from the dialogue. Do not predict an emotion label. '
        'Write 1-2 concise English sentences.'
    )


def encode_prompt(tokenizer, prompt, device, max_input_tokens):
    messages = [{'role': 'user', 'content': prompt}]
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors='pt',
            truncation=True,
            max_length=max_input_tokens,
        )
    else:
        llama2_prompt = f'<s>[INST] {prompt} [/INST]'
        input_ids = tokenizer(
            llama2_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=max_input_tokens,
        )['input_ids']
    return input_ids.to(device)


def generate_profile(model, tokenizer, prompt, max_input_tokens, max_new_tokens):
    input_ids = encode_prompt(tokenizer, prompt, model.device, max_input_tokens)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    profile = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return ' '.join(profile.split())


def load_existing_output(output_path, dataset, model_name_or_path):
    if not os.path.exists(output_path):
        return {
            'dataset': dataset,
            'model_name_or_path': model_name_or_path,
            'profiles': {},
        }
    with open(output_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_output(output_path, payload):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Generate offline speaker persona features with a local LLaMA2 model.')
    parser.add_argument('--project_root', type=str, default='..', help='Project root path, e.g. /home/pc/jcy/SpeechCueLLM')
    parser.add_argument('--dataset', type=str, required=True, choices=['iemocap', 'meld'])
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Local LLaMA2/LLaMA2-chat HuggingFace path')
    parser.add_argument('--output_path', type=str, required=True, help='Where to save persona json')
    parser.add_argument('--max_input_tokens', type=int, default=3500)
    parser.add_argument('--max_new_tokens', type=int, default=96)
    parser.add_argument('--device_map', type=str, default='cuda')
    parser.add_argument('--torch_dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }

    conversations = load_conversation_speakers(args.project_root, args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype_map[args.torch_dtype]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    payload = load_existing_output(args.output_path, args.dataset, args.model_name_or_path)
    profiles = payload.setdefault('profiles', {})

    completed_conversations = 0
    for conv_id, speakers in tqdm(conversations.items(), desc='Generating persona features'):
        profiles.setdefault(conv_id, {})
        for speaker_name, utterances in speakers.items():
            if not args.overwrite and profiles[conv_id].get(speaker_name):
                continue
            prompt = build_prompt(args.dataset, conv_id, speaker_name, utterances)
            profiles[conv_id][speaker_name] = generate_profile(
                model,
                tokenizer,
                prompt,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
            )

        completed_conversations += 1
        if completed_conversations % args.save_every == 0:
            save_output(args.output_path, payload)

    save_output(args.output_path, payload)
    print(args.output_path)


if __name__ == '__main__':
    main()
