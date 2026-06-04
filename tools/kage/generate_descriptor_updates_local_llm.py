#!/usr/bin/env python3
"""Generate KAGE-RS descriptor update responses with a local HF LLM."""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional


SYSTEM_PROMPT = (
    'You generate descriptor-memory updates for remote sensing object '
    'detection. Return only the requested JSON object. Do not add markdown, '
    'comments, explanations, or extra keys.')


def load_prompt_records(path: Path) -> List[dict]:
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid prompt JSONL at line {line_no}: {exc}') from exc
            if 'class_name' not in record or 'prompt' not in record:
                raise ValueError(f'Missing class_name/prompt at line {line_no}')
            records.append(record)
    return records


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?', '', text).strip()
        text = re.sub(r'```$', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find('{')
    if start < 0:
        raise ValueError(f'No JSON object found in model output: {text[:200]}')
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return json.loads(text[start:idx + 1])
    raise ValueError(f'Unclosed JSON object in model output: {text[:200]}')


def extract_preserved_descriptors(prompt: str) -> set:
    preserved = set()
    in_section = False
    for line in prompt.splitlines():
        if line.startswith('Preserved high-usage descriptors:'):
            in_section = True
            continue
        if line.startswith('Frequently confused categories:'):
            break
        if in_section and line.startswith('- '):
            preserved.add(line[2:].strip().lower())
    return preserved


def validate_response(record: dict, expected_class: str,
                      preserved: Optional[set] = None) -> dict:
    class_name = record.get('class_name')
    descriptors = record.get('descriptors')
    if class_name != expected_class:
        raise ValueError(
            f'Expected class_name "{expected_class}", got "{class_name}"')
    if not isinstance(descriptors, list) or not descriptors:
        raise ValueError(f'Missing descriptor list for {expected_class}')
    descriptors = [d.strip() for d in descriptors if isinstance(d, str) and d.strip()]
    if preserved:
        copied = [d for d in descriptors if d.lower().rstrip('.') in preserved]
        if copied:
            raise ValueError(
                f'Model copied preserved descriptors for {expected_class}: {copied}')
    if not descriptors:
        raise ValueError(f'No valid string descriptors for {expected_class}')
    return {'class_name': expected_class, 'descriptors': descriptors}


def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    return f'{SYSTEM_PROMPT}\n\n{user_prompt}\n\nJSON:'


def generate_one(model, tokenizer, prompt: str, device: str, max_new_tokens: int,
                 temperature: float, top_p: float) -> str:
    import torch

    chat_prompt = build_chat_prompt(tokenizer, prompt)
    inputs = tokenizer(chat_prompt, return_tensors='pt').to(device)
    do_sample = temperature > 0
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id)
    generated = outputs[0, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run_generation(prompt_path: Path, output_path: Path, model_name_or_path: str,
                   device: str, dtype: str, limit: Optional[int],
                   max_new_tokens: int, temperature: float, top_p: float,
                   reject_existing: bool) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records = load_prompt_records(prompt_path)
    if limit is not None:
        records = records[:limit]
    torch_dtype = {
        'auto': 'auto',
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }[dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True)
    model.to(device)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for idx, record in enumerate(records, start=1):
            raw = generate_one(model, tokenizer, record['prompt'], device,
                               max_new_tokens, temperature, top_p)
            parsed = extract_json_object(raw)
            preserved = (extract_preserved_descriptors(record['prompt'])
                         if reject_existing else None)
            validated = validate_response(parsed, record['class_name'],
                                          preserved)
            f.write(json.dumps(validated, ensure_ascii=False) + '\n')
            f.flush()
            print(f'[{idx}/{len(records)}] {record["class_name"]}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate descriptor update JSONL with a local HF LLM.')
    parser.add_argument(
        '--prompts',
        type=Path,
        default=Path('work_dirs/kage_descriptor_stats/dior_update_prompts.jsonl'))
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('work_dirs/kage_descriptor_stats/dior_update_responses.jsonl'))
    parser.add_argument(
        '--model',
        required=True,
        help='Local model path or HuggingFace model id, e.g. Qwen/Qwen2.5-7B-Instruct.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--dtype',
        choices=['auto', 'float16', 'bfloat16', 'float32'],
        default='auto')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument(
        '--reject-existing',
        action='store_true',
        help='Fail if the model copies preserved high-usage descriptors.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_generation(args.prompts, args.output, args.model, args.device,
                   args.dtype, args.limit, args.max_new_tokens,
                   args.temperature, args.top_p, args.reject_existing)


if __name__ == '__main__':
    main()
