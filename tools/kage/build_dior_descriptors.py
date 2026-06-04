#!/usr/bin/env python3
"""Build DIOR descriptor seeds for KAGE-RS.

This script keeps descriptor initialization reproducible without baking a
hand-written descriptor bank into the method. It writes one LLM prompt per
DIOR class, then compiles LLM JSON/JSONL responses into the descriptor memory
format consumed by KAGEBranch.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


DIOR_CLASSES = [
    'Expressway-Service-area',
    'Expressway-toll-station',
    'airplane',
    'airport',
    'baseballfield',
    'basketballcourt',
    'bridge',
    'chimney',
    'dam',
    'golffield',
    'groundtrackfield',
    'harbor',
    'overpass',
    'ship',
    'stadium',
    'storagetank',
    'tenniscourt',
    'trainstation',
    'vehicle',
    'windmill',
]

DEFAULT_PROMPT_PATH = Path('tools/kage/dior_descriptor_prompts.jsonl')
DEFAULT_DESCRIPTOR_PATH = Path('data/kage_descriptors/dior_descriptors.json')

REJECT_PATTERNS = [
    r'\bused for\b',
    r'\bdesigned to\b',
    r'\bcapable of\b',
    r'\btypically carries\b',
    r'\bpassengers?\b',
    r'\bdrivers?\b',
    r'\bpeople\b',
    r'\bwheel(s)?\b',
    r'\bheadlight(s)?\b',
    r'\bwindshield(s)?\b',
    r'\bdoor(s)?\b',
    r'\bhandlebar(s)?\b',
    r'\bpedal(s)?\b',
    r'\bengine(s)?\b',
    r'\binterior\b',
]

PROMPT_TEMPLATE = """You are helping initialize descriptor memory for KAGE-RS, a remote sensing open-vocabulary object detector.

Target class: {class_name}

Generate 12 to 20 short English visual descriptors for this class.

Hard constraints:
- Describe only cues that can be observed from overhead remote sensing imagery.
- Prefer footprint shape, relative scale, layout, surrounding context, texture, and cues that distinguish confusing DIOR categories.
- Do not use ground-view object parts unless they are reliably visible from overhead imagery.
- Do not use function-only knowledge, hidden parts, intent, or unverifiable facts.
- Do not mention dataset names, detector names, or the word "descriptor".
- Each descriptor must be a concise noun phrase or visual phrase.
- Avoid duplicates and near-duplicates.

Return exactly one JSON object on one line:
{{"class_name": "{class_name}", "descriptors": ["...", "..."]}}
"""


def normalize_class_name(name: str) -> str:
    return name.strip()


def normalize_descriptor(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    text = re.sub(r'\s+', ' ', text)
    text = text.rstrip('.')
    return text


def is_rejected(text: str) -> bool:
    lower = text.lower()
    if len(lower.split()) < 3:
        return True
    return any(re.search(pattern, lower) for pattern in REJECT_PATTERNS)


def unique_preserve_order(texts: Iterable[str]) -> List[str]:
    seen = set()
    unique = []
    for text in texts:
        key = text.lower()
        if key not in seen:
            seen.add(key)
            unique.append(text)
    return unique


def write_prompts(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for class_name in DIOR_CLASSES:
            record = {
                'class_name': class_name,
                'prompt': PROMPT_TEMPLATE.format(class_name=class_name),
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_llm_records(input_path: Path) -> List[dict]:
    raw = input_path.read_text(encoding='utf-8').strip()
    if not raw:
        return []
    if raw.startswith('['):
        records = json.loads(raw)
        if not isinstance(records, list):
            raise ValueError('JSON input must be a list of objects.')
        return records

    records = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSONL at line {line_no}: {exc}') from exc
    return records


def compile_descriptors(input_path: Path, output_path: Path,
                        min_descriptors: int) -> None:
    class_set = set(DIOR_CLASSES)
    compiled: Dict[str, List[dict]] = {class_name: [] for class_name in DIOR_CLASSES}

    for record in load_llm_records(input_path):
        class_name = normalize_class_name(
            record.get('class_name') or record.get('class') or '')
        if class_name not in class_set:
            raise ValueError(f'Unknown DIOR class in LLM output: {class_name}')
        descriptors = record.get('descriptors')
        if not isinstance(descriptors, list):
            raise ValueError(f'Missing descriptor list for class: {class_name}')
        cleaned = [
            normalize_descriptor(text) for text in descriptors
            if isinstance(text, str)
        ]
        cleaned = [text for text in cleaned if text and not is_rejected(text)]
        cleaned = unique_preserve_order(cleaned)
        compiled[class_name].extend({
            'text': text,
            'active': True,
            'source': 'llm_seed',
            'usage': 0,
        } for text in cleaned)

    for class_name, descriptors in compiled.items():
        if len(descriptors) < min_descriptors:
            raise ValueError(
                f'{class_name} has {len(descriptors)} descriptors after '
                f'filtering; expected at least {min_descriptors}.')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(compiled, f, indent=2, ensure_ascii=False)
        f.write('\n')


def check_descriptors(input_path: Path, min_descriptors: int) -> None:
    records = json.loads(input_path.read_text(encoding='utf-8'))
    missing = [class_name for class_name in DIOR_CLASSES if class_name not in records]
    if missing:
        raise ValueError(f'Missing DIOR classes: {missing}')
    for class_name in DIOR_CLASSES:
        entries = records[class_name]
        if len(entries) < min_descriptors:
            raise ValueError(
                f'{class_name} has {len(entries)} descriptors; expected at '
                f'least {min_descriptors}.')
        texts = [
            entry if isinstance(entry, str) else entry.get('text', '')
            for entry in entries
        ]
        rejected = [text for text in texts if is_rejected(normalize_descriptor(text))]
        if rejected:
            raise ValueError(f'{class_name} has rejected descriptors: {rejected}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create and validate DIOR descriptor seeds for KAGE-RS.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    prompt_parser = subparsers.add_parser(
        'prompts', help='Write one LLM prompt per DIOR class.')
    prompt_parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help='Path to write JSONL prompts.')

    compile_parser = subparsers.add_parser(
        'compile', help='Compile LLM JSON/JSONL responses into descriptor JSON.')
    compile_parser.add_argument(
        '--input', type=Path, required=True, help='LLM response JSON/JSONL path.')
    compile_parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_DESCRIPTOR_PATH,
        help='Path to write KAGE descriptor memory JSON.')
    compile_parser.add_argument(
        '--min-descriptors',
        type=int,
        default=8,
        help='Minimum accepted descriptors per class after filtering.')

    check_parser = subparsers.add_parser(
        'check', help='Validate a compiled descriptor JSON file.')
    check_parser.add_argument(
        '--input',
        type=Path,
        default=DEFAULT_DESCRIPTOR_PATH,
        help='Compiled descriptor JSON path.')
    check_parser.add_argument(
        '--min-descriptors',
        type=int,
        default=8,
        help='Minimum accepted descriptors per class.')

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == 'prompts':
        write_prompts(args.output)
        print(f'Wrote DIOR descriptor prompts to {args.output}')
    elif args.command == 'compile':
        compile_descriptors(args.input, args.output, args.min_descriptors)
        print(f'Wrote KAGE descriptor memory to {args.output}')
    elif args.command == 'check':
        check_descriptors(args.input, args.min_descriptors)
        print(f'Validated descriptor memory at {args.input}')


if __name__ == '__main__':
    main()
