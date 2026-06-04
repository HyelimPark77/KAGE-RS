#!/usr/bin/env python3
"""Merge LLM-generated descriptor updates into KAGE-RS descriptor memory."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


CLASS_ALIASES = {
    'Expressway-Service-area': 'expressway service area',
    'Expressway-toll-station': 'expressway toll station',
}

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
    r'\bsignage\b',
    r'\bsigns?\b',
    r'\bbenches?\b',
    r'\binformation boards?\b',
    r'\bindoor\b',
    r'\bcafes?\b',
    r'\bshops?\b',
    r'\bwindows?\b',
    r'\bpedestrians?\b',
    r'\bcyclists?\b',
    r'\barchitectural\b',
    r'\bcommercial\b',
    r'\bbackground\b',
    r'\bvisible through\b',
]


def canonical_class_name(class_name: str) -> str:
    return CLASS_ALIASES.get(class_name.strip(), class_name.strip())


def normalize_descriptor(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    text = re.sub(r'\s+', ' ', text)
    return text.rstrip('.')


def is_rejected(text: str) -> bool:
    lower = text.lower()
    if len(lower.split()) < 3:
        return True
    return any(re.search(pattern, lower) for pattern in REJECT_PATTERNS)


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_constraints(path: Optional[Path]) -> Dict[str, dict]:
    if path is None or not path.exists():
        return {}
    return load_json(path)


def matches_constraints(text: str, constraints: Optional[dict]) -> bool:
    if not constraints:
        return True
    lower = text.lower()
    required = constraints.get('required_any', [])
    forbidden = constraints.get('reject_any', [])
    if required and not any(term in lower for term in required):
        return False
    if any(term in lower for term in forbidden):
        return False
    return True


def load_records(path: Path) -> List[dict]:
    raw = path.read_text(encoding='utf-8').strip()
    if not raw:
        return []
    if raw.startswith('['):
        return json.loads(raw)
    records = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSONL at line {line_no}: {exc}') from exc
    return records


def entry_text(entry) -> str:
    return entry if isinstance(entry, str) else entry.get('text', '')


def normalize_memory(memory: dict) -> dict:
    normalized = {}
    for class_name, entries in memory.items():
        class_name = canonical_class_name(class_name)
        normalized.setdefault(class_name, [])
        normalized[class_name].extend(entries)
    return normalized


def unique_entries(entries: Iterable[dict]) -> List[dict]:
    seen = set()
    unique = []
    for entry in entries:
        text = normalize_descriptor(entry_text(entry))
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        if isinstance(entry, dict):
            record = dict(entry)
        else:
            record = {'text': text}
        record['text'] = text
        record.setdefault('active', True)
        unique.append(record)
    return unique


def merge_updates(base_path: Path, response_path: Path, output_path: Path,
                  constraint_path: Optional[Path]) -> None:
    memory = normalize_memory(load_json(base_path))
    constraints = load_constraints(constraint_path)
    for class_name, entries in list(memory.items()):
        memory[class_name] = unique_entries(entries)

    for record in load_records(response_path):
        class_name = canonical_class_name(
            record.get('class_name') or record.get('class') or '')
        if class_name not in memory:
            raise ValueError(f'Unknown class in descriptor update: {class_name}')
        descriptors = record.get('descriptors')
        if not isinstance(descriptors, list):
            raise ValueError(f'Missing descriptors for class: {class_name}')
        additions = []
        for text in descriptors:
            if not isinstance(text, str):
                continue
            text = normalize_descriptor(text)
            class_constraints = constraints.get(class_name)
            if (text and not is_rejected(text)
                    and matches_constraints(text, class_constraints)):
                additions.append({
                    'text': text,
                    'active': True,
                    'source': 'llm_update',
                    'usage': 0,
                })
        memory[class_name] = unique_entries([*memory[class_name], *additions])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'Wrote merged descriptor memory to {output_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Merge LLM descriptor updates into KAGE-RS memory.')
    parser.add_argument(
        '--base',
        type=Path,
        default=Path('work_dirs/kage_descriptor_stats/dior_descriptors_updated.json'))
    parser.add_argument(
        '--responses',
        type=Path,
        required=True,
        help='LLM response JSON/JSONL generated from update prompts.')
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('work_dirs/kage_descriptor_stats/dior_descriptors_merged.json'))
    parser.add_argument(
        '--constraints',
        type=Path,
        default=Path('tools/kage/dior_descriptor_constraints.json'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_updates(args.base, args.responses, args.output, args.constraints)


if __name__ == '__main__':
    main()
