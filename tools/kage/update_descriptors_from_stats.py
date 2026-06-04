#!/usr/bin/env python3
"""Prepare dynamic descriptor updates from KAGE-RS usage/confusion stats."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


CLASS_ALIASES = {
    'Expressway-Service-area': 'expressway service area',
    'Expressway-toll-station': 'expressway toll station',
}


PROMPT_TEMPLATE = """You are updating descriptor memory for KAGE-RS, a remote sensing open-vocabulary object detector.

Target class: {class_name}
Preserved high-usage descriptors:
{high_usage}

Frequently confused categories:
{confusions}

Target-specific required cues:
{required_cues}

Target-specific forbidden cues:
{forbidden_cues}

Generate 8 to 10 new short English visual descriptors for the target class.

Hard constraints:
- Describe only cues observable from overhead remote sensing imagery.
- Prefer cues that distinguish the target class from the confusing categories.
- Do not use ground-view object parts, function-only knowledge, hidden parts, intent, or unverifiable facts.
- Do not mention signs, signage, benches, indoor facilities, cafes, shops, windows, pedestrians, cyclists, architectural style, or background buildings.
- Do not copy any preserved descriptor.
- Do not lightly rephrase any preserved descriptor.
- Every descriptor must add a new visual cue or a more discriminative cue.
- Each descriptor should help distinguish the target class from at least one confusing category.
- Each descriptor must include at least one target-specific required cue if provided.
- Do not use any target-specific forbidden cue.

Good descriptor style:
- "paved roadside compound connected by curved access roads"
- "large apron fields containing aircraft-sized objects"
- "fan-shaped sports field with a diamond infield core"

Bad descriptor style:
- "benches and information boards"
- "indoor cafes visible through windows"
- "distinctive architectural design elements"

Return exactly one JSON object on one line:
{{"class_name": "{class_name}", "descriptors": ["...", "..."]}}
"""


def descriptor_text(entry) -> str:
    return entry if isinstance(entry, str) else entry.get('text', '')


def canonical_class_name(class_name: str) -> str:
    return CLASS_ALIASES.get(class_name, class_name)


def descriptor_record(text: str, entry, active: bool, source: str) -> dict:
    if isinstance(entry, dict):
        record = dict(entry)
    else:
        record = {'text': text}
    record['text'] = text
    record['active'] = active
    record.setdefault('source', source)
    return record


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_constraints(path: Optional[Path]) -> Dict[str, dict]:
    if path is None or not path.exists():
        return {}
    return load_json(path)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n')


def format_bullets(items: List[str]) -> str:
    if not items:
        return '- none'
    return '\n'.join(f'- {item}' for item in items)


def update_descriptors(descriptor_path: Path, stats_path: Path,
                       output_path: Path, prompt_path: Path,
                       min_usage: int, max_confusions: int,
                       constraint_path: Optional[Path]) -> None:
    descriptors = load_json(descriptor_path)
    stats = load_json(stats_path)
    constraints = load_constraints(constraint_path)
    usage: Dict[str, Dict[str, int]] = stats.get('usage', {})
    confusion: Dict[str, Dict[str, int]] = stats.get('confusion', {})

    updated = {}
    prompt_records = []
    for raw_class_name, entries in descriptors.items():
        class_name = canonical_class_name(raw_class_name)
        class_usage = usage.get(class_name, {})
        high_usage = []
        new_entries = []
        for entry in entries:
            text = descriptor_text(entry)
            if not text:
                continue
            count = class_usage.get(text, 0)
            active = count >= min_usage
            new_entry = descriptor_record(text, entry, active, 'llm_seed')
            new_entry['usage'] = count
            new_entries.append(new_entry)
            if active:
                high_usage.append(text)
        updated[class_name] = new_entries

        sorted_confusions = sorted(
            confusion.get(class_name, {}).items(),
            key=lambda item: item[1],
            reverse=True)[:max_confusions]
        confusing_names = [
            f'{name} ({count})' for name, count in sorted_confusions
        ]
        if high_usage or confusing_names:
            class_constraints = constraints.get(class_name, {})
            prompt_records.append({
                'class_name': class_name,
                'prompt': PROMPT_TEMPLATE.format(
                    class_name=class_name,
                    high_usage=format_bullets(high_usage),
                    confusions=format_bullets(confusing_names),
                    required_cues=format_bullets(
                        class_constraints.get('required_any', [])),
                    forbidden_cues=format_bullets(
                        class_constraints.get('reject_any', []))),
            })

    write_json(output_path, updated)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    with prompt_path.open('w', encoding='utf-8') as f:
        for record in prompt_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f'Wrote updated descriptor memory to {output_path}')
    print(f'Wrote update prompts to {prompt_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare KAGE-RS dynamic descriptor update artifacts.')
    parser.add_argument(
        '--descriptors',
        type=Path,
        default=Path('tools/kage/dior_descriptors.json'))
    parser.add_argument(
        '--stats',
        type=Path,
        default=Path('work_dirs/kage_descriptor_stats/dior_stats.json'))
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('work_dirs/kage_descriptor_stats/dior_descriptors_updated.json'))
    parser.add_argument(
        '--prompts',
        type=Path,
        default=Path('work_dirs/kage_descriptor_stats/dior_update_prompts.jsonl'))
    parser.add_argument('--min-usage', type=int, default=1)
    parser.add_argument('--max-confusions', type=int, default=3)
    parser.add_argument(
        '--constraints',
        type=Path,
        default=Path('tools/kage/dior_descriptor_constraints.json'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_descriptors(args.descriptors, args.stats, args.output, args.prompts,
                       args.min_usage, args.max_confusions, args.constraints)


if __name__ == '__main__':
    main()
