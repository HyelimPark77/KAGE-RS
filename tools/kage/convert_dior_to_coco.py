#!/usr/bin/env python3
"""Convert raw DIOR horizontal XML annotations to LAE-DINO COCO JSON files."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional


LAE_DIOR_CLASSES = [
    'airplane',
    'airport',
    'groundtrackfield',
    'harbor',
    'baseballfield',
    'overpass',
    'basketballcourt',
    'ship',
    'bridge',
    'stadium',
    'storagetank',
    'tenniscourt',
    'expressway service area',
    'trainstation',
    'expressway toll station',
    'vehicle',
    'golffield',
    'windmill',
    'chimney',
    'dam',
]

CLASS_ALIASES = {
    'expressway-service-area': 'expressway service area',
    'expressway service area': 'expressway service area',
    'expressway-toll-station': 'expressway toll station',
    'expressway toll station': 'expressway toll station',
    'ground track field': 'groundtrackfield',
    'groundtrackfield': 'groundtrackfield',
    'golf field': 'golffield',
    'golffield': 'golffield',
    'storage tank': 'storagetank',
    'storagetank': 'storagetank',
    'baseball field': 'baseballfield',
    'baseballfield': 'baseballfield',
    'basketball court': 'basketballcourt',
    'basketballcourt': 'basketballcourt',
    'tennis court': 'tenniscourt',
    'tenniscourt': 'tenniscourt',
    'train station': 'trainstation',
    'trainstation': 'trainstation',
}


def normalize_class_name(name: str) -> str:
    key = name.strip().lower().replace('_', ' ')
    return CLASS_ALIASES.get(key, key)


def read_split_file(path: Path) -> List[str]:
    image_ids = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            image_id = line.strip().split()[0] if line.strip() else ''
            if image_id:
                image_ids.append(Path(image_id).stem)
    return image_ids


def find_split(root: Path, requested: Optional[str], candidates: Iterable[str]) -> Path:
    if requested:
        path = Path(requested)
        return path if path.is_absolute() else root / path
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f'Could not find any split file from: {candidates}')


def parse_size(root: ET.Element, image_path: Path) -> tuple:
    size = root.find('size')
    if size is not None:
        width = int(float(size.findtext('width', default='0')))
        height = int(float(size.findtext('height', default='0')))
        if width > 0 and height > 0:
            return width, height
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            'XML size is missing; install pillow to infer image dimensions.'
        ) from exc
    with Image.open(image_path) as img:
        return img.size


def convert_split(raw_root: Path, image_ids: List[str],
                  class_to_id: Dict[str, int]) -> dict:
    ann_dir = raw_root / 'Annotations' / 'Horizontal Bounding Boxes'
    image_dir = raw_root / 'JPEGImages-trainval'
    images = []
    annotations = []
    ann_id = 1

    for image_idx, image_id in enumerate(image_ids, start=1):
        xml_path = ann_dir / f'{image_id}.xml'
        if not xml_path.exists():
            raise FileNotFoundError(f'Missing annotation XML: {xml_path}')

        tree = ET.parse(xml_path)
        xml_root = tree.getroot()
        filename = xml_root.findtext('filename') or f'{image_id}.jpg'
        image_path = image_dir / filename
        if not image_path.exists():
            for suffix in ('.jpg', '.png', '.tif', '.tiff'):
                candidate = image_dir / f'{image_id}{suffix}'
                if candidate.exists():
                    image_path = candidate
                    filename = candidate.name
                    break
        width, height = parse_size(xml_root, image_path)
        images.append(
            dict(id=image_idx, file_name=filename, width=width, height=height))

        for obj in xml_root.findall('object'):
            name = normalize_class_name(obj.findtext('name', default=''))
            if name not in class_to_id:
                raise ValueError(f'Unknown DIOR class "{name}" in {xml_path}')
            box = obj.find('bndbox')
            if box is None:
                continue
            xmin = float(box.findtext('xmin'))
            ymin = float(box.findtext('ymin'))
            xmax = float(box.findtext('xmax'))
            ymax = float(box.findtext('ymax'))
            xmin = max(0.0, min(xmin, width - 1.0))
            ymin = max(0.0, min(ymin, height - 1.0))
            xmax = max(0.0, min(xmax, width * 1.0))
            ymax = max(0.0, min(ymax, height * 1.0))
            box_w = max(0.0, xmax - xmin)
            box_h = max(0.0, ymax - ymin)
            if box_w <= 0 or box_h <= 0:
                continue
            annotations.append(
                dict(
                    id=ann_id,
                    image_id=image_idx,
                    category_id=class_to_id[name],
                    bbox=[xmin, ymin, box_w, box_h],
                    area=box_w * box_h,
                    iscrowd=0))
            ann_id += 1

    categories = [
        dict(id=i + 1, name=name, supercategory='object')
        for i, name in enumerate(LAE_DIOR_CLASSES)
    ]
    return dict(images=images, annotations=annotations, categories=categories)


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f)
        f.write('\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert raw DIOR XML annotations to LAE-DINO COCO JSON.')
    parser.add_argument(
        '--raw-root',
        type=Path,
        default=Path('data/OpenDataLab___DIOR/raw/DIOR'),
        help='Raw DIOR root containing JPEGImages-trainval, Annotations, ImageSets.')
    parser.add_argument(
        '--out-root',
        type=Path,
        default=Path('data/LAE-FOD/DIOR'),
        help='Output root expected by mmdetection_lae configs.')
    parser.add_argument('--train-split', default=None)
    parser.add_argument('--val-split', default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_to_id = {name: i + 1 for i, name in enumerate(LAE_DIOR_CLASSES)}
    train_split = find_split(
        args.raw_root, args.train_split,
        ['ImageSets/Main/train.txt', 'ImageSets/Main/trainval.txt'])
    val_split = find_split(
        args.raw_root, args.val_split,
        ['ImageSets/Main/val.txt', 'ImageSets/Main/test.txt'])

    train_ids = read_split_file(train_split)
    val_ids = read_split_file(val_split)
    train_json = convert_split(args.raw_root, train_ids, class_to_id)
    val_json = convert_split(args.raw_root, val_ids, class_to_id)

    write_json(train_json, args.out_root / 'processed_DIOR_train.json')
    write_json(val_json, args.out_root / 'DIOR_val.json')

    image_src = args.raw_root / 'JPEGImages-trainval'
    image_dst = args.out_root / 'JPEGImages-trainval'
    print(f'Wrote {len(train_json["images"])} train images, '
          f'{len(train_json["annotations"])} train boxes')
    print(f'Wrote {len(val_json["images"])} val images, '
          f'{len(val_json["annotations"])} val boxes')
    print(f'Expected image directory for config: {image_dst}')
    print(f'If it does not exist, create a symlink to: {image_src}')


if __name__ == '__main__':
    main()
