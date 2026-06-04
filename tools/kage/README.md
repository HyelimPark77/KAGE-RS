# KAGE-RS Descriptor Initialization

KAGE-RS uses an LLM-initialized descriptor memory, then updates descriptor
statistics during training. The seed descriptors should be generated through a
reproducible prompt/filter/compile workflow instead of being treated as a fixed
manual bank.

Generate one prompt per DIOR class:

```bash
python tools/kage/build_dior_descriptors.py prompts
```

Send the prompts in `tools/kage/dior_descriptor_prompts.jsonl` to an LLM and
save the responses as JSONL, for example:

```json
{"class_name": "airplane", "descriptors": ["elongated fuselage with broad wings", "..."]}
```

Compile the LLM responses into the descriptor memory consumed by KAGE-RS:

```bash
python tools/kage/build_dior_descriptors.py compile --input data/kage_descriptors/dior_llm_responses.jsonl
python tools/kage/build_dior_descriptors.py check
```

The compiled file is written to `data/kage_descriptors/dior_descriptors.json`.
The root `data/` directory is ignored by git, so force-add only the descriptor
artifact if you intentionally want to version it.

During training, KAGE-RS can export descriptor usage/confusion statistics to:

```bash
work_dirs/kage_descriptor_stats/dior_stats.json
```

Prepare a dynamic descriptor update round from those statistics:

```bash
python tools/kage/update_descriptors_from_stats.py \
  --descriptors tools/kage/dior_descriptors.json \
  --stats work_dirs/kage_descriptor_stats/dior_stats.json
```

This writes an updated descriptor memory and an LLM prompt file for generating
additional confusing-category-aware descriptors.

After sending `dior_update_prompts.jsonl` to an LLM, merge the response file:

```bash
python tools/kage/merge_descriptor_updates.py \
  --base work_dirs/kage_descriptor_stats/dior_descriptors_updated.json \
  --responses work_dirs/kage_descriptor_stats/dior_update_responses.jsonl
```

Use the merged memory in the next training round with:

```bash
--cfg-options model.kage_cfg.descriptor_path=../work_dirs/kage_descriptor_stats/dior_descriptors_merged.json
```
