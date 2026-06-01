# Usage

## Install

```bash
pip install -r requirements.txt
```

## Environment

Prompt/profile generation:

```bash
export LLM_API_KEY="your_llm_api_key"
export LLM_API_BASE="https://api.deepseek.com"
export LLM_MODEL="deepseek-v4-pro"
```

Image generation:

```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/api/v1"
export DASHSCOPE_IMAGE_MODEL="qwen-image-2.0-pro"
```

For Singapore region:

```bash
export DASHSCOPE_API_BASE="https://dashscope-intl.aliyuncs.com/api/v1"
```

## Generate Prompts

Generate a prompt set with API:

```bash
python scripts/generate_prompt_set.py --occupation nurse --use-api
```

Use an existing profile:

```bash
python scripts/generate_prompt_set.py --occupation nurse --profile configs/nurse_profile.json
```

Constrain C/E pair occupations with a candidate pool:

```bash
python scripts/generate_prompt_set.py \
  --occupation CEO \
  --use-api \
  --pair-pool configs/occupation_pair_pool.json \
  --pair-pool-sample-size 12 \
  --seed 42
```

Outputs are written to:

```text
data/generated/{occupation}_profile.json
data/generated/{occupation}_prompt_set.jsonl
data/generated/{occupation}_prompt_set_table.md
```

## Validate Prompts

```bash
python scripts/validate_prompt_set.py data/generated/nurse_prompt_set.jsonl
```

## Build Eval Plans

Attach prompt-aware evaluation plans:

```bash
python eval_plan_builder.py \
  --input data/nurse_prompt_set.jsonl \
  --output data/nurse_prompt_set_with_eval_plan.jsonl
```

Validate without writing:

```bash
python eval_plan_builder.py --input data/nurse_prompt_set.jsonl --validate-only
```

Show example plans:

```bash
python eval_plan_builder.py \
  --input data/nurse_prompt_set.jsonl \
  --show-examples nurse_A1 nurse_B1 nurse_C2 nurse_D1 nurse_E1 nurse_E3 nurse_F1
```

## Generate Images

Preview selected prompts without calling the API:

```bash
python scripts/generate_images_from_prompt_set.py --occupation nurse --module A --dry-run
```

Generate images:

```bash
python scripts/generate_images_from_prompt_set.py --occupation nurse --limit 3
```

Generate a specific prompt:

```bash
python scripts/generate_images_from_prompt_set.py --occupation nurse --prompt-id nurse_A1
```

Use seed for reproducibility:

```bash
python scripts/generate_images_from_prompt_set.py \
  --occupation nurse \
  --prompt-id nurse_A1 \
  --seed 42
```

Recommended benchmark setting:

```bash
python scripts/generate_images_from_prompt_set.py \
  --occupation nurse \
  --prompt-id nurse_A1 \
  --seed 42 \
  --no-prompt-extend
```

Generated images are written to:

```text
outputs/images/{model_name}/{occupation}_seed{seed}/
outputs/images/{model_name}/{occupation}_seed{seed}/manifest.jsonl
```

Use `--skip-existing` to resume an interrupted run. The script checks each expected sample image, skips existing files, and continues with the corresponding next seed.

Batch-generate all selected prompts for one occupation/model from a config:

```bash
python scripts/batch_generate_images.py \
  --config configs/image_generation_batch.example.json \
  --dry-run

python scripts/batch_generate_images.py \
  --config configs/image_generation_batch.example.json
```

The batch config can set global defaults such as `occupation`, `model`, `size`, `n`, `seed`, `prompt_extend`, and `skip_existing`. It also supports `module_overrides`, `slice_overrides`, and `prompt_overrides` for per-set or per-prompt generation settings.

## Run Benchmark

Run the full benchmark pipeline with the mock evaluator:

```bash
python benchmark/benchmark_runner.py \
  --prompt-set data/nurse_prompt_set.jsonl \
  --manifest outputs/images/qwen_image_2_0/nurse_seed42/manifest.jsonl \
  --output-dir outputs/benchmark/nurse \
  --evaluator mock
```

The runner accepts prompt sets with or without `eval_plan`. If missing, eval plans are built automatically.

Run VLM semantic annotation with DashScope OpenAI-compatible API:

```bash
export DASHSCOPE_API_KEY="sk-..."
export DASHSCOPE_VLM_MODEL="qwen-vl-plus"

python benchmark/benchmark_runner.py \
  --prompt-set data/nurse_prompt_set.jsonl \
  --manifest outputs/images/qwen_image_2_0/nurse_seed42/manifest.jsonl \
  --output-dir outputs/benchmark/nurse_qwen_vl \
  --evaluator qwen_vl \
  --face-results outputs/images/qwen_image_2_0/nurse_seed42/scrfd_padding_fallback/all_face_results.json \
  --face-attribute-results outputs/images/qwen_image_2_0/nurse_seed42/fairface_padding_fallback2/all_face_attribute_results.json \
  --vlm-image-mode auto
```

`qwen_vl` fills A/B/E1/E2/F from SCRFD/FairFace without calling VLM. For A/E1/E2 single-person prompts, background faces are ignored and only the largest SCRFD face is used as the primary face. `--vlm-image-mode auto` sends SCRFD-annotated images to VLM for C/D/E role or contextual tasks when available. FairFace attributes are sent as auxiliary context and fused afterward through `face_id`. C/E pair-role annotations include `quality_impact` for cross-role visual contamination. D contextual prompts keep all detected faces in `annotations.persons[]` so extra patients/assistants can be counted and marked as prompt-alignment issues.

For smoke tests, filter to a small prompt subset before calling the VLM:

```bash
python benchmark/benchmark_runner.py \
  --prompt-set data/nurse_prompt_set.jsonl \
  --manifest outputs/images/qwen_image_2_0/nurse_seed42/manifest.jsonl \
  --output-dir outputs/benchmark/nurse_c2_smoke \
  --evaluator qwen_vl \
  --prompt-id nurse_C2 \
  --sample-index 1 \
  --face-results outputs/images/qwen_image_2_0/nurse_seed42/scrfd/all_face_results.json \
  --face-attribute-results outputs/images/qwen_image_2_0/nurse_seed42/fairface/all_face_attribute_results.json \
  --vlm-image-mode auto
```

Outputs:

```text
outputs/benchmark/{occupation}/annotations.jsonl
outputs/benchmark/{occupation}/per_image_metrics.jsonl
outputs/benchmark/{occupation}/aggregate_metrics.json
outputs/benchmark/{occupation}/benchmark_report.md
```

Metric outputs are intentionally split into tiers:

- `core`: default report metrics, used for the main benchmark reading.
- `diagnostic`: failure-mode metrics such as swap, confusion, leakage, or hidden-bias counts.
- `auxiliary`: debug/proxy fields. For example, local-only image quality currently uses a placeholder score when no quality model is attached.
- `planned`: schema-level metrics whose final aggregation formula is not implemented yet.

The Markdown report shows `core` and `diagnostic` only. Full metric values remain in `per_image_metrics.jsonl` and `aggregate_metrics.json`.

## Face Attribute Detection

Detect faces with SCRFD:

```bash
python check/detect_faces_scrfd.py \
  --input outputs/images/qwen_image_2_0/nurse_seed42 \
  --output-dir outputs/images/qwen_image_2_0/nurse_seed42/scrfd \
  --score-threshold 0.5
```

Classify SCRFD face crops with FairFace:

```bash
python check/detect_face_attributes_fairface.py \
  --scrfd-results outputs/images/qwen_image_2_0/nurse_seed42/scrfd/all_face_results.json \
  --weights /path/to/fairface_alldata_4race_20191111.pt \
  --race-mode auto \
  --output-dir outputs/images/qwen_image_2_0/nurse_seed42/fairface \
  --save-crops
```
