# Handoff: RF-DETR batched inference benchmark (needs a GPU)

Paste the "Prompt" section into a new Claude Code session on the GPU machine.

## State
- `RFDETRModel.predict()` (src/models/rfdetr_model.py) currently calls `model.predict(image)` once per
  image. A list-input path exists behind `batched_inference = False` / `predict(..., batched=True)`:
  rfdetr stacks a list into ONE forward pass (checked in rfdetr/detr.py `predict`).
- `RFDETRModel.optimize_for_inference(dtype, compile, batch_size)` / `remove_optimization()` wrap
  rfdetr's `model.inference(...)`. rfdetr warns the unoptimized model is slow (~8x on a T4 with fp16).
  `compile=True` fixes the batch size: predict() raises on any other batch size (e.g. the last partial batch).
- `benchmark_rfdetr_batching.py` times loop vs batched (and fp32 vs fp16 with `--fp16`), reports peak GPU
  memory, and compares detections against the fp32 per-image loop.
- Already verified on CPU (Mac, pretrained COCO weights, 8 val images): batched == loop (0 count diffs,
  0 class mismatches, 0.00 px box diff). CPU shows no speedup from batching. GPU numbers are the open question.
- `infer.py --mode image --source <folder>` runs folder inference via `predictor.predict_batch` in chunks of
  `--batch-size` (default 8). It uses whatever `batched_inference` is set to.

## Setup on the GPU machine
    git pull   # needs the latest pushed commit with these files
    pip install -r requirements.txt   # pyarrow<22 is pinned on purpose (numpy<2)
    # fine-tuned weights: gcloud storage cp <.../checkpoint_best_total.pth> models/rfdetr_base_v7.pth

## Run
    python benchmark_rfdetr_batching.py --weights models/rfdetr_base_v7.pth --model-size base \
        --images datasets/rtd/images/val --batch-sizes 1 4 8 16 --fp16 --output bench.json

## What to decide from the results
1. Batched must match the loop (0 count diffs / class mismatches, box diff ~< 1 px). If not, keep it off.
2. If batched is clearly faster at batch 4-16 without running out of memory, set `batched_inference = True`.
3. If fp16 is much faster with negligible drift vs fp32, consider enabling it in infer.py (flag), keeping
   `compile=False` unless a fixed batch size (with remainder handling) is acceptable.

## Prompt
> Read HANDOFF_gpu_benchmark.md. Run benchmark_rfdetr_batching.py on this GPU with my weights, summarize the
> table, and apply the decisions in "What to decide from the results" (ask before flipping defaults).
