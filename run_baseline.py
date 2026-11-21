"""Zero-shot description generation for GastroVL with Qwen2.5-VL.

Runs on the cloud instance. Reads gpu_bundle/test.csv, generates one
description per image, and writes results to CSV for evaluation.

Typical use on AutoDL:
    python run_baseline.py --limit 10          # smoke test first
    python run_baseline.py                     # full test split

Output columns: image_id, patient_id, site, class, generated, reference
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

DEFAULT_MODEL = os.environ.get("GASTROVL_MODEL",
                                "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct")
DEFAULT_BUNDLE = Path(os.environ.get("GASTROVL_BUNDLE", "gpu_bundle"))

PROMPT = (
    "You are an experienced endoscopist reviewing a still image from "
    "esophagogastroduodenoscopy. The image was captured at the {site}. "
    "Describe only what is visible in this image, as one line of "
    "comma-separated phrases in the telegraphic style of a clinical endoscopy "
    "report. Begin the line with the anatomical site. "
    "If an abnormality is visible, name its morphology first, then its size in "
    "cm, its surface and its colour. If the mucosa is unremarkable, say so "
    "briefly. Do not write full sentences. Do not give a diagnosis."
)


def clean(text: str) -> str:
    """Strip common preambles the model adds."""
    t = text.strip()
    for prefix in ("Description:", "Findings:", "Answer:"):
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix):].strip()
    return " ".join(t.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--out", default="results_zeroshot.csv")
    ap.add_argument("--limit", type=int, default=0, help="only first N rows (smoke test)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--adapter", default="", help="path to a trained LoRA adapter")
    ap.add_argument("--only-class", default="", choices=["", "normal", "abnormal"],
                    help="restrict to one class (debugging)")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    meta = pd.read_csv(bundle / "test.csv", dtype=str)
    if args.only_class:
        meta = meta[meta["class"].str.lower() == args.only_class]
    rows_in = meta.to_dict("records")
    if args.limit:
        rows_in = rows_in[: args.limit]
    print(f"rows to process: {len(rows_in)}")

    print("loading model ...", flush=True)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    if args.adapter:
        from peft import PeftModel
        print(f"loading adapter: {args.adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    print(f"model ready in {time.time() - t0:.0f}s | "
          f"gpu mem {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    out_rows = []
    t0 = time.time()
    for i, r in enumerate(rows_in, 1):
        image_id = r["image_id"]
        img_path = bundle / "images" / f"{image_id}.png"
        site = r["anatomical_site"].split(" ", 1)[-1]
        prompt = PROMPT.format(site=site)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            size={"longest_edge": 512 * 28, "shortest_edge": 256 * 28},
        ).to("cuda")

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,          # greedy: reproducible
            )
        out = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)[0]

        ref_path = bundle / "texts" / f"{image_id}.txt"
        reference = ""
        if ref_path.exists():
            for line in ref_path.read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("description:"):
                    reference = line.split(":", 1)[1].strip()

        out_rows.append({
            "image_id": image_id,
            "patient_id": r["patient_id"],
            "site": r["anatomical_site"],
            "class": r["class"],
            "generated": clean(out),
            "reference": reference,
        })

        if i % 25 == 0 or i == len(rows_in):
            el = time.time() - t0
            print(f"  {i}/{len(rows_in)}  {el/i:.2f}s/img  "
                  f"eta {(len(rows_in)-i)*el/i/60:.0f} min", flush=True)

    df = pd.DataFrame(out_rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nwritten: {args.out}  ({len(df)} rows)")
    print("\n--- sample ---")
    for r in df.head(3).to_dict("records"):
        print(f"[{r['image_id']}] {r['generated']}")


if __name__ == "__main__":
    main()
