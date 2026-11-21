"""LoRA fine-tuning of Qwen2.5-VL on the GastroVL training split.

Trains a small adapter on image -> description pairs, then saves it so that
run_baseline.py can load it with --adapter for evaluation.

Usage on the cloud instance:
    python train_lora.py --epochs 3 --out lora_gastrovl
Then evaluate:
    python run_baseline.py --adapter lora_gastrovl --out results_lora.csv
"""
from __future__ import annotations

import argparse
import os
import json
import time
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

DEFAULT_MODEL = os.environ.get("GASTROVL_MODEL",
                                "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct")
DEFAULT_BUNDLE = Path(os.environ.get("GASTROVL_BUNDLE", "train_bundle"))

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

# must match run_baseline.py so that training and inference see the same input
SIZE = {"longest_edge": 512 * 28, "shortest_edge": 256 * 28}


class PairDataset(Dataset):
    def __init__(self, bundle: Path, meta: pd.DataFrame):
        self.bundle = bundle
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        r = self.meta.iloc[idx]
        image_id = r["image_id"]
        site = r["anatomical_site"].split(" ", 1)[-1]
        prompt = PROMPT.format(site=site)

        ref_path = self.bundle / "texts" / f"{image_id}.txt"
        answer = ""
        for line in ref_path.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("description:"):
                answer = line.split(":", 1)[1].strip()
                break
        answer = answer or "No description available."

        return {
            "image": str(self.bundle / "images" / f"{image_id}.png"),
            "prompt": prompt,
            "answer": answer,
        }


def collate(batch: list[dict], processor: AutoProcessor, max_len: int) -> dict:
    texts, images, label_lens = [], [], []
    for b in batch:
        msgs_prompt = [{"role": "user", "content": [
            {"type": "image", "image": b["image"]},
            {"type": "text", "text": b["prompt"]},
        ]}]
        msgs_full = msgs_prompt + [{"role": "assistant", "content": [
            {"type": "text", "text": b["answer"]},
        ]}]

        prompt_text = processor.apply_chat_template(msgs_prompt, tokenize=False,
                                                   add_generation_prompt=True)
        full_text = processor.apply_chat_template(msgs_full, tokenize=False,
                                                  add_generation_prompt=False)
        image_inputs, _ = process_vision_info(msgs_full)

        p_enc = processor(text=[prompt_text], images=image_inputs,
                          return_tensors="pt", size=SIZE)
        f_enc = processor(text=[full_text], images=image_inputs,
                          return_tensors="pt", size=SIZE)

        texts.append(f_enc)
        images.append(image_inputs)
        label_lens.append(p_enc.input_ids.shape[1])

    # pad to the longest sequence in the batch
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    max_l = max(t.input_ids.shape[1] for t in texts)
    max_l = min(max_l, max_len)
    input_ids, attn, labels, pix, grid = [], [], [], [], []
    for enc, plen in zip(texts, label_lens):
        ids = enc.input_ids[0][:max_l]
        pad = max_l - ids.shape[0]
        input_ids.append(torch.cat([ids, torch.full((pad,), pad_id, dtype=ids.dtype)]))
        attn.append(torch.cat([torch.ones(ids.shape[0], dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
        lab = ids.clone()
        lab[:min(plen, lab.shape[0])] = -100
        labels.append(torch.cat([lab, torch.full((pad,), -100, dtype=lab.dtype)]))
        pix.append(enc["pixel_values"])
        grid.append(enc["image_grid_thw"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attn),
        "labels": torch.stack(labels),
        "pixel_values": torch.cat(pix),
        "image_grid_thw": torch.cat(grid),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--out", default="lora_gastrovl")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=200,
                    help="write an adapter checkpoint every N optimizer steps")
    ap.add_argument("--init-adapter", default="",
                    help="resume from a saved adapter checkpoint")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    meta = pd.read_csv(bundle / "train.csv", dtype=str)
    if args.limit:
        meta = meta.head(args.limit)
    print(f"training pairs: {len(meta)}")

    print("loading model ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(args.model)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    if args.init_adapter:
        from peft import PeftModel
        print(f"resuming from {args.init_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = PairDataset(bundle, meta)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    collate_fn=partial(collate, processor=processor,
                                       max_len=args.max_len))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    steps_per_epoch = max(1, len(dl) // args.accum)
    total = steps_per_epoch * args.epochs
    print(f"steps: {steps_per_epoch}/epoch x {args.epochs} epochs = {total}", flush=True)

    model.train()
    global_step = 0
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        running = 0.0
        for i, batch in enumerate(dl):
            batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}
            loss = model(**batch).loss / args.accum
            loss.backward()
            running += loss.item() * args.accum
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
                global_step += 1
                if global_step % 10 == 0:
                    el = time.time() - t0
                    print(f"  epoch {epoch+1} step {global_step}/{total} "
                          f"loss {running/ (i+1):.4f} {el/global_step:.1f}s/step", flush=True)
                if args.save_every and global_step % args.save_every == 0:
                    ck = out / f"step_{global_step}"
                    model.save_pretrained(ck)
                    print(f"  checkpoint: {ck}", flush=True)
        print(f"epoch {epoch+1} done | mean loss {running/max(1,len(dl)):.4f}", flush=True)

    model.save_pretrained(out)
    processor.save_pretrained(out)
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"\nadapter saved: {out.resolve()}")
    print("evaluate with:")
    print(f"  python run_baseline.py --adapter {out} --out results_lora.csv")


if __name__ == "__main__":
    main()
