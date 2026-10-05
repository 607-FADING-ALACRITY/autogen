"""Builds the snapshot workflow: generation tuned for the casual phone/analog-snapshot look.

Recipe taken from a top-rated Krea 2 realism image (civitai.red/images/144541104): the Realistic Snapshot
LoRA at 1.5, euler / simple / 10 steps / CFG 1, ~1 MP 9:16, a plain background and an analog-film style
line in the prompt. That image's checkpoint (Dark Beast 5C) is deliberately NOT used: its model card says
it is tuned toward a younger character age range. FinePorn v5 + the creator's LoRA replace it.

Writes:
  ../snapshot_api.json      API format (for POST /prompt)
  ./snapshot.spec.json      API prompt + layout; ../../tools/to_ui.mjs turns it into ../snapshot.json
"""

import importlib.util
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent
SPEC_DIR = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "postprocess_builder", OUT_DIR.parent / "postprocess" / "tools" / "build_workflow.py")
postprocess = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(postprocess)
node = postprocess.node

BASE_MODEL = "finepornV5INT8FP8_v5FP8.safetensors"               # diffusion_models
KREA2_TE = "qwen3vl_4b_fp8_scaled.safetensors"                   # text_encoders
QWEN_VAE = "qwen_image_vae.safetensors"                          # vae
CHARACTER_LORA = "character_lora.safetensors"                    # loras: the creator's likeness LoRA
SNAPSHOT_LORA = "RealisticSnapshotKrea2V2.safetensors"           # loras: Realistic Snapshot (Krea 2 v1.0)

AGE = 25
WIDTH, HEIGHT = 768, 1376
PROMPT = (
    f"A casual snapshot of a {AGE}-year-old woman with long dark hair against a plain white wall. She is wearing "
    "a fitted white top; her gaze is turned to the side, her expression is neutral with a slight smile, her hair "
    "is tousled. The photo is an analog shot in the style of the '80s, with slight film grain and muted colors."
)


def build():
    p = {}
    p["1"] = node("UNETLoader", "Base model (FinePorn v5)", unet_name=BASE_MODEL, weight_dtype="default")
    p["2"] = node("LoraLoaderModelOnly", "Character LoRA", model=["1", 0], lora_name=CHARACTER_LORA,
                  strength_model=0.85)
    p["3"] = node("LoraLoaderModelOnly", "Realistic Snapshot LoRA", model=["2", 0], lora_name=SNAPSHOT_LORA,
                  strength_model=1.5)
    p["4"] = node("CLIPLoader", "Krea 2 text encoder (Qwen3-VL 4B)", clip_name=KREA2_TE, type="krea2",
                  device="default")
    p["5"] = node("VAELoader", "VAE", vae_name=QWEN_VAE)
    p["10"] = node("CLIPTextEncode", "Prompt (trigger word + her real age)", text=PROMPT, clip=["4", 0])
    p["11"] = node("ConditioningZeroOut", "Negative (ignored at CFG 1)", conditioning=["10", 0])
    p["12"] = node("EmptySD3LatentImage", "Size (9:16, ~1 MP)", width=WIDTH, height=HEIGHT, batch_size=1)
    p["13"] = node("KSampler", "Generate (euler / simple / 10 steps / CFG 1)", model=["3", 0], seed=0, steps=10,
                   cfg=1.0, sampler_name="euler", scheduler="simple", positive=["10", 0], negative=["11", 0],
                   latent_image=["12", 0], denoise=1.0)
    p["14"] = node("VAEDecode", "Decode", samples=["13", 0], vae=["5", 0])
    p["15"] = node("PostFXSaveJPEG", "Save JPEG", images=["14", 0], filename_prefix="snapshot/gen", quality=92,
                   chroma_subsampling="4:2:0 (smallest file)", progressive=True, ai_disclosure="AI-generated")
    return p


WIDTHS = {"1": 340, "2": 340, "3": 340, "4": 330, "5": 330, "10": 460, "13": 320, "15": 420}

SETUP_NOTE = f"""## Snapshot: phone / analog-snapshot generation

The recipe behind a top-rated Krea 2 realism image: **Realistic Snapshot LoRA at 1.5**, euler / simple / **10 steps / CFG 1**, **{WIDTH}×{HEIGHT}**, a plain background and an analog-film line in the prompt.

**Before the first run**
- Pick your file in **Character LoRA** and put its trigger word at the start of the prompt.
- Set her **real age** in the prompt (currently {AGE}).

**What sells the realism:** simple settings (plain wall, bedroom, bathroom mirror) with nothing to render wrong, one light source you describe, and the "analog shot… slight film grain and muted colors" line. Busy scenes with signs and crowds undo it.

**Then:** run the output through the realism pass if the skin needs more texture.
"""

TUNING_NOTE = """## Tuning

- **Look too strong / faces drift** → *Realistic Snapshot LoRA* 1.0–1.2 (1.5 is the author's best; 1.0 does little).
- **Likeness weak** → *Character LoRA* 0.95, or Snapshot LoRA down to 1.2.
- **Sharper / more detail** → 1040×1520 instead of 768×1376 (loses some of the phone softness).
- **SFW promo content** → swap the base model to Realism by Stable Yogi.
"""


def main():
    prompt = build()
    rows = [
        [{"note": {"title": "Read me", "text": SETUP_NOTE, "width": 620, "height": 330}},
         {"title": "1 · Models", "color": "#3f789e", "columns": [["1", "2", "3"], ["4", "5"]]},
         {"title": "2 · Generate", "color": "#b06634", "columns": [["10", "11"], ["12", "13"], ["14", "15"]]},
         {"note": {"title": "Tuning", "text": TUNING_NOTE, "width": 520, "height": 220}}],
    ]
    laid_out = {i for row in rows for block in row for col in block.get("columns", []) for i in col}
    if laid_out != set(prompt):
        raise SystemExit(f"layout/prompt mismatch: missing {sorted(set(prompt) - laid_out)}, "
                         f"extra {sorted(laid_out - set(prompt))}")
    (OUT_DIR / "snapshot_api.json").write_text(json.dumps(prompt, indent=2) + "\n")
    (SPEC_DIR / "snapshot.spec.json").write_text(
        json.dumps({"prompt": prompt, "widths": WIDTHS, "rows": rows}, indent=2) + "\n")
    print(f"wrote snapshot_api.json + snapshot.spec.json ({len(prompt)} nodes)")


if __name__ == "__main__":
    main()
