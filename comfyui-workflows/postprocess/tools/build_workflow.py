"""Builds the post-processing workflow.

Writes:
  ../postprocess_api.json     API format (for POST /prompt)
  ./postprocess.spec.json     API prompt + layout; ../../tools/to_ui.mjs turns it into ../postprocess.json

Run:  python build_workflow.py
Then, with ComfyUI + the comfyui-postfx pack running on :8188:
      node ../../tools/to_ui.mjs postprocess.spec.json ../postprocess.json
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent
SPEC_DIR = Path(__file__).resolve().parent

BG_REMOVAL_MODEL = "birefnet.safetensors"                    # models/background_removal
DEPTH_MODEL = "depth_anything_3_mono_large.safetensors"      # models/geometry_estimation
DEFAULT_LUT = "warm_film.cube"


def node(class_type, title, **inputs):
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def build():
    p = {}
    p["1"] = node("LoadImage", "Input image", image="example.png")

    # Portrait mode: subject mask (BiRefNet) + depth (Depth Anything 3), both skipped when disabled
    p["10"] = node("LoadBackgroundRemovalModel", "Subject model (BiRefNet)", bg_removal_name=BG_REMOVAL_MODEL)
    p["11"] = node("RemoveBackground", "Subject mask", bg_removal_model=["10", 0], image=["1", 0])
    p["12"] = node("LoadDA3Model", "Depth model (Depth Anything 3)", model_name=DEPTH_MODEL, weight_dtype="default")
    p["13"] = node("DA3Inference", "Estimate depth", da3_model=["12", 0], image=["1", 0], resolution=504,
                   resize_method="upper_bound_resize", mode="mono")
    p["14"] = node("DA3Render", "Depth map (near = white)", da3_geometry=["13", 0], output="depth",
                   **{"output.normalization": "v2_style", "output.apply_sky_clip": False})
    p["15"] = node("PostFXPortraitBlur", "Portrait blur", image=["1", 0], enabled=True, blur_strength=1.5,
                   falloff=0.3, highlight_bloom=0.35, edge_softness=1.5, subject_mask=["11", 0], depth=["14", 0])

    p["20"] = node("PostFXApplyLUT", "LUT grade", image=["15", 0], lut_name=DEFAULT_LUT, strength=0.75)
    p["30"] = node("PostFXSharpen", "Sharpen", image=["20", 0], amount=0.5, radius=1.0, threshold=0.02)
    p["31"] = node("PostFXFilmGrain", "Grain", image=["30", 0], amount=0.3, size=1.5, color=0.15, seed=0)
    p["40"] = node("PostFXSaveJPEG", "Save JPEG", images=["31", 0], filename_prefix="postfx/final", quality=92,
                   chroma_subsampling="4:4:4 (sharpest color)", progressive=True, ai_disclosure="AI-generated")
    return p


WIDTHS = {"1": 360, "10": 320, "12": 340, "13": 320, "14": 320, "15": 340, "20": 320, "30": 300, "31": 300,
          "40": 420}

SETUP_NOTE = """## Post-processing: portrait blur → LUT → sharpen → grain → JPEG

**Custom nodes:** copy `comfyui-postfx/` into `ComfyUI/custom_nodes/`. No pip installs.

**Models** (only needed while Portrait blur is on)
- `models/background_removal/` birefnet.safetensors
- `models/geometry_estimation/` depth_anything_3_mono_large.safetensors

**LUTs:** drop `.cube` files into `models/luts/`. Two starters ship with the pack.

**Run:** load an image in group 1 and queue. Output: `output/postfx/final_*.jpg`. The JPEG carries no prompt or workflow, and is tagged as AI-generated unless you change *ai_disclosure*.
"""

TUNING_NOTE = """## Tuning

- **Portrait blur off** → set *enabled* to false. Depth and subject models are skipped entirely.
- **More / less blur** → *blur_strength* (% of long side). Phone portrait mode ≈ 1–2.
- **Background goes soft too gradually** → lower *falloff* (0.15 = soft right behind the subject).
- **Hair edge looks cut out** → raise *edge_softness* to 3–4.
- **Grade too strong** → *LUT grade* strength 0.4–0.6.
- **Crunchy skin** → raise *Sharpen* threshold to 0.04, or amount to 0.3.
- **Grain** → 0.2 subtle, 0.3 phone, 0.5+ film. Raise *size* for coarser grain.
- **Smaller files** → quality 88 and 4:2:0.
"""


def main():
    prompt = build()
    rows = [
        [{"note": {"title": "Read me", "text": SETUP_NOTE, "width": 640, "height": 250}},
         {"title": "1 · Input", "color": "#3f789e", "columns": [["1"]]},
         {"title": "2 · Portrait mode (optional)", "color": "#a1309b",
          "columns": [["10", "11"], ["12", "13", "14"], ["15"]]}],
        [{"title": "3 · Grade", "color": "#b06634", "columns": [["20"]]},
         {"title": "4 · Sharpen + grain", "color": "#88aa88", "columns": [["30"], ["31"]]},
         {"title": "5 · Export", "color": "#3f789e", "columns": [["40"]]},
         {"note": {"title": "Tuning", "text": TUNING_NOTE, "width": 520, "height": 210}}],
    ]
    laid_out = {i for row in rows for block in row for col in block.get("columns", []) for i in col}
    if laid_out != set(prompt):
        raise SystemExit(f"layout/prompt mismatch: missing {sorted(set(prompt) - laid_out)}, "
                         f"extra {sorted(laid_out - set(prompt))}")
    (OUT_DIR / "postprocess_api.json").write_text(json.dumps(prompt, indent=2) + "\n")
    (SPEC_DIR / "postprocess.spec.json").write_text(
        json.dumps({"prompt": prompt, "widths": WIDTHS, "rows": rows}, indent=2) + "\n")
    print(f"wrote postprocess_api.json + postprocess.spec.json ({len(prompt)} nodes)")


if __name__ == "__main__":
    main()
