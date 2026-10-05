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


# Node ids of the finishing chain in postprocess.json. Other workflows reuse add_finish() with their own ids.
FINISH_IDS = {"bg_model": "10", "mask": "11", "depth_model": "12", "depth": "13", "depth_map": "14",
              "blur": "15", "lut": "20", "lens": "21", "sharpen": "30", "grain": "31", "save": "40"}


def add_finish(p, image, ids=FINISH_IDS, blur_enabled=True, filename_prefix="postfx/final",
               lut_strength=0.75, grain_amount=0.3, quality=92, subsampling="4:4:4 (sharpest color)",
               include_blur=True):
    """Adds portrait blur (optional) -> LUT -> lens -> sharpen -> grain -> JPEG after `image`.
    include_blur=False leaves the portrait group out entirely (no BiRefNet / Depth Anything files needed)."""
    i = ids
    if not include_blur:
        _add_grade_to_save(p, image, i, filename_prefix, lut_strength, grain_amount, quality, subsampling)
        return
    # Portrait mode: subject mask (BiRefNet) + depth (Depth Anything 3), both skipped when disabled
    p[i["bg_model"]] = node("LoadBackgroundRemovalModel", "Subject model (BiRefNet)", bg_removal_name=BG_REMOVAL_MODEL)
    p[i["mask"]] = node("RemoveBackground", "Subject mask", bg_removal_model=[i["bg_model"], 0], image=image)
    p[i["depth_model"]] = node("LoadDA3Model", "Depth model (Depth Anything 3)", model_name=DEPTH_MODEL,
                               weight_dtype="default")
    p[i["depth"]] = node("DA3Inference", "Estimate depth", da3_model=[i["depth_model"], 0], image=image,
                         resolution=504, resize_method="upper_bound_resize", mode="mono")
    p[i["depth_map"]] = node("DA3Render", "Depth map (near = white)", da3_geometry=[i["depth"], 0], output="depth",
                             **{"output.normalization": "v2_style", "output.apply_sky_clip": False})
    p[i["blur"]] = node("PostFXPortraitBlur", "Portrait blur", image=image, enabled=blur_enabled, blur_strength=1.5,
                        falloff=0.3, highlight_bloom=0.35, edge_softness=1.5, subject_mask=[i["mask"], 0],
                        depth=[i["depth_map"], 0])
    _add_grade_to_save(p, [i["blur"], 0], i, filename_prefix, lut_strength, grain_amount, quality, subsampling)


def _add_grade_to_save(p, image, i, filename_prefix, lut_strength, grain_amount, quality, subsampling):
    p[i["lut"]] = node("PostFXApplyLUT", "LUT grade", image=image, lut_name=DEFAULT_LUT, strength=lut_strength)
    p[i["lens"]] = node("PostFXLens", "Lens", image=[i["lut"], 0], chromatic_aberration=1.0, vignette=0.25)
    p[i["sharpen"]] = node("PostFXSharpen", "Sharpen", image=[i["lens"], 0], amount=0.5, radius=1.0, threshold=0.02)
    p[i["grain"]] = node("PostFXFilmGrain", "Grain", image=[i["sharpen"], 0], amount=grain_amount, size=1.5,
                         color=0.15, seed=0)
    p[i["save"]] = node("PostFXSaveJPEG", "Save JPEG", images=[i["grain"], 0], filename_prefix=filename_prefix,
                        quality=quality, chroma_subsampling=subsampling, progressive=True,
                        ai_disclosure="AI-generated")


def finish_groups(ids=FINISH_IDS, first_number=2, include_blur=True):
    """Layout groups for add_finish(), numbered from first_number."""
    i, n = ids, first_number
    groups = []
    if include_blur:
        groups.append({"title": f"{n} · Portrait mode (optional)", "color": "#a1309b",
                       "columns": [[i["bg_model"], i["mask"]], [i["depth_model"], i["depth"], i["depth_map"]],
                                   [i["blur"]]]})
        n += 1
    return groups + [
        {"title": f"{n} · Grade + lens", "color": "#b06634", "columns": [[i["lut"]], [i["lens"]]]},
        {"title": f"{n + 1} · Sharpen + grain", "color": "#88aa88", "columns": [[i["sharpen"]], [i["grain"]]]},
        {"title": f"{n + 2} · Export", "color": "#3f789e", "columns": [[i["save"]]]},
    ]


def finish_widths(ids=FINISH_IDS):
    return {ids[k]: w for k, w in (("bg_model", 320), ("depth_model", 340), ("depth", 320), ("depth_map", 320),
                                   ("blur", 340), ("lut", 320), ("lens", 320), ("sharpen", 300), ("grain", 300),
                                   ("save", 420))}


def build():
    p = {"1": node("LoadImage", "Input image", image="example.png")}
    add_finish(p, ["1", 0])
    return p


WIDTHS = {"1": 360, **finish_widths()}

SETUP_NOTE = """## Post-processing: portrait blur → LUT → lens → sharpen → grain → JPEG

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
- **Lens** → *chromatic_aberration* 0 and *vignette* 0 turn it off.
- **Crunchy skin** → raise *Sharpen* threshold to 0.04, or amount to 0.3.
- **Grain** → 0.2 subtle, 0.3 phone, 0.5+ film. Raise *size* for coarser grain.
- **Smaller files** → quality 88 and 4:2:0.
"""


def main():
    prompt = build()
    groups = finish_groups()
    rows = [
        [{"note": {"title": "Read me", "text": SETUP_NOTE, "width": 640, "height": 250}},
         {"title": "1 · Input", "color": "#3f789e", "columns": [["1"]]}, groups[0]],
        [*groups[1:], {"note": {"title": "Tuning", "text": TUNING_NOTE, "width": 520, "height": 230}}],
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
