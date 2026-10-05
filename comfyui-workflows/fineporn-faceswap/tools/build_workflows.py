"""Builds the FinePorn v5 face-swap workflows.

Writes:
  ../fineporn_faceswap_api.json   generate with FinePorn v5 -> head swap -> FinePorn texture pass (API format)
  ../faceswap_only_api.json       head swap onto an existing image                                (API format)
  ./*.spec.json                   API prompt + layout; ../../tools/to_ui.mjs turns these into the drag-and-drop UI files

Run:  python build_workflows.py
Then, with ComfyUI + the three custom node packs running on :8188:
      node ../../tools/to_ui.mjs fineporn_faceswap.spec.json ../fineporn_faceswap.json
      node ../../tools/to_ui.mjs faceswap_only.spec.json ../faceswap_only.json
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent
SPEC_DIR = Path(__file__).resolve().parent

# ---- Model files. Names must match what sits in ComfyUI/models/<folder>/ ---------------------
FINEPORN_UNET = "finepornV5INT8FP8_v5FP8.safetensors"                      # diffusion_models
KREA2_TE = "qwen3vl_4b_fp8_scaled.safetensors"                              # text_encoders
QWEN_VAE = "qwen_image_vae.safetensors"                                     # vae (same file for Krea 2 and Qwen Edit)
QWEN_EDIT_UNET = "qwen_image_edit_2511_fp8mixed.safetensors"                # diffusion_models
QWEN_EDIT_TE = "qwen_2.5_vl_7b_fp8_scaled.safetensors"                      # text_encoders
LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"  # loras
BFS_LORA = "bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors"       # loras
FACE_DETECTOR = "bbox/face_yolov8m.pt"                                      # ultralytics/bbox

# ---- Stage 1: FinePorn v5 author settings (euler / simple / 12 steps / CFG 1) -----------------
GEN_WIDTH, GEN_HEIGHT = 1040, 1520
GEN_STEPS = 12
GEN_PROMPT = (
    "this is an amateur photo taken from smartphone, casual photo. The lighting is soft and diffused.\n\n"
    "A 26 year old woman with long dark brown hair sitting on the edge of a bed in a sunlit apartment "
    "bedroom, facing the camera, relaxed smile, white tank top, natural skin texture."
)

# ---- Stage 2: head swap (Qwen-Image-Edit-2511 + Lightning 4-step + BFS head swap V5) ----------
SWAP_STEPS = 4
SWAP_SHIFT = 3.1
SWAP_PROMPT = (
    "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. "
    "remove the head from Picture 1 completely and replace it with the head from Picture 2, strictly "
    "preserving the hair, eye color, and nose structure of Picture 2. copy the eye direction, head rotation, "
    "and micro-expressions from Picture 1."
)
HEAD_GROW_EXPR = "round(max(c - a, d - b) * 0.30)"   # mask grows 30% of the face box: covers hair + ears
TARGET_CONTEXT = 1.6                                  # crop = head mask + 60% context each side
REFERENCE_CONTEXT = 1.8
CROP_SIZE = 1024

# ---- Stage 3: FinePorn texture pass over the swapped head -------------------------------------
HARMONIZE_DENOISE = 0.20
HARMONIZE_PROMPT = (
    "this is an amateur photo taken from smartphone, casual photo. Close-up of a woman's face and hair, "
    "natural skin texture with visible pores, soft diffused lighting, sharp focus."
)


def node(class_type, title, **inputs):
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def crop(image, mask, context, expand_px, blend_px):
    return {
        "image": image,
        "mask": mask,
        "downscale_algorithm": "bilinear",
        "upscale_algorithm": "bicubic",
        "preresize": False,
        "preresize_mode": "ensure minimum resolution",
        "preresize_min_width": 1024,
        "preresize_min_height": 1024,
        "preresize_max_width": 16384,
        "preresize_max_height": 16384,
        "mask_fill_holes": True,
        "mask_expand_pixels": expand_px,
        "mask_invert": False,
        "mask_blend_pixels": blend_px,
        "mask_hipass_filter": 0.1,
        "extend_for_outpainting": False,
        "extend_up_factor": 1.0,
        "extend_down_factor": 1.0,
        "extend_left_factor": 1.0,
        "extend_right_factor": 1.0,
        "context_from_mask_extend_factor": context,
        "output_resize_to_target_size": True,
        "output_target_width": CROP_SIZE,
        "output_target_height": CROP_SIZE,
        "output_padding": "32",
        "device_mode": "gpu (much faster)",
    }


def add_face_detect(p, first_id, image):
    """Adds detect -> keep largest face -> mask. Returns (segs_ref, mask_ref)."""
    det, keep, mask = str(first_id), str(first_id + 1), str(first_id + 2)
    p[det] = node("BboxDetectorSEGS", "Detect faces", bbox_detector=["30", 0], image=image, threshold=0.5,
                  dilation=0, crop_factor=3.0, drop_size=10, labels="all")
    p[keep] = node("ImpactSEGSOrderedFilter", "Keep largest face", segs=[det, 0], target="area(=w*h)",
                   order=True, take_start=0, take_count=1)
    p[mask] = node("SegsToCombinedMask", "Face mask", segs=[keep, 0])
    return [keep, 0], [mask, 0]


def build(mode):
    """mode: 'generate' (FinePorn -> swap -> texture pass) or 'swap_only' (any image -> swap)."""
    p = {}

    # Shared loaders
    p["12"] = node("VAELoader", "VAE (shared: Krea 2 + Qwen Edit)", vae_name=QWEN_VAE)
    p["30"] = node("UltralyticsDetectorProvider", "Face detector", model_name=FACE_DETECTOR)

    # ---- 1. Target image --------------------------------------------------------------------
    if mode == "generate":
        p["10"] = node("UNETLoader", "FinePorn v5 (Krea 2)", unet_name=FINEPORN_UNET, weight_dtype="default")
        p["11"] = node("CLIPLoader", "Krea 2 text encoder (Qwen3-VL 4B)", clip_name=KREA2_TE, type="krea2",
                       device="default")
        p["13"] = node("CLIPTextEncode", "Prompt", text=GEN_PROMPT, clip=["11", 0])
        p["14"] = node("ConditioningZeroOut", "Negative (ignored at CFG 1)", conditioning=["13", 0])
        p["15"] = node("EmptySD3LatentImage", "Size", width=GEN_WIDTH, height=GEN_HEIGHT, batch_size=1)
        p["16"] = node("KSampler", "Generate (euler / simple / 12 steps / CFG 1)", model=["10", 0], seed=0,
                       steps=GEN_STEPS, cfg=1.0, sampler_name="euler", scheduler="simple", positive=["13", 0],
                       negative=["14", 0], latent_image=["15", 0], denoise=1.0)
        p["17"] = node("VAEDecode", "Decode", samples=["16", 0], vae=["12", 0])
        p["18"] = node("SaveImage", "Save: base generation", images=["17", 0], filename_prefix="faceswap/base")
    else:
        p["17"] = node("LoadImage", "Target image (body to keep)", image="target.png")
    target = ["17", 0]

    # ---- 2. Creator reference head ------------------------------------------------------------
    p["20"] = node("LoadImage", "Creator reference photo", image="creator_reference.png")
    _, ref_mask = add_face_detect(p, 21, ["20", 0])
    p["24"] = node("InpaintCropImproved", "Crop reference head", **crop(["20", 0], ref_mask, REFERENCE_CONTEXT, 0, 0))
    p["25"] = node("PreviewImage", "Reference crop (Picture 2)", images=["24", 1])

    # ---- 3. Target head: detect, size the mask relative to the face, crop ----------------------
    target_segs, target_mask = add_face_detect(p, 31, target)
    p["34"] = node("ImpactDecomposeSEGS", "Face box", segs=target_segs)
    p["35"] = node("ImpactFrom_SEG_ELT", "Face box", seg_elt=["34", 1])
    p["36"] = node("ImpactFrom_SEG_ELT_bbox", "Face box coords", bbox=["35", 4])
    p["37"] = node("ComfyMathExpression", "Head mask grow (px) = 30% of face size", expression=HEAD_GROW_EXPR,
                   **{"values.a": ["36", 0], "values.b": ["36", 1], "values.c": ["36", 2], "values.d": ["36", 3]})
    p["38"] = node("InpaintCropImproved", "Crop target head",
                   **crop(target, target_mask, TARGET_CONTEXT, ["37", 1], 32))
    p["39"] = node("PreviewImage", "Target crop (Picture 1)", images=["38", 1])

    # ---- 4. Head swap -----------------------------------------------------------------------
    p["40"] = node("UNETLoader", "Qwen-Image-Edit-2511", unet_name=QWEN_EDIT_UNET, weight_dtype="default")
    p["41"] = node("LoraLoaderModelOnly", "Lightning 4-step LoRA", model=["40", 0], lora_name=LIGHTNING_LORA,
                   strength_model=1.0)
    p["42"] = node("LoraLoaderModelOnly", "BFS head swap V5 LoRA", model=["41", 0], lora_name=BFS_LORA,
                   strength_model=1.0)
    p["43"] = node("ModelSamplingAuraFlow", "Shift", model=["42", 0], shift=SWAP_SHIFT, sampling="flow")
    p["44"] = node("CFGNorm", "CFG norm", model=["43", 0], strength=1.0, pre_cfg=False)
    p["45"] = node("CLIPLoader", "Qwen 2.5 VL 7B text encoder", clip_name=QWEN_EDIT_TE, type="qwen_image",
                   device="default")
    p["46"] = node("TextEncodeQwenImageEditPlus", "Swap instruction (Picture 1 = target, Picture 2 = creator)",
                   clip=["45", 0], prompt=SWAP_PROMPT, vae=["12", 0], image1=["38", 1], image2=["24", 1])
    p["47"] = node("FluxKontextMultiReferenceLatentMethod", "Reference method", conditioning=["46", 0],
                   reference_latents_method="index_timestep_zero")
    p["48"] = node("TextEncodeQwenImageEditPlus", "Negative (ignored at CFG 1)", clip=["45", 0], prompt="",
                   vae=["12", 0], image1=["38", 1], image2=["24", 1])
    p["49"] = node("FluxKontextMultiReferenceLatentMethod", "Reference method", conditioning=["48", 0],
                   reference_latents_method="index_timestep_zero")
    p["50"] = node("VAEEncode", "Encode target crop", pixels=["38", 1], vae=["12", 0])
    p["51"] = node("SetLatentNoiseMask", "Only repaint the head", samples=["50", 0], mask=["38", 2])
    p["52"] = node("KSampler", "Swap (4 steps / CFG 1)", model=["44", 0], seed=0, steps=SWAP_STEPS, cfg=1.0,
                   sampler_name="euler", scheduler="simple", positive=["47", 0], negative=["49", 0],
                   latent_image=["51", 0], denoise=1.0)
    p["53"] = node("VAEDecode", "Decode swap", samples=["52", 0], vae=["12", 0])
    p["54"] = node("InpaintStitchImproved", "Paste head back", stitcher=["38", 0], inpainted_image=["53", 0])
    if mode == "generate":
        p["55"] = node("SaveImage", "Save: swap (before texture pass)", images=["54", 0],
                       filename_prefix="faceswap/swap_raw")
    else:
        p["55"] = node("SaveImage", "Save: swap", images=["54", 0], filename_prefix="faceswap/swap")

    # ---- 5. FinePorn texture pass over the new head (generate mode only) ----------------------
    if mode == "generate":
        p["60"] = node("InpaintCropImproved", "Re-crop swapped head",
                       **crop(["54", 0], target_mask, TARGET_CONTEXT, ["37", 1], 32))
        p["61"] = node("VAEEncode", "Encode", pixels=["60", 1], vae=["12", 0])
        p["62"] = node("SetLatentNoiseMask", "Only touch the head", samples=["61", 0], mask=["60", 2])
        p["63"] = node("CLIPTextEncode", "Texture prompt", text=HARMONIZE_PROMPT, clip=["11", 0])
        p["64"] = node("ConditioningZeroOut", "Negative (ignored at CFG 1)", conditioning=["63", 0])
        p["65"] = node("KSampler", f"Texture pass (denoise {HARMONIZE_DENOISE:.2f})", model=["10", 0], seed=0,
                       steps=GEN_STEPS, cfg=1.0, sampler_name="euler", scheduler="simple", positive=["63", 0],
                       negative=["64", 0], latent_image=["62", 0], denoise=HARMONIZE_DENOISE)
        p["66"] = node("VAEDecode", "Decode", samples=["65", 0], vae=["12", 0])
        p["67"] = node("InpaintStitchImproved", "Paste back", stitcher=["60", 0], inpainted_image=["66", 0])
        p["68"] = node("SaveImage", "Save: final", images=["67", 0], filename_prefix="faceswap/final")
    return p


# ---- Layout: groups of columns; to_ui.mjs stacks nodes using their real rendered sizes -------
WIDTHS = {
    "10": 340, "11": 340, "12": 340, "13": 440, "16": 320, "18": 380,
    "20": 320, "30": 320, "24": 320, "25": 320,
    "37": 340, "38": 320, "39": 320,
    "40": 340, "41": 340, "42": 340, "45": 340, "46": 440, "48": 440, "52": 320, "55": 380,
    "60": 320, "63": 440, "65": 320, "68": 380,
}
G_GENERATE = {"title": "1 · Generate — FinePorn v5 (Krea 2)", "color": "#3f789e",
              "columns": [["10", "11", "12"], ["13", "14", "15"], ["16", "17"], ["18"]]}
G_TARGET_IMAGE = {"title": "1 · Target image", "color": "#3f789e", "columns": [["17", "12"]]}
G_REFERENCE = {"title": "2 · Creator reference — consented, KYC-verified creator only", "color": "#a1309b",
               "columns": [["20", "30"], ["21", "22", "23"], ["24"], ["25"]]}
G_TARGET_HEAD = {"title": "3 · Find + crop target head", "color": "#88aa88",
                 "columns": [["31", "32", "33"], ["34", "35", "36", "37"], ["38"], ["39"]]}
G_SWAP = {"title": "4 · Head swap — Qwen-Image-Edit-2511 + Lightning + BFS V5", "color": "#b06634",
          "columns": [["40", "41", "42", "43", "44", "45"], ["46", "47", "48", "49"], ["50", "51", "52"],
                      ["53", "54", "55"]]}
G_TEXTURE = {"title": "5 · Texture pass — FinePorn at low denoise", "color": "#3f789e",
             "columns": [["60"], ["61", "62", "63", "64"], ["65", "66", "67"], ["68"]]}


def note(title, text, width, height):
    return {"note": {"title": title, "text": text, "width": width, "height": height}}


SETUP_NOTE = """## FinePorn v5 → head swap

**Before running this on anyone:** use only the creator's own reference photos, and only with a signed AI-likeness release on file that covers explicit AI content. Label the output as AI on OnlyFans/Fanvue. See README.md.

**Models** (folder under `ComfyUI/models/`)
- `diffusion_models/` finepornV5INT8FP8_v5FP8.safetensors · qwen_image_edit_2511_fp8mixed.safetensors
- `text_encoders/` qwen3vl_4b_fp8_scaled.safetensors · qwen_2.5_vl_7b_fp8_scaled.safetensors
- `vae/` qwen_image_vae.safetensors
- `loras/` Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors · bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors
- `ultralytics/bbox/` face_yolov8m.pt

**Custom nodes:** ComfyUI-Impact-Pack, ComfyUI-Impact-Subpack, ComfyUI-Inpaint-CropAndStitch. ComfyUI ≥ 0.26 (Krea 2 support).

**Run:** write the prompt in group 1, load the creator photo in group 2, queue. Outputs go to `output/faceswap/`: base, swap_raw, final.
"""

SWAP_NOTE = """## Tuning

- **Likeness weak** → raise *BFS head swap V5 LoRA* to 1.1–1.3. Use a sharp, front-facing, well-lit reference.
- **Seam or pasted look** → raise the 0.30 in *Head mask grow*, or raise `mask_blend_pixels` on *Crop target head*.
- **Hair wrong** → BFS swaps the whole head, including hair. Describe the creator's real hair in the group-1 prompt so the hair outside the mask matches.
- **Swap looks too smooth or "AI"** → raise *Texture pass* denoise to 0.25–0.30. Above 0.35 it starts to drift from the creator's face.
- **No face found** → lower the *Detect faces* threshold to 0.3. Strong profiles may not detect; regenerate.
- **Better quality, slower** → set *Lightning 4-step LoRA* strength to 0 and the swap sampler to 20 steps / CFG 2.5.
"""


SETUP_NOTE_SWAP_ONLY = """## Head swap onto any image

**Before running this on anyone:** use only the creator's own reference photos, and only with a signed AI-likeness release on file that covers explicit AI content. Label the output as AI on OnlyFans/Fanvue. See README.md.

**Models** (folder under `ComfyUI/models/`)
- `diffusion_models/` qwen_image_edit_2511_fp8mixed.safetensors
- `text_encoders/` qwen_2.5_vl_7b_fp8_scaled.safetensors
- `vae/` qwen_image_vae.safetensors
- `loras/` Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors · bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors
- `ultralytics/bbox/` face_yolov8m.pt

**Custom nodes:** ComfyUI-Impact-Pack, ComfyUI-Impact-Subpack, ComfyUI-Inpaint-CropAndStitch.

**Run:** load the target image in group 1 and the creator photo in group 2, then queue. Output goes to `output/faceswap/swap_*.png`.
"""

SWAP_NOTE_SWAP_ONLY = """## Tuning

- **Likeness weak** → raise *BFS head swap V5 LoRA* to 1.1–1.3. Use a sharp, front-facing, well-lit reference.
- **Seam or pasted look** → raise the 0.30 in *Head mask grow*, or raise `mask_blend_pixels` on *Crop target head*.
- **Hair wrong** → BFS swaps the whole head, including hair. Pick targets whose hair already roughly matches the creator's.
- **No face found** → lower the *Detect faces* threshold to 0.3.
- **Better quality, slower** → set *Lightning 4-step LoRA* strength to 0 and the swap sampler to 20 steps / CFG 2.5.
"""


def main():
    variants = {
        "fineporn_faceswap": (build("generate"), [
            [note("Read me", SETUP_NOTE, 720, 330), G_GENERATE, G_REFERENCE],
            [G_TARGET_HEAD, G_SWAP, G_TEXTURE, note("Tuning", SWAP_NOTE, 560, 330)],
        ]),
        "faceswap_only": (build("swap_only"), [
            [note("Read me", SETUP_NOTE_SWAP_ONLY, 720, 300), G_TARGET_IMAGE, G_REFERENCE],
            [G_TARGET_HEAD, G_SWAP, note("Tuning", SWAP_NOTE_SWAP_ONLY, 560, 290)],
        ]),
    }
    for name, (prompt, rows) in variants.items():
        laid_out = {i for row in rows for block in row for col in block.get("columns", []) for i in col}
        if laid_out != set(prompt):
            raise SystemExit(f"{name}: layout/prompt mismatch: missing {sorted(set(prompt) - laid_out)}, "
                             f"extra {sorted(laid_out - set(prompt))}")
        widths = {k: v for k, v in WIDTHS.items() if k in prompt}
        (OUT_DIR / f"{name}_api.json").write_text(json.dumps(prompt, indent=2) + "\n")
        (SPEC_DIR / f"{name}.spec.json").write_text(
            json.dumps({"prompt": prompt, "widths": widths, "rows": rows}, indent=2) + "\n")
        print(f"wrote {name}_api.json + {name}.spec.json ({len(prompt)} nodes)")


if __name__ == "__main__":
    main()
