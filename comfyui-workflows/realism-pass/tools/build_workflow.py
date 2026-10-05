"""Builds the realism-pass workflow: adds real skin texture to a finished generation without changing
her face.

  1. Texture pass   upscale to ~2.4 MP, re-denoise 0.25 with a realism refiner + the character LoRA,
                    then Detail Transfer: only detail finer than ~3 px is taken from the refined image;
                    shape, features, makeup, color and light stay from the original. Eyes/lashes/brows
                    and lips are masked out entirely (MediaPipe landmarks).
  2. Face pass      1024px face crop, re-denoise 0.30, pasted back, then the same protected Detail Transfer.
                    Skipped automatically when no face is found.
  3. Finish         LUT (off) -> lens -> sharpen -> grain -> JPEG, from the post-processing chain.

Only needs comfyui-postfx + core ComfyUI nodes (no Impact Pack / CropAndStitch / depth models).

Writes:
  ../realism_pass_api.json     API format (for POST /prompt)
  ./realism_pass.spec.json     API prompt + layout; ../../tools/to_ui.mjs turns it into ../realism_pass.json
"""

import importlib.util
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent
SPEC_DIR = Path(__file__).resolve().parent

# Reuse the finishing chain from ../../postprocess/tools/build_workflow.py
_spec = importlib.util.spec_from_file_location(
    "postprocess_builder", OUT_DIR.parent / "postprocess" / "tools" / "build_workflow.py")
postprocess = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(postprocess)
node = postprocess.node

# ---- Models (names must match ComfyUI/models/<folder>/; pick yours in the loaders) -------------
REFINER = "realismByStableYogi_v30_fp8_scaled.safetensors"     # diffusion_models (Realism by Stable Yogi v3.0)
KREA2_TE = "qwen3vl_4b_fp8_scaled.safetensors"                  # text_encoders
QWEN_VAE = "qwen_image_vae.safetensors"                         # vae
CHARACTER_LORA = "character_lora.safetensors"                   # loras: the creator's likeness LoRA
FACE_MODEL = "mediapipe_face_fp32.safetensors"                  # detection

# ---- Settings ----------------------------------------------------------------------------------
AGE = 25            # set to the creator's real age
CHARACTER_STRENGTH = 0.85
TEXTURE_MEGAPIXELS = 2.4
TEXTURE_DENOISE = 0.25
FACE_DENOISE = 0.30
DETAIL_RADIUS = 3.0
STEPS = 12

TEXTURE_PROMPT = (
    f"amateur smartphone photo of a {AGE}-year-old woman. Natural skin texture: fine visible pores, subtle "
    "uneven skin tone, faint redness on knees, elbows and knuckles, matte skin without glossy highlights."
)
FACE_PROMPT = (
    f"close-up smartphone photo of a {AGE}-year-old woman's face. Natural skin texture with fine visible pores, "
    "subtle uneven skin tone, matte skin."
)

FINISH_IDS = {"lut": "50", "lens": "51", "sharpen": "52", "grain": "53", "save": "60"}


def sampler(model, positive, negative, latent, denoise, title):
    return node("KSampler", title, model=model, seed=0, steps=STEPS, cfg=1.0, sampler_name="euler",
                scheduler="simple", positive=positive, negative=negative, latent_image=latent, denoise=denoise)


def build():
    p = {}
    # 1. Input + models
    p["1"] = node("LoadImage", "Generated image", image="example.png")
    p["2"] = node("UNETLoader", "Refiner (Realism by Stable Yogi)", unet_name=REFINER, weight_dtype="default")
    p["3"] = node("LoraLoaderModelOnly", "Character LoRA", model=["2", 0], lora_name=CHARACTER_LORA,
                  strength_model=CHARACTER_STRENGTH)
    p["5"] = node("CLIPLoader", "Krea 2 text encoder (Qwen3-VL 4B)", clip_name=KREA2_TE, type="krea2",
                  device="default")
    p["6"] = node("VAELoader", "VAE", vae_name=QWEN_VAE)
    p["7"] = node("LoadMediaPipeFaceLandmarker", "Face landmark model", model_name=FACE_MODEL)

    # 2. Texture pass
    p["10"] = node("ImageScaleToTotalPixels", f"Upscale to {TEXTURE_MEGAPIXELS} MP", image=["1", 0],
                   upscale_method="lanczos", megapixels=TEXTURE_MEGAPIXELS, resolution_steps=16)
    p["11"] = node("CLIPTextEncode", "Texture prompt (set her real age)", text=TEXTURE_PROMPT, clip=["5", 0])
    p["12"] = node("ConditioningZeroOut", "Negative (ignored at CFG 1)", conditioning=["11", 0])
    p["13"] = node("VAEEncode", "Encode", pixels=["10", 0], vae=["6", 0])
    p["14"] = sampler(["3", 0], ["11", 0], ["12", 0], ["13", 0], TEXTURE_DENOISE,
                      f"Texture pass (denoise {TEXTURE_DENOISE:.2f})")
    p["15"] = node("VAEDecode", "Decode", samples=["14", 0], vae=["6", 0])
    p["17"] = node("MediaPipeFaceLandmarker", "Find faces", face_detection_model=["7", 0], image=["10", 0],
                   detector_variant="both", num_faces=0, min_confidence=0.65, missing_frame_fallback="empty")
    p["18"] = node("PostFXFaceFeatureMask", "Protect eyes, lashes, brows, lips", face_landmarks=["17", 0],
                   eyes_and_brows=True, lips=True, grow=0.12, feather=4.0)
    p["19"] = node("PostFXDetailTransfer", "Keep her look, take only fine texture", base=["10", 0],
                   detail=["15", 0], radius=DETAIL_RADIUS, strength=1.0, protect_mask=["18", 0])
    p["16"] = node("PreviewImage", "After texture pass", images=["19", 0])

    # 3. Face pass (skipped automatically when no face is found)
    p["20"] = node("PostFXFaceCrop", "Crop face to 1024px", image=["19", 0], face_landmarks=["17", 0],
                   context=1.6, size=1024, paste_grow=0.06)
    p["21"] = node("CLIPTextEncode", "Face prompt (set her real age)", text=FACE_PROMPT, clip=["5", 0])
    p["22"] = node("ConditioningZeroOut", "Negative (ignored at CFG 1)", conditioning=["21", 0])
    p["23"] = node("VAEEncode", "Encode face", pixels=["20", 0], vae=["6", 0])
    p["24"] = sampler(["3", 0], ["21", 0], ["22", 0], ["23", 0], FACE_DENOISE,
                      f"Face pass (denoise {FACE_DENOISE:.2f})")
    p["25"] = node("VAEDecode", "Decode face", samples=["24", 0], vae=["6", 0])
    p["26"] = node("PostFXFacePaste", "Paste face back", image=["19", 0], face_crop=["20", 1], face=["25", 0])
    p["27"] = node("PostFXDetailTransfer", "Keep her face, take only fine texture", base=["19", 0],
                   detail=["26", 0], radius=DETAIL_RADIUS, strength=1.0, protect_mask=["18", 0])

    # 4+. Camera finish (no portrait blur here; use the post-processing workflow for that)
    postprocess.add_finish(p, ["27", 0], ids=FINISH_IDS, include_blur=False, filename_prefix="realism/final",
                           lut_strength=0.0, grain_amount=0.3, quality=90, subsampling="4:2:0 (smallest file)")
    return p


WIDTHS = {"1": 340, "2": 340, "3": 340, "5": 330, "6": 330, "7": 330, "10": 300, "11": 420, "14": 320, "16": 300,
          "17": 320, "18": 320, "19": 340, "20": 320, "21": 420, "24": 320, "26": 300, "27": 340}

SETUP_NOTE = f"""## Realism pass: real skin texture, same face

Feed it a finished generation. It upscales to ~{TEXTURE_MEGAPIXELS} MP and re-renders skin with a realism refiner, gives the face its own 1024px pass, then finishes like a phone camera.

Her look is locked: only detail finer than ~{DETAIL_RADIUS:g} px is taken from the refiner, so face shape, features, makeup, colors and light stay from your image. Eyes, lashes, brows and lips are masked out completely.

**Before the first run**
- Pick your file in **Character LoRA**.
- Set her **real age** in both prompts (currently {AGE}).

**Needs:** comfyui-postfx; refiner (Realism by Stable Yogi v3.0), Krea 2 text encoder + VAE, mediapipe_face_fp32.

**Output:** `output/realism/final_*.jpg`.
"""

TUNING_NOTE = """## Tuning

- **Skin still too smooth** → *Detail Transfer* radius 4–5 on both, or *Texture pass* denoise 0.30.
- **Texture too strong / aged** → *Detail Transfer* radius 2, or strength 0.7.
- **Ghost edges or doubled hair strands** → lower that pass's denoise (0.2); the refine drifted out of alignment.
- **Brow or lash edge changed** → *Protect* grow 0.18.
- **Want texture on the lips too** → *Protect* lips = false.
- **Grade** → *LUT grade* strength 0.3–0.5 (off by default so colors stay yours).
- **Grain** → 0.2–0.4.
"""


def main():
    prompt = build()
    finish = postprocess.finish_groups(FINISH_IDS, first_number=4, include_blur=False)
    rows = [
        [{"note": {"title": "Read me", "text": SETUP_NOTE, "width": 640, "height": 360}},
         {"title": "1 · Input + models", "color": "#3f789e", "columns": [["1"], ["2", "3"], ["5", "6", "7"]]},
         {"title": "2 · Texture pass", "color": "#b06634",
          "columns": [["10", "13"], ["11", "12"], ["14", "15"], ["17", "18"], ["19", "16"]]}],
        [{"title": "3 · Face pass", "color": "#88aa88",
          "columns": [["20"], ["21", "22"], ["23", "24"], ["25", "26"], ["27"]]},
         *finish,
         {"note": {"title": "Tuning", "text": TUNING_NOTE, "width": 560, "height": 280}}],
    ]
    laid_out = {i for row in rows for block in row for col in block.get("columns", []) for i in col}
    if laid_out != set(prompt):
        raise SystemExit(f"layout/prompt mismatch: missing {sorted(set(prompt) - laid_out)}, "
                         f"extra {sorted(laid_out - set(prompt))}")
    finish_ids = set(FINISH_IDS.values())
    widths = {**WIDTHS, **{k: v for k, v in postprocess.finish_widths({**postprocess.FINISH_IDS, **FINISH_IDS}).items()
                           if k in finish_ids}}
    (OUT_DIR / "realism_pass_api.json").write_text(json.dumps(prompt, indent=2) + "\n")
    (SPEC_DIR / "realism_pass.spec.json").write_text(
        json.dumps({"prompt": prompt, "widths": widths, "rows": rows}, indent=2) + "\n")
    print(f"wrote realism_pass_api.json + realism_pass.spec.json ({len(prompt)} nodes)")


if __name__ == "__main__":
    main()
