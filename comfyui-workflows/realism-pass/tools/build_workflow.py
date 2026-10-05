"""Builds the realism-pass workflow: takes a finished FinePorn/Krea 2 generation and makes it read as a
real phone photo.

  1. Texture pass   upscale to ~2.4 MP, re-denoise 0.25 with the character LoRA + skin detail LoRA (0.3)
  2. Face pass      1024px face crop, re-denoise 0.30 with the character LoRA only, stitched back;
                    skipped entirely when no face is detected
  3. Finish         the post-processing chain: portrait blur (off) -> LUT -> lens -> sharpen -> grain -> JPEG

Writes:
  ../realism_pass_api.json     API format (for POST /prompt)
  ./realism_pass.spec.json     API prompt + layout; ../../tools/to_ui.mjs turns it into ../realism_pass.json

Run:  python build_workflow.py
Then, with ComfyUI + Impact Pack/Subpack + Inpaint-CropAndStitch + comfyui-postfx running on :8188:
      node ../../tools/to_ui.mjs realism_pass.spec.json ../realism_pass.json
"""

import importlib.util
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent
SPEC_DIR = Path(__file__).resolve().parent

# Reuse the post-processing chain from ../../postprocess/tools/build_workflow.py
_spec = importlib.util.spec_from_file_location(
    "postprocess_builder", OUT_DIR.parent / "postprocess" / "tools" / "build_workflow.py")
postprocess = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(postprocess)
node = postprocess.node

# ---- Models (names must match ComfyUI/models/<folder>/; pick yours in the loaders) -------------
FINEPORN_UNET = "finepornV5INT8FP8_v5FP8.safetensors"          # diffusion_models
KREA2_TE = "qwen3vl_4b_fp8_scaled.safetensors"                  # text_encoders
QWEN_VAE = "qwen_image_vae.safetensors"                         # vae
CHARACTER_LORA = "character_lora.safetensors"                   # loras: the creator's likeness LoRA
SKIN_LORA = "skin_detail_lora.safetensors"                      # loras: your Krea 2 skin enhancer
FACE_DETECTOR = "bbox/face_yolov8m.pt"                          # ultralytics/bbox

# ---- Settings ----------------------------------------------------------------------------------
AGE = 25            # set to the creator's real age: texture LoRAs read age words, and it keeps her looking adult
CHARACTER_STRENGTH = 0.85
SKIN_STRENGTH = 0.30
TEXTURE_MEGAPIXELS = 2.4
TEXTURE_DENOISE = 0.25
FACE_DENOISE = 0.30
STEPS = 12

TEXTURE_PROMPT = (
    f"this is an amateur photo taken from smartphone, casual photo. A {AGE}-year-old woman with natural skin "
    "texture: fine visible pores, subtle uneven skin tone, slight redness on knees, elbows and knuckles, matte skin "
    "without glossy highlights, natural light with real shadows."
)
FACE_PROMPT = (
    f"close-up smartphone photo of a {AGE}-year-old woman's face, natural matte skin with fine pores, light everyday "
    "makeup, natural eyelashes, faint under-eye shadows, slight redness around the nose, relaxed natural "
    "expression, natural light."
)

FINISH_IDS = {"bg_model": "40", "mask": "41", "depth_model": "42", "depth": "43", "depth_map": "44",
              "blur": "45", "lut": "50", "lens": "51", "sharpen": "52", "grain": "53", "save": "60"}


def sampler(model, positive, negative, latent, denoise, title):
    return node("KSampler", title, model=model, seed=0, steps=STEPS, cfg=1.0, sampler_name="euler",
                scheduler="simple", positive=positive, negative=negative, latent_image=latent, denoise=denoise)


def build():
    p = {}
    # 1. Input + models
    p["1"] = node("LoadImage", "Generated image", image="example.png")
    p["2"] = node("UNETLoader", "FinePorn v5 (Krea 2)", unet_name=FINEPORN_UNET, weight_dtype="default")
    p["3"] = node("LoraLoaderModelOnly", "Character LoRA", model=["2", 0], lora_name=CHARACTER_LORA,
                  strength_model=CHARACTER_STRENGTH)
    p["4"] = node("LoraLoaderModelOnly", "Skin detail LoRA (texture pass only)", model=["3", 0], lora_name=SKIN_LORA,
                  strength_model=SKIN_STRENGTH)
    p["5"] = node("CLIPLoader", "Krea 2 text encoder (Qwen3-VL 4B)", clip_name=KREA2_TE, type="krea2",
                  device="default")
    p["6"] = node("VAELoader", "VAE", vae_name=QWEN_VAE)

    # 2. Texture pass: more pixels + a light re-denoise adds real fine detail without changing the image
    p["10"] = node("ImageScaleToTotalPixels", f"Upscale to {TEXTURE_MEGAPIXELS} MP", image=["1", 0],
                   upscale_method="lanczos", megapixels=TEXTURE_MEGAPIXELS, resolution_steps=16)
    p["11"] = node("CLIPTextEncode", "Texture prompt (set her real age)", text=TEXTURE_PROMPT, clip=["5", 0])
    p["12"] = node("ConditioningZeroOut", "Negative (ignored at CFG 1)", conditioning=["11", 0])
    p["13"] = node("VAEEncode", "Encode", pixels=["10", 0], vae=["6", 0])
    p["14"] = sampler(["4", 0], ["11", 0], ["12", 0], ["13", 0], TEXTURE_DENOISE,
                      f"Texture pass (denoise {TEXTURE_DENOISE:.2f})")
    p["15"] = node("VAEDecode", "Decode", samples=["14", 0], vae=["6", 0])
    p["16"] = node("PreviewImage", "After texture pass", images=["15", 0])

    # 3. Face pass: the face gets a full 1024px canvas; no face found = this pass changes nothing
    p["20"] = node("UltralyticsDetectorProvider", "Face detector", model_name=FACE_DETECTOR)
    p["21"] = node("BboxDetectorSEGS", "Detect faces", bbox_detector=["20", 0], image=["15", 0], threshold=0.5,
                   dilation=0, crop_factor=3.0, drop_size=10, labels="all")
    p["22"] = node("ImpactSEGSOrderedFilter", "Keep largest face", segs=["21", 0], target="area(=w*h)", order=True,
                   take_start=0, take_count=1)
    p["23"] = node("SegsToCombinedMask", "Face mask", segs=["22", 0])
    p["24"] = node("InpaintCropImproved", "Crop face to 1024px", image=["15", 0], mask=["23", 0],
                   downscale_algorithm="bilinear", upscale_algorithm="bicubic", preresize=False,
                   preresize_mode="ensure minimum resolution", preresize_min_width=1024, preresize_min_height=1024,
                   preresize_max_width=16384, preresize_max_height=16384, mask_fill_holes=True,
                   mask_expand_pixels=24, mask_invert=False, mask_blend_pixels=32, mask_hipass_filter=0.1,
                   extend_for_outpainting=False, extend_up_factor=1.0, extend_down_factor=1.0,
                   extend_left_factor=1.0, extend_right_factor=1.0, context_from_mask_extend_factor=1.3,
                   output_resize_to_target_size=True, output_target_width=1024, output_target_height=1024,
                   output_padding="32", device_mode="gpu (much faster)")
    p["25"] = node("CLIPTextEncode", "Face prompt (set her real age)", text=FACE_PROMPT, clip=["5", 0])
    p["26"] = node("ConditioningZeroOut", "Negative (ignored at CFG 1)", conditioning=["25", 0])
    p["27"] = node("VAEEncode", "Encode face", pixels=["24", 1], vae=["6", 0])
    p["28"] = node("SetLatentNoiseMask", "Only touch the face", samples=["27", 0], mask=["24", 2])
    p["29"] = sampler(["3", 0], ["25", 0], ["26", 0], ["28", 0], FACE_DENOISE,
                      f"Face pass (denoise {FACE_DENOISE:.2f})")
    p["30"] = node("VAEDecode", "Decode face", samples=["29", 0], vae=["6", 0])
    p["31"] = node("InpaintStitchImproved", "Paste face back", stitcher=["24", 0], inpainted_image=["30", 0])
    # Gate: with no face found, skip the whole face pass (lazy branch) and use the texture-pass image as is.
    # Without it, the crop node falls back to the full image, pads it to a square and softens the edges.
    p["33"] = node("ImpactIsNotEmptySEGS", "Face found?", segs=["22", 0])
    p["34"] = node("ImpactConditionalBranch", "Face pass only if a face was found", cond=["33", 0],
                   tt_value=["31", 0], ff_value=["15", 0])

    # 4+. Camera finish
    postprocess.add_finish(p, ["34", 0], ids=FINISH_IDS, blur_enabled=False, filename_prefix="realism/final",
                           lut_strength=0.5, grain_amount=0.35, quality=90, subsampling="4:2:0 (smallest file)")
    return p


WIDTHS = {"1": 340, "2": 330, "3": 340, "4": 340, "5": 330, "6": 330, "10": 300, "11": 420, "14": 320, "16": 300,
          "20": 300, "21": 300, "22": 300, "24": 320, "25": 420, "29": 320, "34": 320,
          **postprocess.finish_widths(FINISH_IDS)}

SETUP_NOTE = f"""## Realism pass: texture → face → camera finish

Feed it a finished generation. It upscales to ~{TEXTURE_MEGAPIXELS} MP, re-renders fine skin detail, gives the face its own 1024px pass, then finishes like a phone camera.

**Before the first run**
- Pick your files in **Character LoRA** and **Skin detail LoRA**. No skin LoRA? Select that node and press Ctrl+B to bypass it.
- Set her **real age** in both prompts (currently {AGE}).

**Needs:** comfyui-postfx, Impact Pack + Subpack, Inpaint-CropAndStitch; FinePorn v5, Krea 2 text encoder + VAE, face_yolov8m.pt, plus BiRefNet + Depth Anything 3 (only run when Portrait blur is on).

**Output:** `output/realism/final_*.jpg`.
"""

TUNING_NOTE = """## Tuning

- **Skin still waxy** → *Skin detail LoRA* 0.4, or *Texture pass* denoise 0.30.
- **Skin looks older / freckles / skin tags** → *Skin detail LoRA* 0.15–0.2. Keep texture denoise ≤ 0.3: above that the LoRA starts adding moles and lines.
- **Face drifts from the creator** → *Face pass* denoise 0.2–0.25, or *Character LoRA* 0.95.
- **Face still airbrushed** → *Face pass* denoise 0.35.
- **Background has gibberish signs / odd people** → turn *Portrait blur* on.
- **Too grainy / not grainy enough** → *Grain* 0.25–0.45.
- **Seams around the face** → raise `mask_blend_pixels` on *Crop face to 1024px* to 48.
"""


def main():
    prompt = build()
    finish = postprocess.finish_groups(FINISH_IDS, first_number=4)
    rows = [
        [{"note": {"title": "Read me", "text": SETUP_NOTE, "width": 640, "height": 330}},
         {"title": "1 · Input + models", "color": "#3f789e", "columns": [["1"], ["2", "3", "4"], ["5", "6"]]},
         {"title": "2 · Texture pass", "color": "#b06634", "columns": [["10"], ["11", "12"], ["13", "14"], ["15", "16"]]}],
        [{"title": "3 · Face pass", "color": "#88aa88",
          "columns": [["20", "21", "22", "23"], ["24"], ["25", "26"], ["27", "28", "29"], ["30", "31"], ["33", "34"]]},
         finish[0]],
        [*finish[1:], {"note": {"title": "Tuning", "text": TUNING_NOTE, "width": 560, "height": 270}}],
    ]
    laid_out = {i for row in rows for block in row for col in block.get("columns", []) for i in col}
    if laid_out != set(prompt):
        raise SystemExit(f"layout/prompt mismatch: missing {sorted(set(prompt) - laid_out)}, "
                         f"extra {sorted(laid_out - set(prompt))}")
    (OUT_DIR / "realism_pass_api.json").write_text(json.dumps(prompt, indent=2) + "\n")
    (SPEC_DIR / "realism_pass.spec.json").write_text(
        json.dumps({"prompt": prompt, "widths": WIDTHS, "rows": rows}, indent=2) + "\n")
    print(f"wrote realism_pass_api.json + realism_pass.spec.json ({len(prompt)} nodes)")


if __name__ == "__main__":
    main()
