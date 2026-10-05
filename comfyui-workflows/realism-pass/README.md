# Realism pass

Takes a finished FinePorn/Krea 2 generation and makes it read like a real phone photo instead of an AI render:

```
generated image ─► upscale to ~2.4 MP ─► texture pass ─► face found? ─yes─► 1024px face pass ─┐
                                         (denoise 0.25,       │                                │
                                          character LoRA      └──no─────────────────────────────┤
                                          + skin LoRA 0.3)                                     ▼
            JPEG ◄─ grain ◄─ sharpen ◄─ lens ◄─ LUT ◄─ portrait blur (off by default) ◄────────┘
```

## Why each stage exists

| Stage | Fixes |
|---|---|
| **Texture pass** | Waxy, airbrushed skin. More pixels plus a light re-denoise adds real fine detail without changing the composition, and it's the only stage where the skin LoRA runs. |
| **Face pass** | The airbrushed face. In a full-body shot the face is rendered at ~200px, so the model has no room for pores. Here it gets a full 1024px canvas at denoise 0.30, with the character LoRA only. |
| **Lens** | "Too clean". Faint red/cyan fringing toward the corners and gentle corner falloff, which every phone lens leaves and AI renders don't. |
| **Grain** | Smooth skin next to razor-sharp eyes. Uniform sensor-style noise evens out the sharpness mismatch. |
| **JPEG 90, 4:2:0** | No compression. Phone photos are compressed; this matches it, with no prompt/workflow metadata and an AI-disclosure tag. |
| **Portrait blur** (off) | Gibberish signs and garbled bystanders. Turn it on for busy backgrounds and they're blurred away. |

**Why the skin enhancer aged her before, and doesn't here.** Skin LoRAs learn from extreme skin close-ups, many of them older skin with freckles and moles. Run at 0.8–1.0 during generation, they rewrite skin structure. Here the skin LoRA only runs at 0.3, and only at denoise 0.25 on an image that's already composed, which is enough to add pores but not moles, lines or skin tags. The face pass leaves it off entirely.

## Install

This uses the same pod setup as the other two workflows.

- **Custom nodes:** ComfyUI-Impact-Pack, ComfyUI-Impact-Subpack, ComfyUI-Inpaint-CropAndStitch (all from `../fineporn-faceswap`), and `comfyui-postfx` (from `../postprocess`).
- **Models** (folder under `ComfyUI/models/`)
  - `diffusion_models/` finepornV5INT8FP8_v5FP8.safetensors
  - `text_encoders/` qwen3vl_4b_fp8_scaled.safetensors
  - `vae/` qwen_image_vae.safetensors
  - `loras/` your character LoRA and your skin enhancer
  - `ultralytics/bbox/` face_yolov8m.pt
  - `background_removal/` birefnet.safetensors and `geometry_estimation/` depth_anything_3_mono_large.safetensors. They only *run* when portrait blur is on, but ComfyUI checks every dropdown before a run, so the files must exist.

Download links for all of these are in `../fineporn-faceswap/README.md` and `../postprocess/README.md`.

**Hardware.** Only FinePorn and its text encoder are loaded (~18 GB of weights), so it's much lighter than the face swap. A 24 GB GPU is comfortable; 16 GB should work with ComfyUI's offloading but is untested.

## Before the first run

1. **Pick your LoRAs** in *Character LoRA* and *Skin detail LoRA*. The defaults (`character_lora.safetensors`, `skin_detail_lora.safetensors`) are just names. No skin LoRA? Select that node and press **Ctrl+B** to bypass it.
2. **Set her real age** in both prompts (*Texture prompt* and *Face prompt*, default 25). The skin LoRA responds to age words, and an explicit adult age keeps generations from drifting young.

Then load an image in group 1 and queue. The result is `ComfyUI/output/realism/final_*.jpg`. It works on face-swap output too: feed it `output/faceswap/final_*.png`.

## Tuning

| Problem | Change |
|---|---|
| Skin still waxy | *Skin detail LoRA* 0.4, or *Texture pass* denoise 0.30 |
| Skin looks older, freckles, skin tags | *Skin detail LoRA* 0.15–0.2. Keep *Texture pass* denoise at 0.30 or below. |
| Face drifts away from the creator | *Face pass* denoise 0.20–0.25, or *Character LoRA* 0.95 |
| Face still airbrushed | *Face pass* denoise 0.35 |
| Seam around the face | `mask_blend_pixels` 48 on *Crop face to 1024px* |
| Background gives it away | *Portrait blur* `enabled` = true |
| Grain too strong or weak | *Grain* amount 0.25–0.45 |
| Want no grade | *LUT grade* strength 0 |

All the samplers use euler / simple / 12 steps / CFG 1, the FinePorn v5 author's settings. At denoise 0.25–0.30 that's 3–4 real steps per pass.

## Fix it at generation time too

The realism pass can't fix what's wrong with the composition. Changes to your generation prompts and LoRA:

- **Describe the light on her**, not just the scene. For example: "harsh sunlight from the skylight casting shadows across her face and chest". Otherwise she's lit flat while the scene has hard sun, and looks pasted in.
- **Drop beauty words**: "very beautiful face", "perfect", "flawless", "brushed eyelashes". They pull toward the beauty-filter face.
- **State her real age** and avoid "girl", "young", "teen" or "19yo"-style words. FinePorn's sample prompts use them, and they push faces young.
- **Avoid signage and crowds**, or plan to turn portrait blur on. Gibberish text is the fastest AI tell there is.
- **Character LoRA:** train it on unfiltered phone photos. A LoRA trained on beauty-filtered Instagram shots bakes the filter in. Run it at 0.8–0.9 rather than 1.0+.

## Automating (API format)

Post `{"prompt": <realism_pass_api.json>}` to `/prompt` after uploading the image. Inputs to patch:

| Node id | Inputs |
|---|---|
| `1` | `image` |
| `3`, `4` | `lora_name`, `strength_model` (character, skin) |
| `11`, `25` | `text`, the texture and face prompts (age) |
| `14`, `29` | `seed`, `denoise` (texture pass, face pass) |
| `45` | `enabled` (portrait blur) |
| `50` | `lut_name`, `strength` |
| `53` | `seed` (grain; use a new one per image) |
| `60` | `filename_prefix`, `quality`, `ai_disclosure` |

## What was tested, and what wasn't

**Validated and executed:**
- The full graph passes ComfyUI 0.38.0's own validation with zero errors.
- The face-pass plumbing (detect → crop → stitch → face-found gate) was executed on CPU, with the diffusion step replaced by a pass-through:
  - **With a face:** the head is cropped to 1024px and pasted back cleanly.
  - **Without a face:** the face-pass nodes never execute (checked via the execution trace), and the image comes out byte-identical. This gate exists because, without it, the crop node falls back to the full image, pads it to a square, and the padding bleeds into the top and bottom edges, while also running a wasted full sampling pass.
- The finishing chain, including the new lens node, ran end to end on real images. There are 23 unit tests in `../postprocess/tools/test_postfx.py`.

**Not executed:** the two diffusion passes, since there's no GPU here. The denoise values and LoRA strengths are starting points chosen from how low-denoise passes and texture LoRAs behave. Expect to tune them on your first batch with the table above.

## Rebuilding

```bash
cd tools
python build_workflow.py            # reuses the finishing chain from ../postprocess/tools/build_workflow.py
node ../../tools/to_ui.mjs realism_pass.spec.json ../realism_pass.json
```
