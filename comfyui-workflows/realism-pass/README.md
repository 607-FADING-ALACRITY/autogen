# Realism pass

Adds real skin texture to a finished generation **without changing her face**. Face shape, eyes, lashes, brows, lips, makeup, colors and lighting all stay exactly as generated; only fine skin detail is added.

```
generated image ─► upscale ~2.4 MP ─► texture pass (refiner, denoise 0.25) ─► Detail Transfer ─┐
                                                                                                │
        face found? ─yes─► 1024px face pass (denoise 0.30) ─► paste back ─► Detail Transfer ───┤
                     └no──────────────────────────────────────────────────────────────────────┤
                                                                                                ▼
                                                    lens ─► sharpen ─► grain ─► JPEG ◄─ (LUT, off)
```

## How it keeps her look

| Mechanism | What it does |
|---|---|
| **Detail Transfer** (frequency separation) | The refine passes are only allowed to contribute detail finer than ~3 px: pores, fine skin structure. Everything coarser (face shape, features, makeup, skin tone, shading, light) is taken from your original image, even if the refiner drifted. |
| **Feature protection** | Eyes with lashes and lids, brows, and lips are located with ComfyUI's native MediaPipe face landmarks and masked out of both passes. Measured on a test portrait: pixels inside the protected area change by at most 1/255. |
| **Low denoise** | 0.25 / 0.30 keeps the refined image aligned with the original, so the transferred texture lands in the right place. |
| **No face, no face pass** | When no face is found, the face pass never runs (verified via the execution trace). |

**Refiner:** Realism by Stable Yogi v3.0, a Krea 2 checkpoint trained on real photographs for skin, with your character LoRA at 0.85. It replaces the skin-enhancer LoRA, which aged her.

## Install

Only `comfyui-postfx` (from `../postprocess`) plus core ComfyUI 0.38+ nodes are needed. No Impact Pack, CropAndStitch, BiRefNet or Depth Anything.

| File | Folder under `models/` | Source |
|---|---|---|
| `realismByStableYogi_v30_fp8_scaled.safetensors` (13.2 GB) | `diffusion_models/` | [Civitai, v3.0 fp8_scaled](https://civitai.red/models/2786499/realism-by-stable-yogi-krea2). Download URL: `https://civitai.com/api/download/models/3329215?fileId=3218460` |
| `qwen3vl_4b_fp8_scaled.safetensors` | `text_encoders/` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) |
| `qwen_image_vae.safetensors` | `vae/` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors) |
| `mediapipe_face_fp32.safetensors` (5 MB) | `detection/` | [Comfy-Org/mediapipe](https://huggingface.co/Comfy-Org/mediapipe/resolve/main/detection/mediapipe_face_fp32.safetensors), Apache-2.0 |
| your character LoRA | `loras/` | |

Licensing:
- The Stable Yogi license allows selling generated images. It requires crediting the creator, and does not allow redistributing or merging the model.
- It is a Krea 2 derivative, so the Krea 2 Community License revenue cap applies, the same as FinePorn.

## Before the first run

1. Pick your file in **Character LoRA**.
2. Set her **real age** in both prompts (default 25).

Load an image in group 1 and queue it. The result goes to `ComfyUI/output/realism/final_*.jpg`. It works on output from the face swap and snapshot workflows too.

## Tuning

| Problem | Change |
|---|---|
| Skin still too smooth | *Detail Transfer* radius 4–5 on both, or *Texture pass* denoise 0.30 |
| Texture too strong, or looks aged | *Detail Transfer* radius 2, or strength 0.7 |
| Ghost edges or doubled hair strands | Lower that pass's denoise to 0.20; the refine drifted out of alignment |
| Brow or lash edge still changed | *Protect* grow 0.18 |
| Want texture on the lips | *Protect* lips = false |
| Nude areas look smoothed | The refiner is SFW-trained. Swap *Refiner* to FinePorn v5 for those images. |
| A face is missed, e.g. strong profile | Lower *Find faces* min_confidence to 0.55. At 0.5 it falsely detected a sunflower in testing. |
| Want a grade | *LUT grade* strength 0.3–0.5. It's off by default so the colors stay yours. |

**Detail Transfer radius guide:**

| Radius | What comes from the refiner |
|---|---|
| 1.5–2 px | Pores only |
| 3 px (default) | Pores and fine skin structure |
| 6+ px | Starts moving shading and makeup |

## Automating (API format)

Post `{"prompt": <realism_pass_api.json>}` to `/prompt` after uploading the image.

| Node id | Inputs |
|---|---|
| `1` | `image` |
| `2` | `unet_name` (refiner) |
| `3` | `lora_name`, `strength_model` |
| `11`, `21` | `text` (texture and face prompts) |
| `14`, `24` | `seed`, `denoise` |
| `19`, `27` | `radius`, `strength` (Detail Transfer) |
| `53` | `seed` (grain) |
| `60` | `filename_prefix`, `quality`, `ai_disclosure` |

## What was tested, and what wasn't

**Validated and executed:**
- The graph passes ComfyUI 0.38.0's validation with zero errors.
- On CPU, with the real MediaPipe model, I swapped the two diffusion passes for "add noise" stand-ins and checked where the noise ended up:
  - **Portrait:** texture landed on skin. The eyes, lashes, brows and lips changed by at most 1/255.
  - **No-face image:** the face pass never executed.
- The node pack's 28 unit tests pass. They cover Detail Transfer, the feature mask, and face crop/paste (bounds, round trip, no-face skip).

**Not executed:** the two diffusion passes, since there's no GPU here. Denoise 0.25/0.30 and radius 3 are starting points.

## Rebuilding

```bash
cd tools
python build_workflow.py
node ../../tools/to_ui.mjs realism_pass.spec.json ../realism_pass.json
```
