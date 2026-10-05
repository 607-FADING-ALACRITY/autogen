# Snapshot

A generation workflow for the casual phone-snapshot / analog look. It's built from the recipe behind [a top-rated Krea 2 realism image](https://civitai.red/images/144541104), with your creator's LoRA in place of the original character.

## Where that image's realism comes from

| Ingredient | Setting |
|---|---|
| **Realistic Snapshot LoRA** (Krea 2 v1.0) | strength **1.5** (its author's recommendation; 1.0 does little) |
| Sampler | euler / simple, **10 steps, CFG 1** |
| Resolution | **768×1376** (9:16, ~1 MP). Lower resolution keeps the phone softness. |
| Prompt | A **plain background** and one simple action, plus the line *"The photo is an analog shot in the style of the '80s, with slight film grain and muted colors."* |

**What's deliberately different from that image:**
- **Checkpoint.** It used "Dark Beast 5C", whose model card says it is tuned toward a younger character age range. That's the wrong direction for explicit content: it pushes faces young, which gets accounts banned regardless of the stated age. This workflow uses FinePorn v5.
- **Age wording.** Its prompt used "19 years old, petite, cute". Here the prompt states her real age, and none of the realism depends on youth-coded words.
- **Refusal-reduction LoRA.** It also used a TextFusion refusal-reduction LoRA. FinePorn already generates NSFW, so it isn't needed.

## Install

Needs `comfyui-postfx` (for Save JPEG) plus core ComfyUI.

| File | Folder under `models/` | Source |
|---|---|---|
| `finepornV5INT8FP8_v5FP8.safetensors` | `diffusion_models/` | already in your pod |
| `qwen3vl_4b_fp8_scaled.safetensors`, `qwen_image_vae.safetensors` | `text_encoders/`, `vae/` | already in your pod |
| `RealisticSnapshotKrea2V2.safetensors` (1.5 GB) | `loras/` | [Civitai](https://civitai.red/models/2268008). Download URL: `https://civitai.com/api/download/models/3371723?fileId=3260432` |
| your character LoRA | `loras/` | |

The Realistic Snapshot license allows commercial use with no credit required. It was in paid early access until 2026-10-05 18:01 UTC.

## Use

1. Pick your **Character LoRA** and put its trigger word at the start of the prompt.
2. Set her real age in the prompt.
3. Keep scenes simple: a plain wall, a bedroom, a bathroom mirror. Describe one light source. Signs, crowds and busy backgrounds undo the effect.
4. Queue it. The output goes to `ComfyUI/output/snapshot/gen_*.jpg`. Run it through `../realism-pass` if the skin needs more texture.

## Tuning

| Problem | Change |
|---|---|
| Look too strong, or faces drift | Realistic Snapshot LoRA 1.0–1.2 |
| Likeness weak | Character LoRA 0.95, or Realistic Snapshot down to 1.2 |
| Want more detail | 1040×1520 (loses some phone softness) |
| SFW promo content | Base model → Realism by Stable Yogi |

The graph passes ComfyUI 0.38.0's validation. Generation itself wasn't run here (no GPU).
