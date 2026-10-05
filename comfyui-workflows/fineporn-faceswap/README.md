# FinePorn v5 → head swap (ComfyUI)

This workflow generates an image with [FinePorn v5](https://civitai.red/models/2762538/fineporn-v5-int8fp8bf16?modelVersionId=3361846), a Krea 2 turbo merge. It then swaps in a creator's head from a reference photo and runs a light FinePorn pass over the new head so its skin texture matches the rest of the image.

The pipeline:

```
FinePorn v5 generation ──► detect largest face ──► head mask (grows with face size) ──► 1024² head crop
                                                                                            │
creator reference photo ──► detect face ──► 1024² head crop ────────────────────────────────┤
                                                                                            ▼
                         Qwen-Image-Edit-2511 + Lightning 4-step + BFS head-swap V5 (masked: only the head is repainted)
                                                                                            │
                         stitch back ──► FinePorn texture pass on the head only (denoise 0.20) ──► final
```

## Read this before you use it commercially

**1. FinePorn needs a Krea Enterprise License at your revenue.** FinePorn is a merge of Krea 2, so it is a "Derivative" under the Krea 2 Community License. Here is what the license says:
- **§2.3:** commercial use of the model, its Derivatives *or their Outputs* is only allowed if your company's total annual revenue is under $1M (trailing twelve months, all sources, affiliates included). Above that you need an Enterprise License from Krea *before* any commercial use. Contact: opensource@krea.ai.
- **§4.2:** you must run content filtering. Human review of every output before it is posted counts.
- **§4.3:** you must disclose AI-generated content wherever a platform requires it.
- **§9.1:** any breach terminates the license automatically, and §8 makes you indemnify Krea.

`faceswap_only.json` uses no Krea weights. Its components are Apache-2.0 or MIT, so it is not affected by this.

**2. Consent, per creator, in writing.** Only use a creator's likeness if all of the following are true:
- They have signed an AI-likeness release that explicitly covers AI-generated explicit content, names the platforms it will be posted on, and says what happens when they revoke it. On revocation you stop generating, take the content down and delete their reference set.
- That release sits in the same file as their platform KYC.
- The reference photos come from that creator's own vetted set.

Never use this on prospects, on leads' public photos, or on anyone who isn't under contract with you. In `faceswap_only`, only use target images you hold the rights to that don't show another real person: FinePorn generations, or the creator's own shoots.

**3. Disclose it.** Label posts as AI-generated in the way each platform requires. OnlyFans requires AI content to depict the verified creator and to be clearly tagged; Fanvue has its own AI label. Check the current terms of each platform before launch.

**4. The law.** Publishing sexual deepfakes without consent is a crime under federal law (TAKE IT DOWN Act, 2025) and Pennsylvania law (2024). The signed release is what keeps your content on the right side of both.

**5. The BFS LoRA's own terms.** BFS is MIT-licensed on Hugging Face. Its README says: *"Do not use or share results involving public figures or people who have not given consent."* A consenting creator fits that wording. The author's Civitai page is stricter: it says the LoRA is *"intended only for artistic and fictional characters"* and asks people not to share results involving real people.
- Download BFS from Hugging Face (link below).
- If the Civitai wording is a problem for you, set the *BFS head swap V5 LoRA* strength to `0`. Qwen-Image-Edit-2511 then does the swap on its own, with weaker likeness.

## Files

| File | What it is |
|---|---|
| `fineporn_faceswap.json` | Main workflow: generate → head swap → texture pass. Drag it into ComfyUI. |
| `faceswap_only.json` | Head swap onto an existing image. No Krea weights. |
| `fineporn_faceswap_api.json`, `faceswap_only_api.json` | The same graphs in API format, for queueing through ComfyUI's `/prompt` endpoint. |
| `tools/` | The builder that generates all four files. See [Rebuilding](#rebuilding). |

## Install

**ComfyUI:** version 0.26 or later (native Krea 2 support). Validated on 0.38.0 with frontend 1.53.10.

**Custom nodes.** Install via ComfyUI-Manager, or `git clone` into `ComfyUI/custom_nodes/` and `pip install -r requirements.txt` for each.

| Pack | Tested version | Used for |
|---|---|---|
| [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) | 8.28.3 | Face boxes, keeping the largest face |
| [ComfyUI-Impact-Subpack](https://github.com/ltdrdata/ComfyUI-Impact-Subpack) | 1.3.5 | YOLO face detector |
| [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) | 3.0.17 | Head crop and stitch |

**Models.** Folders are relative to `ComfyUI/models/`. Total download is about 50 GB.

| File | Folder | Size | Source | License |
|---|---|---|---|---|
| `finepornV5INT8FP8_v5FP8.safetensors` | `diffusion_models/` | 13.1 GB | [Civitai, V5 FP8](https://civitai.red/models/2762538/fineporn-v5-int8fp8bf16?modelVersionId=3361846) (requires login) | Krea 2 Community |
| `qwen3vl_4b_fp8_scaled.safetensors` | `text_encoders/` | 5.2 GB | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) | Krea 2 Community |
| `qwen_image_vae.safetensors` | `vae/` | 0.25 GB | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors) | Apache-2.0 |
| `qwen_image_edit_2511_fp8mixed.safetensors` | `diffusion_models/` | 20.5 GB | [Comfy-Org/Qwen-Image-Edit_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors) | Apache-2.0 |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `text_encoders/` | 9.4 GB | [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) | Apache-2.0 |
| `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | `loras/` | 0.85 GB | [lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors) | Apache-2.0 |
| `bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors` | `loras/` | 0.31 GB | [Alissonerdx/BFS-Best-Face-Swap](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors) | MIT, plus the author's terms above |
| `face_yolov8m.pt` | `ultralytics/bbox/` | 52 MB | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt) | Apache-2.0 |

Notes on the files:
- Krea 2 and Qwen-Image-Edit use the identical VAE file (same SHA-256), so it is loaded once.
- For FinePorn's INT8 or BF16 builds, pick that file in the *FinePorn v5 (Krea 2)* loader.

**Hardware.** Each run loads FinePorn, then Qwen-Image-Edit, then FinePorn again.
- Plan on a 24 GB GPU and 64 GB of system RAM, so ComfyUI keeps both models cached instead of reloading them from disk at every stage.
- Smaller GPUs rely on ComfyUI's offloading. That is untested here.

## Run

1. Drag `fineporn_faceswap.json` into ComfyUI. Any red node means a missing custom node pack or model file.
2. **Group 1:** write the prompt. Describe the creator's hair, skin tone and build. The swap replaces the head, so everything outside the head mask comes from FinePorn.
3. **Group 2:** load the creator's reference photo. Use a sharp, front-facing, evenly lit photo with nothing covering the face: no sunglasses, no hands.
4. Queue the run. Results land in `ComfyUI/output/faceswap/`:
   - `base_*`: the raw FinePorn generation
   - `swap_raw_*`: after the head swap
   - `final_*`: after the texture pass

   Compare `swap_raw` against `final` on your first batch to decide whether you want the texture pass.

For `faceswap_only.json`, load the target image in group 1 instead of writing a prompt. Output: `faceswap/swap_*`.

To finish either output with portrait blur, a LUT grade, grain and a metadata-clean JPEG, run it through [`../postprocess`](../postprocess/README.md).

## How it works

- **Detection.** `face_yolov8m` finds faces. If there are several people, only the largest face is swapped.
- **Head mask.** The face box is grown by 30% of its own size (*Head mask grow* node), so the mask covers hair, ears and jaw whether the shot is a close-up or full-body. The 30% figure is the expression `round(max(c - a, d - b) * 0.30)`.
- **Crop.** The head and 60% context on each side are cropped and upscaled to 1024². In a full-body shot that is 2–3× the face's original resolution in each direction, which is where most of the likeness comes from.
- **Swap.**
  - **Model and LoRAs:** Qwen-Image-Edit-2511 with the Lightning 4-step LoRA (4 steps, CFG 1, euler/simple, shift 3.1, per ComfyUI's official 2511 template) and the BFS V5 head-swap LoRA at 1.0.
  - **Inputs:** Picture 1 is the target crop. Picture 2 is the creator crop.
  - **Mask:** `SetLatentNoiseMask` limits repainting to the head mask, so the body stays pixel-for-pixel from FinePorn.
- **Texture pass.** The head is re-cropped and FinePorn re-runs it at denoise 0.20 (euler/simple, 12 steps, CFG 1). This puts back FinePorn's grainy smartphone texture, so the head doesn't read as pasted in.

## Tuning

| Problem | Fix |
|---|---|
| Doesn't look enough like the creator | Raise the *BFS head swap V5 LoRA* strength to 1.1–1.3. Use a better reference photo; it matters more than any setting. |
| Visible seam or "pasted" edge | Raise the `0.30` in *Head mask grow* to 0.4, or raise `mask_blend_pixels` on *Crop target head* to 48–64. |
| Hair outside the mask doesn't match | Describe the creator's real hair in the group 1 prompt. |
| Head looks smooth or plasticky | Raise the *Texture pass* denoise to 0.25–0.30. Above about 0.35 FinePorn starts redrawing the face and likeness drops. |
| Texture pass changes the face | Lower the denoise to 0.12, or use `swap_raw`. |
| No face found (nothing after the base image) | Lower the *Detect faces* threshold to 0.3. Strong profiles and turned-away faces may not be detected at all; regenerate instead. |
| Want max quality and can afford the time | Set the *Lightning 4-step LoRA* strength to 0, and the *Swap* sampler to 20 steps / CFG 2.5. |

## Automating (API format)

`*_api.json` can be posted to ComfyUI's `POST /prompt` as `{"prompt": <file contents>}`. Upload images with `POST /upload/image` first. Fields to patch per job:

| Node id | Input | What it is |
|---|---|---|
| `13` | `text` | Generation prompt (main workflow) |
| `16` | `seed` | Generation seed |
| `20` | `image` | Creator reference filename |
| `17` | `image` | Target image filename (`faceswap_only` only) |
| `42` | `strength_model` | BFS strength |
| `52` | `seed` | Swap seed |
| `65` | `denoise` | Texture-pass strength |

## What was tested, and what wasn't

**Verified:**
- Both graphs pass ComfyUI 0.38.0's own validation, which covers every node, input and link, with zero errors.
- The detect → head mask → crop → stitch chain was executed on a test image.
- The `.json` UI files were produced by the real ComfyUI frontend and convert back to exactly the API graphs.

**Not executed:** the three diffusion stages. The build machine has no GPU. Their settings come from:
- the FinePorn v5 author: euler / simple / 12 steps / CFG 1
- ComfyUI's official Qwen-Image-Edit-2511 template
- the BFS V5 documentation

Expect to tune BFS strength and texture-pass denoise on your first batch.

## Rebuilding

The workflows are generated, not hand-edited. Change settings in `tools/build_workflows.py`, then:

```bash
cd tools
python build_workflows.py                        # writes the *_api.json files and the layout specs
# with ComfyUI + the three node packs running on :8188, and `npm i playwright`:
node ../../tools/to_ui.mjs fineporn_faceswap.spec.json ../fineporn_faceswap.json
node ../../tools/to_ui.mjs faceswap_only.spec.json ../faceswap_only.json
```

`comfyui-workflows/tools/to_ui.mjs` builds the drag-and-drop file inside the real ComfyUI frontend. It refuses to write it unless the result converts back to an identical API graph.
