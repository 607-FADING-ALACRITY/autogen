"""PostFX: finishing nodes for ComfyUI.

Nodes (category "PostFX"):
  PostFX · Portrait Blur   depth/mask-driven lens blur with halo-free edges and highlight bloom
  PostFX · Apply LUT       .cube 3D/1D LUTs with a strength slider
  PostFX · Sharpen         luminance unsharp mask with a noise threshold
  PostFX · Film Grain      luminance-weighted, sized, optionally colored grain
  PostFX · Save JPEG       quality/subsampling control, sRGB profile, no workflow/prompt metadata,
                           IPTC "digital source type" AI disclosure

Install: copy this folder into ComfyUI/custom_nodes/ and restart ComfyUI.
Dependencies: torch, numpy, Pillow (all already required by ComfyUI).
LUTs: put .cube files in ComfyUI/models/luts/. The LUTs bundled in ./luts are listed too.
"""

import io
import math
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths

# ---------------------------------------------------------------------------------------------
# LUT folder registration: ComfyUI/models/luts plus the LUTs bundled with this pack
# ---------------------------------------------------------------------------------------------
BUNDLED_LUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "luts")
MODELS_LUT_DIR = os.path.join(folder_paths.models_dir, "luts")


def _register_lut_folder():
    paths, exts = folder_paths.folder_names_and_paths.get("luts", ([], set()))
    paths, exts = list(paths), set(exts) | {".cube"}
    for path in (MODELS_LUT_DIR, BUNDLED_LUT_DIR):
        if path not in paths:
            paths.append(path)
    folder_paths.folder_names_and_paths["luts"] = (paths, exts)


_register_lut_folder()


# ---------------------------------------------------------------------------------------------
# Shared helpers. Images are ComfyUI IMAGE tensors: (B, H, W, C) float in [0, 1].
# ---------------------------------------------------------------------------------------------
def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp(min=0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.pow(1.0 / 2.4) - 0.055)


def luminance(rgb):
    """Rec.709 luma of (..., 3) values."""
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def gaussian_blur(x, sigma):
    """Separable Gaussian blur of (N, C, H, W) with edge replication."""
    if sigma <= 0:
        return x
    radius = max(1, int(math.ceil(sigma * 3.0)))
    t = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(t * t) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    x = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1), groups=channels)
    x = F.pad(x, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(x, kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1), groups=channels)


def _split_alpha(image):
    """Returns (rgb, extra_channels_or_None) so RGBA inputs keep their alpha untouched."""
    if image.shape[-1] > 3:
        return image[..., :3], image[..., 3:]
    return image, None


def _join_alpha(rgb, extra):
    return rgb if extra is None else torch.cat([rgb, extra], dim=-1)


def _resize_plane(plane, height, width):
    """Bilinear-resize a (H, W) plane to (height, width) if needed."""
    if plane.shape[0] == height and plane.shape[1] == width:
        return plane
    return F.interpolate(plane[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]


# ---------------------------------------------------------------------------------------------
# Portrait blur
# ---------------------------------------------------------------------------------------------
def disc_kernel(radius, device, dtype):
    """Anti-aliased flat disc (the shape of a real lens's out-of-focus blur), normalized to sum 1."""
    n = int(math.ceil(radius))
    t = torch.arange(-n, n + 1, device=device, dtype=dtype)
    dist = torch.sqrt(t[None, :] ** 2 + t[:, None] ** 2)
    kernel = (radius + 0.5 - dist).clamp(0.0, 1.0)
    return kernel / kernel.sum()


def portrait_blur(image_hwc, coc, max_radius, highlight_bloom):
    """Lens blur of one image.

    image_hwc: (H, W, 3) sRGB. coc: (H, W) circle of confusion in [0, 1] (0 = sharp, 1 = max_radius).
    Blurs in linear light at several disc radii and blends per pixel by its coc. Each blurred layer
    is a normalized convolution weighted by coc, so in-focus pixels (the subject) never bleed into
    the background blur: no glow/halo around the subject's outline.
    """
    height, width, _ = image_hwc.shape
    if max_radius < 0.5 or float(coc.max()) <= 0.0:
        return image_hwc

    linear = srgb_to_linear(image_hwc)
    if highlight_bloom > 0:
        # Push small, near-clipped highlights (lamps, sun through leaves) above 1.0 so they spread into
        # visible bokeh discs. A white top-hat (luminance minus its morphological opening) keeps only
        # bright features smaller than the blur, so sky and other large bright areas, including their
        # edges, keep their brightness instead of blowing out.
        lum = luminance(linear)
        knee = ((lum - 0.75) / 0.25).clamp(0.0, 1.0)
        size = max(5, int(max_radius)) | 1
        plane = lum[None, None]
        opened = F.max_pool2d(-F.max_pool2d(-plane, size, stride=1, padding=size // 2), size, stride=1, padding=size // 2)
        isolation = ((plane - opened)[0, 0] / 0.15).clamp(0.0, 1.0)
        boost = 1.0 + highlight_bloom * 8.0 * knee * knee * (3.0 - 2.0 * knee) * isolation
        source = linear * boost[..., None]
    else:
        source = linear

    weight = coc
    planes = torch.cat([source * weight[..., None], weight[..., None]], dim=-1).permute(2, 0, 1)  # (4, H, W)

    levels = max(4, min(16, int(math.ceil(max_radius / 3.0))))
    pad = int(math.ceil(max_radius)) + 2
    padded = F.pad(planes[None], (pad, pad, pad, pad), mode="replicate")[0]
    padded_h, padded_w = padded.shape[-2:]
    spectrum = torch.fft.rfft2(padded)

    position = coc * levels                       # 0 = sharp layer, levels = max blur layer
    result = linear * (1.0 - position).clamp(0.0, 1.0)[..., None]
    for level in range(1, levels + 1):
        layer_weight = (1.0 - (position - level).abs()).clamp(0.0, 1.0)
        if float(layer_weight.max()) <= 0.0:
            continue
        kernel = disc_kernel(max_radius * level / levels, planes.device, planes.dtype)
        k = kernel.shape[0]
        kernel_full = torch.zeros((padded_h, padded_w), device=planes.device, dtype=planes.dtype)
        kernel_full[:k, :k] = kernel
        kernel_full = torch.roll(kernel_full, shifts=(-(k // 2), -(k // 2)), dims=(0, 1))
        blurred = torch.fft.irfft2(spectrum * torch.fft.rfft2(kernel_full), s=(padded_h, padded_w))
        blurred = blurred[:, pad:pad + height, pad:pad + width]
        numerator, denominator = blurred[:3].permute(1, 2, 0), blurred[3]
        layer = torch.where(denominator[..., None] > 1e-4,
                            numerator / denominator.clamp(min=1e-4)[..., None],
                            linear)
        result = result + layer * layer_weight[..., None]
    return linear_to_srgb(result).clamp(0.0, 1.0)


class PostFXPortraitBlur:
    """Phone-style portrait mode: subject stays sharp, background blur grows with distance."""

    CATEGORY = "PostFX"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True,
                                        "tooltip": "Off = pass the image through untouched and skip the depth/mask models."}),
                "blur_strength": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.05,
                                            "tooltip": "Maximum blur radius, as % of the image's long side."}),
                "falloff": ("FLOAT", {"default": 0.3, "min": 0.02, "max": 1.0, "step": 0.01,
                                      "tooltip": "Depth distance from the subject at which blur reaches maximum. "
                                                 "Lower = background goes fully soft right behind the subject."}),
                "highlight_bloom": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                                              "tooltip": "How much bright background lights bloom into bokeh discs."}),
                "edge_softness": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.5,
                                            "tooltip": "Feathers the subject mask edge, in pixels."}),
            },
            "optional": {
                "subject_mask": ("MASK", {"lazy": True,
                                          "tooltip": "Subject = 1 (kept sharp). From Remove Background."}),
                "depth": ("IMAGE", {"lazy": True,
                                    "tooltip": "Depth map, near = white. From Render Depth Anything 3."}),
            },
        }

    def check_lazy_status(self, enabled, **kwargs):
        # Unconnected optional inputs are absent; connected-but-not-yet-computed ones are None.
        if not enabled:
            return []
        return [name for name in ("subject_mask", "depth") if name in kwargs and kwargs[name] is None]

    def apply(self, image, enabled, blur_strength, falloff, highlight_bloom, edge_softness,
              subject_mask=None, depth=None):
        if not enabled or blur_strength <= 0:
            return (image,)
        if subject_mask is None and depth is None:
            raise ValueError("PostFX Portrait Blur needs a subject_mask, a depth map, or both. "
                             "Connect Remove Background and/or Render Depth Anything 3.")
        rgb, extra = _split_alpha(image)
        batch, height, width, _ = rgb.shape
        max_radius = blur_strength / 100.0 * max(height, width)
        out = []
        for i in range(batch):
            mask = None
            if subject_mask is not None:
                m = subject_mask[min(i, subject_mask.shape[0] - 1)].to(rgb.device, rgb.dtype)
                mask = _resize_plane(m, height, width).clamp(0.0, 1.0)
                if edge_softness > 0:
                    mask = gaussian_blur(mask[None, None], edge_softness)[0, 0]
            if depth is not None:
                d = depth[min(i, depth.shape[0] - 1), ..., 0].to(rgb.device, rgb.dtype)
                d = _resize_plane(d, height, width)
                if mask is not None and float((mask > 0.5).sum()) > 16:
                    focus = d[mask > 0.5].median()
                else:
                    focus = torch.quantile(d.flatten()[:: max(1, d.numel() // 200_000)], 0.9)
                coc = ((d - focus).abs() / falloff).clamp(0.0, 1.0)
                if mask is not None:
                    coc = coc * (1.0 - mask)
            else:
                coc = 1.0 - mask
            out.append(portrait_blur(rgb[i], coc, max_radius, highlight_bloom))
        return (_join_alpha(torch.stack(out), extra),)


# ---------------------------------------------------------------------------------------------
# LUT
# ---------------------------------------------------------------------------------------------
_LUT_CACHE = {}


def load_cube(path):
    """Parses an Adobe/Resolve .cube file. Returns dict(kind='3d'|'1d', size, table, domain_min, domain_max).

    3D tables are returned as (N, N, N, 3) indexed [b][g][r] (red varies fastest in the file).
    """
    key = (path, os.path.getmtime(path))
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]
    size_3d = size_1d = None
    domain_min, domain_max = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    rows = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            keyword = parts[0].upper()
            try:
                if keyword == "LUT_3D_SIZE":
                    size_3d = int(parts[1])
                elif keyword == "LUT_1D_SIZE":
                    size_1d = int(parts[1])
                elif keyword == "DOMAIN_MIN":
                    domain_min = [float(v) for v in parts[1:4]]
                elif keyword == "DOMAIN_MAX":
                    domain_max = [float(v) for v in parts[1:4]]
                elif keyword in ("LUT_3D_INPUT_RANGE", "LUT_1D_INPUT_RANGE"):
                    domain_min, domain_max = [float(parts[1])] * 3, [float(parts[2])] * 3
                elif keyword[0].isalpha():
                    continue  # TITLE and vendor keywords
                else:
                    rows.append([float(v) for v in parts[:3]])
            except (ValueError, IndexError) as e:
                raise ValueError(f"{os.path.basename(path)} line {line_no}: can't parse {raw.strip()!r} ({e})")
    if size_3d is None and size_1d is None:
        raise ValueError(f"{os.path.basename(path)}: no LUT_3D_SIZE or LUT_1D_SIZE line; not a .cube LUT")
    expected = size_3d ** 3 if size_3d else size_1d
    if len(rows) != expected:
        raise ValueError(f"{os.path.basename(path)}: expected {expected} rows for size "
                         f"{size_3d or size_1d}, found {len(rows)}")
    table = torch.tensor(rows, dtype=torch.float32)
    if size_3d:
        lut = {"kind": "3d", "size": size_3d, "table": table.view(size_3d, size_3d, size_3d, 3)}
    else:
        lut = {"kind": "1d", "size": size_1d, "table": table}
    lut["domain_min"] = torch.tensor(domain_min, dtype=torch.float32)
    lut["domain_max"] = torch.tensor(domain_max, dtype=torch.float32)
    _LUT_CACHE.clear()  # keep at most one LUT parsed at a time; they can be tens of MB
    _LUT_CACHE[key] = lut
    return lut


def apply_lut(rgb, lut):
    """rgb: (B, H, W, 3). Trilinear interpolation for 3D LUTs, linear for 1D."""
    device, dtype = rgb.device, rgb.dtype
    lo, hi = lut["domain_min"].to(device, dtype), lut["domain_max"].to(device, dtype)
    x = ((rgb - lo) / (hi - lo).clamp(min=1e-8)).clamp(0.0, 1.0)
    table = lut["table"].to(device, dtype)
    if lut["kind"] == "3d":
        volume = table.permute(3, 0, 1, 2)[None]                      # (1, 3, D=b, H=g, W=r)
        b, h, w, _ = x.shape
        grid = (x * 2.0 - 1.0).reshape(1, 1, b * h, w, 3)             # last dim = (r->W, g->H, b->D)
        out = F.grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return out[0, :, 0].permute(1, 2, 0).reshape(b, h, w, 3)
    n = lut["size"]
    pos = x * (n - 1)
    i0 = pos.floor().long().clamp(0, n - 1)
    i1 = (i0 + 1).clamp(max=n - 1)
    frac = pos - i0.to(dtype)
    channels = []
    for c in range(3):
        column = table[:, c]
        channels.append(column[i0[..., c]] * (1 - frac[..., c]) + column[i1[..., c]] * frac[..., c])
    return torch.stack(channels, dim=-1)


class PostFXApplyLUT:
    CATEGORY = "PostFX"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_name": (folder_paths.get_filename_list("luts"),
                             {"tooltip": "A .cube LUT from ComfyUI/models/luts. Use display-referred "
                                         "(Rec.709/sRGB) LUTs, not camera LOG conversion LUTs."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    def apply(self, image, lut_name, strength):
        if strength <= 0:
            return (image,)
        path = folder_paths.get_full_path("luts", lut_name)
        if path is None:
            raise FileNotFoundError(f"LUT {lut_name!r} not found in: {folder_paths.get_folder_paths('luts')}")
        rgb, extra = _split_alpha(image)
        graded = apply_lut(rgb, load_cube(path)).clamp(0.0, 1.0)
        return (_join_alpha(torch.lerp(rgb, graded, strength), extra),)


# ---------------------------------------------------------------------------------------------
# Sharpen
# ---------------------------------------------------------------------------------------------
class PostFXSharpen:
    """Unsharp mask on luminance only (no color fringing), with a threshold that leaves
    low-contrast texture (skin, grain, compression noise) alone."""

    CATEGORY = "PostFX"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0, "step": 0.05}),
                "radius": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 5.0, "step": 0.1,
                                     "tooltip": "Gaussian sigma in pixels. 0.7–1.2 for full-resolution output."}),
                "threshold": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005,
                                        "tooltip": "Edges weaker than this are not sharpened (0.02 ≈ 5/255)."}),
            }
        }

    def apply(self, image, amount, radius, threshold):
        if amount <= 0:
            return (image,)
        rgb, extra = _split_alpha(image)
        y = luminance(rgb)
        detail = y - gaussian_blur(y[:, None], radius)[:, 0]
        if threshold > 0:
            detail = detail * ((detail.abs() - threshold) / threshold).clamp(0.0, 1.0)
        out = (rgb + (amount * detail)[..., None]).clamp(0.0, 1.0)
        return (_join_alpha(out, extra),)


# ---------------------------------------------------------------------------------------------
# Film grain
# ---------------------------------------------------------------------------------------------
class PostFXFilmGrain:
    """Gaussian grain, clumped to `size` pixels, strongest in the midtones like real sensor/film noise."""

    CATEGORY = "PostFX"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                                     "tooltip": "1.0 = strong grain (std ≈ 15/255 in the midtones)."}),
                "size": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.1,
                                   "tooltip": "Grain clump size in pixels."}),
                "color": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "0 = monochrome grain, 1 = fully colored (sensor-like) noise."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True,
                                 "tooltip": "Randomized per run by default, so a batch doesn't share one grain pattern."}),
            }
        }

    def apply(self, image, amount, size, color, seed):
        if amount <= 0:
            return (image,)
        rgb, extra = _split_alpha(image)
        batch, height, width, _ = rgb.shape
        small_h, small_w = max(1, round(height / size)), max(1, round(width / size))
        out = []
        for i in range(batch):
            generator = torch.Generator(device="cpu").manual_seed(int(seed) + i)
            noise = torch.randn((1, 4, small_h, small_w), generator=generator, dtype=torch.float32)
            if (small_h, small_w) != (height, width):
                noise = F.interpolate(noise, size=(height, width), mode="bicubic", align_corners=False)
            noise = noise / noise.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            mono, chroma = noise[:, :1], noise[:, 1:]
            grain = (mono * (1.0 - color) + chroma * color) / math.sqrt((1.0 - color) ** 2 + color ** 2)
            grain = grain[0].permute(1, 2, 0).to(rgb.device, rgb.dtype)        # (H, W, 3)
            y = luminance(rgb[i])
            response = 0.25 + 0.75 * (4.0 * y * (1.0 - y))                     # 1.0 at mid-grey, 0.25 at black/white
            out.append((rgb[i] + amount * 0.06 * response[..., None] * grain).clamp(0.0, 1.0))
        return (_join_alpha(torch.stack(out), extra),)


# ---------------------------------------------------------------------------------------------
# JPEG export
# ---------------------------------------------------------------------------------------------
_DIGITAL_SOURCE_TYPES = {
    "AI-generated": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
    "Real photo edited with AI": "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia",
    "No AI used": None,
}
_SUBSAMPLING = {"4:4:4 (sharpest color)": 0, "4:2:2": 1, "4:2:0 (smallest file)": 2}


def _srgb_icc_profile():
    try:
        from PIL import ImageCms
        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:
        return None  # Pillow built without LittleCMS: browsers assume sRGB anyway


_SRGB_ICC = _srgb_icc_profile()


def xmp_packet(digital_source_type):
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '    xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"\n'
        f'    Iptc4xmpExt:DigitalSourceType="{digital_source_type}"/>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    ).encode("utf-8")


def insert_xmp(jpeg_bytes, xmp):
    """Inserts an XMP APP1 segment after SOI (and after the JFIF APP0 segment when present)."""
    if jpeg_bytes[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG stream")
    payload = b"http://ns.adobe.com/xap/1.0/\x00" + xmp
    if len(payload) + 2 > 0xFFFF:
        raise ValueError("XMP packet too large for one APP1 segment")
    segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    pos = 2
    if jpeg_bytes[2:4] == b"\xff\xe0":
        pos = 4 + struct.unpack(">H", jpeg_bytes[4:6])[0]
    return jpeg_bytes[:pos] + segment + jpeg_bytes[pos:]


def encode_jpeg(rgb_hwc, quality, subsampling, progressive, digital_source_type):
    array = (rgb_hwc.clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    buffer = io.BytesIO()
    options = {"quality": int(quality), "subsampling": subsampling, "progressive": bool(progressive),
               "optimize": True}
    if _SRGB_ICC:
        options["icc_profile"] = _SRGB_ICC
    Image.fromarray(array, "RGB").save(buffer, "JPEG", **options)
    data = buffer.getvalue()
    if digital_source_type:
        data = insert_xmp(data, xmp_packet(digital_source_type))
    return data


class PostFXSaveJPEG:
    """Saves JPEGs. Unlike Save Image it never embeds the ComfyUI prompt/workflow."""

    CATEGORY = "PostFX"
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "postfx/final"}),
                "quality": ("INT", {"default": 92, "min": 1, "max": 100,
                                    "tooltip": "90–95 is visually lossless. Above 95 mostly grows the file."}),
                "chroma_subsampling": (list(_SUBSAMPLING), {"default": "4:4:4 (sharpest color)"}),
                "progressive": ("BOOLEAN", {"default": True}),
                "ai_disclosure": (list(_DIGITAL_SOURCE_TYPES), {
                    "default": "AI-generated",
                    "tooltip": "Writes the IPTC DigitalSourceType tag that Meta, Google and C2PA tools read "
                               "to apply AI labels. 'No AI used' writes no tag."}),
            }
        }

    def save(self, images, filename_prefix, quality, chroma_subsampling, progressive, ai_disclosure):
        output_dir = folder_paths.get_output_directory()
        full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir, images[0].shape[1], images[0].shape[0])
        results = []
        for image in images:
            data = encode_jpeg(image[..., :3], quality, _SUBSAMPLING[chroma_subsampling], progressive,
                               _DIGITAL_SOURCE_TYPES[ai_disclosure])
            name = f"{filename}_{counter:05}_.jpg"
            with open(os.path.join(full_folder, name), "wb") as f:
                f.write(data)
            results.append({"filename": name, "subfolder": subfolder, "type": "output"})
            counter += 1
        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "PostFXPortraitBlur": PostFXPortraitBlur,
    "PostFXApplyLUT": PostFXApplyLUT,
    "PostFXSharpen": PostFXSharpen,
    "PostFXFilmGrain": PostFXFilmGrain,
    "PostFXSaveJPEG": PostFXSaveJPEG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PostFXPortraitBlur": "PostFX · Portrait Blur",
    "PostFXApplyLUT": "PostFX · Apply LUT",
    "PostFXSharpen": "PostFX · Sharpen",
    "PostFXFilmGrain": "PostFX · Film Grain",
    "PostFXSaveJPEG": "PostFX · Save JPEG",
}
