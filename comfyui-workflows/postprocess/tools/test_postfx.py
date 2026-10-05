"""Unit tests for the PostFX node pack.

Run from a Python environment that has ComfyUI's requirements installed:
    COMFYUI_PATH=/path/to/ComfyUI python -m pytest test_postfx.py -q
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

COMFYUI_PATH = os.environ.get("COMFYUI_PATH")
if not COMFYUI_PATH:
    pytest.skip("set COMFYUI_PATH to a ComfyUI checkout", allow_module_level=True)
sys.path.insert(0, COMFYUI_PATH)

PACK_DIR = Path(__file__).resolve().parent.parent / "comfyui-postfx"
spec = importlib.util.spec_from_file_location("postfx", PACK_DIR / "__init__.py")
postfx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(postfx)


def gradient_image(h=64, w=96):
    y = torch.linspace(0, 1, h)[:, None].expand(h, w)
    x = torch.linspace(0, 1, w)[None, :].expand(h, w)
    return torch.stack([x, y, (x + y) / 2], dim=-1)[None]          # (1, H, W, 3)


def write_cube(path, size, fn, header=""):
    steps = torch.linspace(0, 1, size)
    rows = []
    for b in steps:
        for g in steps:
            for r in steps:                                          # red fastest
                rows.append(fn(torch.tensor([r, g, b])))
    body = "\n".join(f"{v[0]:.7f} {v[1]:.7f} {v[2]:.7f}" for v in rows)
    path.write_text(f'TITLE "t"\n{header}LUT_3D_SIZE {size}\n# comment\n{body}\n')


# ---- LUT ---------------------------------------------------------------------------------------
def test_identity_3d_lut_is_lossless(tmp_path):
    path = tmp_path / "identity.cube"
    write_cube(path, 17, lambda v: v)
    img = gradient_image()
    out = postfx.apply_lut(img[..., :3], postfx.load_cube(str(path)))
    assert torch.allclose(out, img, atol=1e-5)


def test_3d_lut_channel_order(tmp_path):
    path = tmp_path / "swap.cube"
    write_cube(path, 9, lambda v: torch.stack([v[2], v[1], v[0]]))   # swap red and blue
    img = torch.tensor([[[[0.9, 0.2, 0.1]]]])
    out = postfx.apply_lut(img, postfx.load_cube(str(path)))
    assert torch.allclose(out, torch.tensor([[[[0.1, 0.2, 0.9]]]]), atol=1e-5)


def test_1d_lut_and_domain(tmp_path):
    path = tmp_path / "invert.cube"
    path.write_text("LUT_1D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 2 2 2\n1 1 1\n0 0 0\n")
    img = torch.tensor([[[[0.0, 1.0, 2.0]]]])
    out = postfx.apply_lut(img, postfx.load_cube(str(path)))
    assert torch.allclose(out, torch.tensor([[[[1.0, 0.5, 0.0]]]]), atol=1e-6)


def test_bad_cube_reports_line(tmp_path):
    path = tmp_path / "bad.cube"
    path.write_text("LUT_3D_SIZE 2\n0 0 0\n1 x 0\n")
    with pytest.raises(ValueError, match="line 3"):
        postfx.load_cube(str(path))


def test_bundled_luts_parse_and_stay_in_range():
    for name in ("warm_film.cube", "bright_clean.cube"):
        lut = postfx.load_cube(str(PACK_DIR / "luts" / name))
        assert lut["kind"] == "3d" and lut["size"] == 33
        assert float(lut["table"].min()) >= 0.0 and float(lut["table"].max()) <= 1.0


def test_lut_node_strength_zero_is_identity():
    img = gradient_image()
    (out,) = postfx.PostFXApplyLUT().apply(img, "warm_film.cube", 0.0)
    assert out is img


# ---- Lens --------------------------------------------------------------------------------------
def test_lens_zero_is_identity():
    img = gradient_image()
    assert postfx.PostFXLens().apply(img, 0.0, 0.0)[0] is img


def test_vignette_darkens_corners_only():
    img = torch.full((1, 101, 151, 3), 0.6)
    (out,) = postfx.PostFXLens().apply(img, 0.0, 1.0)
    assert abs(float(out[0, 50, 75, 0]) - 0.6) < 1e-4                  # center untouched
    assert float(out[0, 0, 0, 0]) < 0.6 * 0.75                           # corner clearly darker
    assert float(out[0, 50, 0, 0]) > float(out[0, 0, 0, 0])             # edge midpoint between the two


def test_chromatic_aberration_fringes_grow_toward_corners():
    img = torch.zeros(1, 200, 200, 3)
    img[:, 20:30, 20:30] = 1.0                                           # white square near the top-left corner
    img[:, 95:105, 95:105] = 1.0                                         # and one at the center
    (out,) = postfx.PostFXLens().apply(img, 6.0, 0.0)
    corner, center = out[0, 10:40, 10:40], out[0, 85:115, 85:115]
    assert float((corner[..., 0] - corner[..., 2]).abs().sum()) > 1.0     # red/blue split at the corner
    assert float((center[..., 0] - center[..., 2]).abs().sum()) < 0.1 * float((corner[..., 0] - corner[..., 2]).abs().sum())
    assert torch.allclose(out[..., 1], img[..., 1])                      # green is the reference channel
    # red is magnified (pushed outward, toward the top-left), blue pulled inward
    red_cols = corner[..., 0].sum(0).nonzero().flatten()
    blue_cols = corner[..., 2].sum(0).nonzero().flatten()
    assert int(red_cols.min()) < int(blue_cols.min())


# ---- Detail transfer ---------------------------------------------------------------------------
def test_detail_transfer_keeps_coarse_takes_fine():
    torch.manual_seed(0)
    base = gradient_image(96, 96)
    noise = torch.randn(1, 96, 96, 3) * 0.05
    node = postfx.PostFXDetailTransfer()
    (same,) = node.apply(base, base.clone(), 3.0, 1.0)
    assert torch.allclose(same, base, atol=1e-6)
    (shifted,) = node.apply(base, (base + 0.2).clamp(0, 1), 3.0, 1.0)      # brighter overall = coarse change
    assert float((shifted - base)[:, 8:-8, 8:-8].abs().max()) < 0.02
    (textured,) = node.apply(base, base + noise, 3.0, 1.0)                  # fine texture = transferred
    expected = postfx.highpass(noise.permute(0, 3, 1, 2), 3.0).permute(0, 2, 3, 1)
    got = textured - base
    corr = float((got * expected).sum() / (got.norm() * expected.norm()))
    assert corr > 0.95


def test_detail_transfer_protect_mask_keeps_base_exactly():
    torch.manual_seed(0)
    base = gradient_image(64, 64)
    detail = (base + torch.randn(1, 64, 64, 3) * 0.05).clamp(0, 1)
    protect = torch.zeros(1, 64, 64)
    protect[:, :, :32] = 1.0
    (out,) = postfx.PostFXDetailTransfer().apply(base, detail, 2.0, 1.0, protect_mask=protect)
    assert torch.equal(out[:, :, :32], base[:, :, :32])
    assert not torch.allclose(out[:, :, 40:], base[:, :, 40:])


# ---- Face tools (synthetic landmarks with the same structure as MediaPipe's) ---------------------
def ring(cx, cy, rx, ry, n, start):
    t = torch.linspace(0, 2 * torch.pi, n + 1)[:-1]
    pts = torch.stack([cx + rx * torch.cos(t), cy + ry * torch.sin(t)], dim=-1).numpy()
    return pts, [(start + k, start + (k + 1) % n) for k in range(n)]


def fake_landmarks(height=200, width=200, cx=100, cy=100, scale=1.0, faces=1):
    parts = {"face_oval": (0, 0, 60, 75, 36), "left_eye": (-25, -15, 12, 5, 16), "right_eye": (25, -15, 12, 5, 16),
             "left_eyebrow": (-25, -30, 14, 3, 8), "right_eyebrow": (25, -30, 14, 3, 8), "lips": (0, 35, 18, 7, 20)}
    pts, sets, start = [], {}, 0
    for name, (dx, dy, rx, ry, n) in parts.items():
        p, edges = ring(cx + dx * scale, cy + dy * scale, rx * scale, ry * scale, n, start)
        pts.append(p); sets[name] = frozenset(edges); start += n
    xy = np.concatenate(pts).astype(np.float32)
    return {"frames": [[{"landmarks_xy": xy} for _ in range(faces)]], "image_size": (height, width),
            "connection_sets": sets}



def test_feature_mask_covers_eyes_brows_lips_not_cheeks():
    (mask,) = postfx.PostFXFaceFeatureMask().apply(fake_landmarks(), True, True, 0.12, 0.0)
    assert mask.shape == (1, 200, 200)
    for x, y in ((75, 85), (125, 85), (75, 70), (100, 135)):              # eye, eye, brow, lips
        assert float(mask[0, y, x]) == 1.0
    assert float(mask[0, 110, 60]) == 0.0                                   # cheek stays editable
    (none,) = postfx.PostFXFaceFeatureMask().apply(fake_landmarks(faces=0), True, True, 0.12, 4.0)
    assert float(none.max()) == 0.0


def test_face_crop_paste_roundtrip_and_bounds():
    torch.manual_seed(0)
    img = torch.rand(1, 200, 300, 3)
    lm = fake_landmarks(height=200, width=300, cx=40, cy=60)               # face near the top-left corner
    crop, info = postfx.PostFXFaceCrop().apply(img, lm, 1.6, 256, 0.06)
    assert crop.shape == (1, 256, 256, 3)
    i = info[0]
    assert i["found"] and i["left"] >= 0 and i["top"] >= 0
    assert i["left"] + i["side"] <= 300 and i["top"] + i["side"] <= 200    # never pads past the image
    paste = postfx.PostFXFacePaste()
    assert paste.check_lazy_status(img, info, None) == ["face"]
    (out,) = paste.apply(img, info, face=torch.zeros(1, 256, 256, 3))
    inside = i["mask"] > 0.99
    y, x = inside.nonzero()[0].tolist()
    assert float(out[0, i["top"] + y, i["left"] + x].max()) < 0.01        # face area replaced
    assert torch.equal(out[0, :, 250:], img[0, :, 250:])                  # far from the face untouched


def test_face_crop_with_no_face_skips_the_face_pass():
    img = torch.rand(1, 120, 160, 3)
    crop, info = postfx.PostFXFaceCrop().apply(img, fake_landmarks(120, 160, faces=0), 1.6, 256, 0.06)
    paste = postfx.PostFXFacePaste()
    assert info == [{"found": False}]
    assert paste.check_lazy_status(img, info, None) == []                 # face pass never requested
    assert paste.apply(img, info, None)[0] is img


# ---- Sharpen -----------------------------------------------------------------------------------
def test_sharpen_leaves_flat_areas_alone_and_boosts_edges():
    img = torch.full((1, 32, 32, 3), 0.5)
    img[:, :, 16:] = 0.7
    (out,) = postfx.PostFXSharpen().apply(img, amount=1.0, radius=1.0, threshold=0.0)
    assert torch.allclose(out[:, :, :8], img[:, :, :8], atol=1e-6)      # flat region untouched
    assert float(out[0, 0, 15, 0]) < 0.5 and float(out[0, 0, 16, 0]) > 0.7   # overshoot either side


def test_sharpen_threshold_ignores_small_detail():
    img = torch.full((1, 32, 32, 3), 0.5)
    img[:, ::2, ::2] += 0.004                                            # ~1/255 texture
    (out,) = postfx.PostFXSharpen().apply(img, amount=2.0, radius=1.0, threshold=0.02)
    assert torch.allclose(out, img, atol=1e-6)


# ---- Grain -------------------------------------------------------------------------------------
def test_grain_is_deterministic_per_seed_and_calibrated():
    img = torch.full((1, 256, 256, 3), 0.5)
    node = postfx.PostFXFilmGrain()
    (a,) = node.apply(img, amount=1.0, size=1.0, color=0.0, seed=7)
    (b,) = node.apply(img, amount=1.0, size=1.0, color=0.0, seed=7)
    (c,) = node.apply(img, amount=1.0, size=1.0, color=0.0, seed=8)
    assert torch.equal(a, b) and not torch.equal(a, c)
    delta = a - img
    assert abs(float(delta.mean())) < 0.003
    assert 0.055 < float(delta.std()) < 0.065                           # amount 1.0 -> std 0.06 at mid-grey
    assert torch.allclose(delta[..., 0], delta[..., 2])                  # color=0 -> monochrome


def test_grain_weaker_in_shadows():
    node = postfx.PostFXFilmGrain()
    dark = torch.full((1, 128, 128, 3), 0.03)
    mid = torch.full((1, 128, 128, 3), 0.5)
    (d,) = node.apply(dark, 1.0, 1.0, 0.0, 1)
    (m,) = node.apply(mid, 1.0, 1.0, 0.0, 1)
    assert float((d - dark).std()) < 0.5 * float((m - mid).std())


# ---- Portrait blur -----------------------------------------------------------------------------
def test_portrait_disabled_passes_through_and_requests_nothing():
    node = postfx.PostFXPortraitBlur()
    img = gradient_image()
    assert node.check_lazy_status(enabled=False, subject_mask=None, depth=None) == []
    assert node.apply(img, False, 1.5, 0.3, 0.3, 1.5)[0] is img


def test_portrait_lazy_requests_only_connected_inputs():
    node = postfx.PostFXPortraitBlur()
    assert sorted(node.check_lazy_status(enabled=True, subject_mask=None, depth=None)) == ["depth", "subject_mask"]
    assert node.check_lazy_status(enabled=True, subject_mask=None) == ["subject_mask"]   # depth not connected
    assert node.check_lazy_status(enabled=True, subject_mask=torch.zeros(1, 4, 4)) == []


def test_portrait_keeps_subject_pixels_and_blurs_background():
    torch.manual_seed(0)
    img = torch.rand(1, 120, 160, 3)
    mask = torch.zeros(1, 120, 160)
    mask[:, :, :60] = 1.0                                                # left side = subject
    (out,) = postfx.PostFXPortraitBlur().apply(img, True, 4.0, 0.3, 0.0, 0.0, subject_mask=mask)
    assert torch.allclose(out[:, :, :55], img[:, :, :55], atol=1e-5)    # subject untouched
    assert float(out[:, :, 80:].std()) < 0.5 * float(img[:, :, 80:].std())   # background smoothed


def test_portrait_no_halo_from_bright_subject():
    img = torch.full((1, 100, 100, 3), 0.2)
    img[:, :, :50] = 1.0                                                  # bright subject on the left
    mask = torch.zeros(1, 100, 100)
    mask[:, :, :50] = 1.0
    (out,) = postfx.PostFXPortraitBlur().apply(img, True, 5.0, 0.3, 0.0, 0.0, subject_mask=mask)
    assert float(out[0, :, 52:, 0].max()) < 0.21                         # no glow bleeding into background


def test_portrait_depth_drives_blur_amount():
    torch.manual_seed(0)
    img = torch.rand(1, 100, 200, 3)
    depth = torch.zeros(1, 100, 200, 3)
    depth[:, :, :100] = 1.0                                               # near half
    depth[:, :, 100:] = 0.0                                               # far half
    (out,) = postfx.PostFXPortraitBlur().apply(img, True, 3.0, 0.5, 0.0, 0.0, depth=depth)
    assert torch.allclose(out[:, :, :95], img[:, :, :95], atol=1e-5)     # focus lands on the nearest region
    assert float(out[:, :, 110:].std()) < 0.5 * float(img[:, :, 110:].std())


def test_bloom_spares_large_bright_areas_but_blooms_point_lights():
    node = postfx.PostFXPortraitBlur()
    background = torch.zeros(1, 120, 120)                                   # everything is background
    sky = torch.full((1, 120, 120, 3), 0.85)
    (out,) = node.apply(sky, True, 5.0, 0.3, 1.0, 0.0, subject_mask=background)
    assert torch.allclose(out, sky, atol=0.01)                              # flat bright sky is not blown out

    skyline = torch.full((1, 120, 120, 3), 0.1)                            # dark building under bright sky
    skyline[:, :60] = 0.9
    (plain,) = node.apply(skyline, True, 5.0, 0.3, 0.0, 0.0, subject_mask=background)
    (bloomed,) = node.apply(skyline, True, 5.0, 0.3, 1.0, 0.0, subject_mask=background)
    assert float((bloomed - plain).abs().max()) < 0.02                     # no bright rim along the skyline

    lamp = torch.full((1, 120, 120, 3), 0.1)
    lamp[:, 58:62, 58:62] = 1.0
    (no_bloom,) = node.apply(lamp, True, 5.0, 0.3, 0.0, 0.0, subject_mask=background)
    (bloom,) = node.apply(lamp, True, 5.0, 0.3, 1.0, 0.0, subject_mask=background)
    assert float(bloom[0, 60, 54, 0]) > float(no_bloom[0, 60, 54, 0]) + 0.1   # lamp spreads into a bright disc


def test_portrait_requires_an_input():
    with pytest.raises(ValueError, match="subject_mask"):
        postfx.PostFXPortraitBlur().apply(gradient_image(), True, 1.5, 0.3, 0.3, 1.5)


# ---- JPEG --------------------------------------------------------------------------------------
def test_jpeg_has_ai_tag_icc_and_no_workflow(tmp_path):
    data = postfx.encode_jpeg(gradient_image()[0], 92, 0, True, postfx._DIGITAL_SOURCE_TYPES["AI-generated"])
    path = tmp_path / "x.jpg"
    path.write_bytes(data)
    im = Image.open(path)
    im.load()
    assert im.format == "JPEG" and im.size == (96, 64)
    assert "trainedAlgorithmicMedia" in str(im.getxmp())
    assert im.info.get("icc_profile")
    assert b"workflow" not in data and b"prompt" not in data


def test_jpeg_no_ai_used_writes_no_xmp():
    data = postfx.encode_jpeg(gradient_image()[0], 92, 2, False, None)
    assert b"ns.adobe.com/xap" not in data


def test_save_node_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr(postfx.folder_paths, "get_output_directory", lambda: str(tmp_path))
    imgs = torch.cat([gradient_image(), gradient_image()])
    result = postfx.PostFXSaveJPEG().save(imgs, "post/test", 90, "4:2:0 (smallest file)", True, "AI-generated")
    files = sorted(p.name for p in (tmp_path / "post").iterdir())
    assert files == ["test_00001_.jpg", "test_00002_.jpg"]
    assert [r["filename"] for r in result["ui"]["images"]] == files
