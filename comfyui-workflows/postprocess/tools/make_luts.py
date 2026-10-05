"""Generates the starter LUTs bundled in ../comfyui-postfx/luts/.

  warm_film.cube     soft S-curve, lifted blacks, warm highlights / slightly cool shadows, -8% saturation
  bright_clean.cube  +1/3 stop, gentle highlight roll-off, low contrast, faint warmth, +4% saturation

Both are display-referred (sRGB in, sRGB out), 33x33x33, the size Resolve and Photoshop export.
Run: python make_luts.py
"""

from pathlib import Path

import numpy as np

SIZE = 33
OUT_DIR = Path(__file__).resolve().parent.parent / "comfyui-postfx" / "luts"


def luma(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def saturate(rgb, amount):
    y = luma(rgb)[..., None]
    return y + (rgb - y) * amount


def warm_film(rgb):
    s_curve = rgb * rgb * (3.0 - 2.0 * rgb)
    out = rgb + (s_curve - rgb) * 0.30
    out = 0.035 + out * (0.975 - 0.035)                     # lifted blacks, softened whites
    y = luma(out)[..., None]
    highlights, shadows = y ** 1.5, (1.0 - y) ** 2
    out = out + highlights * np.array([0.035, 0.008, -0.045]) + shadows * np.array([-0.012, 0.004, 0.018])
    return saturate(out, 0.92)


def bright_clean(rgb):
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    linear = linear * 2 ** (1 / 3)
    linear = linear / (1.0 + linear * 0.18)                  # highlight roll-off so +1/3 stop doesn't clip
    linear = linear / (2 ** (1 / 3) / (1.0 + 2 ** (1 / 3) * 0.18))   # keep pure white at 1.0
    out = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    out = 0.5 + (out - 0.5) * 0.94                           # slightly lower contrast
    out = out * np.array([1.012, 1.0, 0.985])
    return saturate(out, 1.04)


def write_cube(path, title, grade):
    steps = np.linspace(0.0, 1.0, SIZE)
    b, g, r = np.meshgrid(steps, steps, steps, indexing="ij")    # red varies fastest when flattened
    rgb = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    out = np.clip(grade(rgb), 0.0, 1.0)
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {SIZE}", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    lines += [f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in out]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path} ({len(out)} entries)")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_cube(OUT_DIR / "warm_film.cube", "Warm Film", warm_film)
    write_cube(OUT_DIR / "bright_clean.cube", "Bright Clean", bright_clean)
