# =============================================================================
# backend/tools/augment_dataset.py
#
# Augments training images when only a few photos are available.
# Creates 8 variations per image: flips, rotations, brightness, blur.
#
# Run from backend/ directory:
#   python tools/augment_dataset.py
# =============================================================================

import os
import glob
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

TRAIN_DIR = Path(__file__).resolve().parent.parent / "test_dataset" / "train"
IMG_EXTS  = ("*.jpg", "*.jpeg", "*.png")


def augment_image(img: Image.Image, base_stem: str, out_dir: Path):
    """Saves 8 augmented variants of *img* into *out_dir*."""
    variants = []

    # 1. Horizontal flip
    variants.append(("flip_h",      img.transpose(Image.FLIP_LEFT_RIGHT)))
    # 2. Small left tilt
    variants.append(("rot_neg10",   img.rotate(-10, expand=False)))
    # 3. Small right tilt
    variants.append(("rot_pos10",   img.rotate(+10, expand=False)))
    # 4. Slight left tilt
    variants.append(("rot_neg5",    img.rotate(-5,  expand=False)))
    # 5. Brighter
    variants.append(("bright_hi",   ImageEnhance.Brightness(img).enhance(1.3)))
    # 6. Darker
    variants.append(("bright_lo",   ImageEnhance.Brightness(img).enhance(0.7)))
    # 7. Slightly blurred (simulates motion / lower-res camera)
    variants.append(("blur",        img.filter(ImageFilter.GaussianBlur(radius=1.2))))
    # 8. Flip + rotate
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    variants.append(("flip_rot5",   flipped.rotate(5, expand=False)))

    saved = 0
    for suffix, var_img in variants:
        out_path = out_dir / f"{base_stem}_aug_{suffix}.jpg"
        if not out_path.exists():
            var_img.convert("RGB").save(str(out_path), quality=92)
            saved += 1
    return saved


def main():
    users = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
    if not users:
        print(f"No user folders found in {TRAIN_DIR}")
        return

    total_saved = 0
    for user_dir in sorted(users):
        originals = []
        for ext in IMG_EXTS:
            originals.extend(user_dir.glob(ext))
        # skip files that are already augmented to avoid double-augmenting
        originals = [p for p in originals if "_aug_" not in p.stem]

        print(f"  {user_dir.name}: {len(originals)} original(s) → augmenting …")
        for img_path in sorted(originals):
            try:
                img = Image.open(str(img_path)).convert("RGB")
                n = augment_image(img, img_path.stem, user_dir)
                total_saved += n
                print(f"    {img_path.name} → +{n} variants")
            except Exception as e:
                print(f"    WARN: could not process {img_path.name}: {e}")

    print(f"\nDone.  {total_saved} new images written to {TRAIN_DIR}")


if __name__ == "__main__":
    main()
