"""Generate synthetic damage-photo fixtures for ClaimSight Slice 3 (Vision Agent).

IMPORTANT LIMITATION: These images are simple PIL car-silhouette shapes with
colored irregular "damage" markers. They will NOT produce meaningful zero-shot
detections from OWL-ViT / CLIP / BLIP. Their only purpose is to prove the
Vision Agent pipeline runs without crashing and returns well-formed
VisionOutput for automated tests. For meaningful confidences, drop real car
damage photos into fixtures/images/real/ (see fixtures/images/README.md).

Outputs:
  fixtures/images/synthetic_1.jpg
  fixtures/images/synthetic_2.jpg
  fixtures/images/synthetic_3.jpg
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

IMAGES_DIR = Path(__file__).resolve().parent / "images"


def _draw_car_silhouette(draw: ImageDraw.ImageDraw, *, y_offset: int = 0) -> None:
    """Simple side-view car: body rectangle + cabin + wheels."""
    body = [80, 180 + y_offset, 520, 280 + y_offset]
    cabin = [180, 110 + y_offset, 400, 180 + y_offset]
    draw.rounded_rectangle(body, radius=20, fill=(70, 90, 120))
    draw.rounded_rectangle(cabin, radius=12, fill=(100, 130, 160))
    draw.ellipse([120, 250 + y_offset, 200, 320 + y_offset], fill=(40, 40, 40))
    draw.ellipse([400, 250 + y_offset, 480, 320 + y_offset], fill=(40, 40, 40))


def _draw_damage_markers(draw: ImageDraw.ImageDraw, variant: int) -> None:
    """Colored irregular shapes at known locations (not real damage textures)."""
    if variant == 1:
        # Red "dent" blob on front bumper area
        draw.polygon([(90, 220), (130, 200), (160, 230), (120, 260)], fill=(200, 40, 40))
        draw.ellipse([100, 210, 150, 250], fill=(180, 30, 30))
    elif variant == 2:
        # Yellow scratch streak on door
        draw.line([(220, 200), (360, 240)], fill=(220, 200, 40), width=6)
        draw.line([(225, 210), (355, 250)], fill=(200, 180, 20), width=3)
        # Blue glass shatter-like shards near windshield
        draw.polygon([(250, 120), (280, 140), (260, 160)], fill=(80, 160, 220))
        draw.polygon([(290, 125), (320, 145), (300, 165)], fill=(60, 140, 200))
    else:
        # Orange bumper damage + dark "airbag" blob in cabin
        draw.polygon(
            [(100, 240), (170, 210), (200, 270), (130, 290)],
            fill=(230, 120, 40),
        )
        draw.ellipse([260, 130, 340, 190], fill=(230, 230, 230))
        draw.ellipse([270, 140, 330, 180], fill=(200, 200, 200))


def build_synthetic_image(path: Path, variant: int) -> None:
    img = Image.new("RGB", (600, 400), color=(220, 225, 230))
    draw = ImageDraw.Draw(img)
    _draw_car_silhouette(draw)
    _draw_damage_markers(draw, variant)
    # Label so humans can tell fixtures apart; models ignore this text.
    draw.text((10, 10), f"synthetic_{variant}", fill=(80, 80, 80))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=90)


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    (IMAGES_DIR / "real").mkdir(parents=True, exist_ok=True)
    for i in (1, 2, 3):
        out = IMAGES_DIR / f"synthetic_{i}.jpg"
        build_synthetic_image(out, i)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
