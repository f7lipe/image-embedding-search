#!/usr/bin/env python3
"""Generate simple sample images for testing/demonstration purposes.

Run once from the project root:
    python scripts/create_sample_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError:
        print("Pillow is required: pip install Pillow")
        sys.exit(1)

    out_dir = Path("data/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        ("red_circle.jpg", (255, 100, 100), "circle", "Red Circle"),
        ("blue_square.jpg", (100, 100, 255), "square", "Blue Square"),
        ("green_triangle.jpg", (100, 200, 100), "triangle", "Green Triangle"),
        ("yellow_star.jpg", (255, 220, 50), "star", "Yellow Star"),
        ("purple_diamond.jpg", (180, 80, 200), "diamond", "Purple Diamond"),
        ("orange_oval.jpg", (255, 160, 50), "oval", "Orange Oval"),
        ("cyan_cross.jpg", (50, 200, 220), "cross", "Cyan Cross"),
        ("white_hexagon.jpg", (220, 220, 220), "hexagon", "White Hexagon"),
    ]

    for filename, color, shape, label in samples:
        img = Image.new("RGB", (224, 224), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        cx, cy, s = 112, 112, 80  # center and half-size

        if shape == "circle":
            draw.ellipse([cx - s, cy - s, cx + s, cy + s], fill=color)
        elif shape == "square":
            draw.rectangle([cx - s, cy - s, cx + s, cy + s], fill=color)
        elif shape == "triangle":
            draw.polygon([(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)], fill=color)
        elif shape == "star":
            import math
            pts = []
            for i in range(10):
                angle = math.radians(i * 36 - 90)
                r = s if i % 2 == 0 else s // 2
                pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
            draw.polygon(pts, fill=color)
        elif shape == "diamond":
            draw.polygon([(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)], fill=color)
        elif shape == "oval":
            draw.ellipse([cx - s, cy - s // 2, cx + s, cy + s // 2], fill=color)
        elif shape == "cross":
            t = s // 4  # thickness
            draw.rectangle([cx - t, cy - s, cx + t, cy + s], fill=color)
            draw.rectangle([cx - s, cy - t, cx + s, cy + t], fill=color)
        elif shape == "hexagon":
            import math
            pts = [
                (cx + s * math.cos(math.radians(60 * i - 30)),
                 cy + s * math.sin(math.radians(60 * i - 30)))
                for i in range(6)
            ]
            draw.polygon(pts, fill=color)

        # Attempt to draw a label (font may not be available everywhere).
        try:
            font = ImageFont.load_default()
        except (OSError, AttributeError):
            font = None
        draw.text((10, 200), label, fill=(50, 50, 50), font=font)

        path = out_dir / filename
        img.save(str(path))
        print(f"  Created {path}")

    print(f"\nSample images written to '{out_dir}/'.")


if __name__ == "__main__":
    main()
