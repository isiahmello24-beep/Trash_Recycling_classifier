"""
review_near_dups.py - spot-check the near-duplicate rules by eye.

Runs the same duplicate search as build_splits.py (nothing is copied or
written to data/), takes the removed pairs with the HIGHEST distances, which
are the borderline ones, and draws them side by side in one picture:

    results/near_dup_review.png   (git-ignored, like every .png)

Each row shows the removed image on the left and the image it matched on the
right, with the pHash and dHash distances. Open the picture and count how
many pairs are really the same photo.

Run from the repo root:
    python src/data/review_near_dups.py              # top 20 pairs
    python src/data/review_near_dups.py --pairs 30
"""

import argparse

from PIL import Image, ImageDraw

# build_splits.py sits in the same folder, so Python can import it directly.
from build_splits import RESULTS_DIR, find_duplicates, load_config, rel

SHEET_PATH = RESULTS_DIR / "near_dup_review.png"
THUMB = 220  # size of each thumbnail in pixels


def main():
    parser = argparse.ArgumentParser(description="Contact sheet of the most borderline near-duplicate pairs.")
    parser.add_argument("--pairs", type=int, default=20, help="how many pairs to show (default 20)")
    args = parser.parse_args()

    config, active, _ = load_config()
    _, removed_rows, _, _ = find_duplicates(config, active)

    # Only rows that came from a pair comparison (not corrupted files), highest distance first.
    pairs = [r for r in removed_rows if "pair" in r]
    pairs.sort(key=lambda r: (-r["distance"], -r["dhash_distance"], r["file"]))
    pairs = pairs[: args.pairs]

    # Two pairs per row of the sheet: [removed | matched]   [removed | matched]
    cols, label_h = 2, 34
    rows = (len(pairs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 2 * THUMB + (cols - 1) * 20, rows * (THUMB + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    print(f"\nTop {len(pairs)} removed pairs by distance:")
    for k, r in enumerate(pairs):
        row, col = divmod(k, cols)
        x0, y0 = col * (2 * THUMB + 20), row * (THUMB + label_h)
        for m, path in enumerate((r["pair"], r["pair_with"])):
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((THUMB - 4, THUMB - 4))
                sheet.paste(img, (x0 + m * THUMB + 2, y0 + 2))
        reason = "conflict" if r["reason"].startswith("label_conflict") else r["reason"].replace("_duplicate", "")
        draw.text((x0 + 4, y0 + THUMB + 2),
                  f"#{k + 1} {reason}  pHash {r['distance']}  dHash {r['dhash_distance']}", fill="black")
        draw.text((x0 + 4, y0 + THUMB + 16),
                  f"{r['pair'].parent.name}/{r['pair'].name}  vs  {r['pair_with'].parent.name}/{r['pair_with'].name}",
                  fill="gray")
        print(f"  #{k + 1:<3} pHash {r['distance']}  dHash {r['dhash_distance']}  {reason:<9} "
              f"{rel(r['pair'])}  vs  {rel(r['pair_with'])}")

    RESULTS_DIR.mkdir(exist_ok=True)
    sheet.save(SHEET_PATH)
    print(f"\nSaved {rel(SHEET_PATH)}")


if __name__ == "__main__":
    main()
