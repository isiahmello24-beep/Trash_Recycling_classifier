"""
build_splits.py - turn the raw downloaded datasets into train / validation / test data.

What it does, in order:
  1. Read configs/label_map.yaml to learn which raw folder -> which training class.
  2. Collect every image from those folders under data/raw/.
     - Skip folders marked VERIFY (not decided yet).
     - Skip folders whose class is excluded (hazardous when num_classes is 3).
     - Drop corrupted files that Pillow cannot fully read.
  3. Remove EXACT duplicates (identical bytes -> identical SHA-256 hash).
  4. Remove NEAR duplicates using two image fingerprints, pHash and dHash.
     The cutoffs live under `near_duplicates:` in label_map.yaml.
     - Same class:      pHash <= same_class_max_phash  AND dHash <= max_dhash -> near duplicate.
     - Different class: pHash <= cross_class_max_phash AND dHash <= max_dhash -> label conflict,
       and the whole group is dropped because we cannot trust its labels.
     An image only joins a group if it matches the group's kept copy directly.
  5. Shuffle with seed 42 and split each class 70 / 15 / 15.
  6. Copy files into data/train, data/validation, data/test (one sub-folder per class).
  7. Write CSV manifests: data/splits/train.csv, validation.csv, test.csv (path, label).
  8. Re-check the copied files: no exact or near duplicate may sit in two splits.
  9. Compute two sets of class weights from the TRAIN split only -> results/class_weights.json.
 10. Write every removed file to results/removed_duplicates.csv.
 11. Print the always-guess-the-biggest-class baseline on the validation split.
 12. Check the arithmetic: images found - images removed = images in the splits.

Run from the repo root:
    python src/data/build_splits.py                    # uses num_classes from label_map.yaml
    python src/data/build_splits.py --num-classes 3    # one-off override without editing the yaml
    python src/data/build_splits.py --dry-run          # count only, write nothing

Re-running deletes and rebuilds the outputs, so the result is always the same
for the same raw data and seed. Images under data/ are never committed to git.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
import yaml
from PIL import Image

# --- Paths. Everything is relative to the repo root, two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
CONFIG_PATH = REPO_ROOT / "configs" / "label_map.yaml"
SPLIT_NAMES = ["train", "validation", "test"]
SPLIT_DIRS = {name: REPO_ROOT / "data" / name for name in SPLIT_NAMES}
MANIFEST_DIR = REPO_ROOT / "data" / "splits"
RESULTS_DIR = REPO_ROOT / "results"
REMOVED_CSV = RESULTS_DIR / "removed_duplicates.csv"
WEIGHTS_JSON = RESULTS_DIR / "class_weights.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REMOVED_COLUMNS = ["file", "kept_copy", "reason", "distance", "dhash_distance"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def rel(path):
    """Path relative to the repo root, with forward slashes (same on Mac and Windows)."""
    return Path(path).relative_to(REPO_ROOT).as_posix()


def file_hash(path):
    """SHA-256 of the file bytes. Two files with the same hash are exact duplicates."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # Read 1 MB at a time so big files never load fully into memory.
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_hashes(path):
    """Return (pHash, dHash) of an image, each a 64-bit number.

    Both shrink the image to a tiny grayscale version:
      - pHash keeps the overall light/dark pattern (good at spotting re-saved or resized copies).
      - dHash keeps which way brightness changes from pixel to pixel (edges).
    Requiring BOTH to be close cuts false matches, like two different bottles
    that happen to have the same outline on a white background.
    """
    with Image.open(path) as img:
        phash = int(str(imagehash.phash(img)), 16)  # hex string -> integer
        dhash = int(str(imagehash.dhash(img)), 16)
    return phash, dhash


def hamming_distances(one_hash, all_hashes):
    """How many bits differ between one hash and every hash in a numpy array."""
    return np.bitwise_count(np.uint64(one_hash) ^ all_hashes)


def is_near_match(same_label, phash_dist, dhash_dist, rules):
    """True if two images count as the same picture under the label_map.yaml rules.

    Works on single numbers or on whole numpy arrays at once.
    Same label uses the looser pHash cutoff; different labels use the stricter one,
    because a cross-class match throws away every image in the group.
    """
    phash_limit = np.where(same_label, rules["same_class_max_phash"], rules["cross_class_max_phash"])
    return (phash_dist <= phash_limit) & (dhash_dist <= rules["max_dhash"])


def is_valid_image(path):
    """True if Pillow can fully read the file. Catches corrupted or truncated images."""
    try:
        with Image.open(path) as img:
            img.verify()  # checks the file structure
        with Image.open(path) as img:
            img.load()    # decodes every pixel; a cut-off file fails here
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Step 1-2: read the config and collect images
# ---------------------------------------------------------------------------

def load_config(num_classes_override=None):
    """Read label_map.yaml and return (config, active class set, num_classes).

    The yaml has one entry under `class_sets` per class count (3 or 4).
    `num_classes` at the top picks which one is active. That is the
    "one line config change".
    """
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    num_classes = num_classes_override or config["num_classes"]
    if num_classes not in config["class_sets"]:
        raise SystemExit(f"label_map.yaml has no class_sets entry for {num_classes}. Options: 3 or 4.")

    active = config["class_sets"][num_classes]
    print(f"Using {num_classes} classes: {active['classes']}  (excluded: {active['exclude'] or 'none'})")
    return config, active, num_classes


def folders_for_dataset(name, dataset_cfg):
    """Return a list of (folder_path, target_class) pairs for one dataset.

    Two shapes are allowed in the yaml:
      - `root` + `folders`: a dict of raw folder name -> target class
      - `label`: every image directly inside data/raw/<name>/ gets this class
    Returns an empty list if data/raw/<name>/ does not exist (not downloaded
    yet, or the dataset key is misspelled; main() prints a [skip] line for it).
    A missing `root` or `folders` entry is an error, so a typo there can never
    silently drop a class.
    """
    dataset_dir = RAW_DIR / name
    if not dataset_dir.exists():
        return []

    if "label" in dataset_cfg:
        return [(dataset_dir, dataset_cfg["label"])]

    dataset_root = dataset_dir / (dataset_cfg.get("root") or "")
    if not dataset_root.is_dir():
        raise SystemExit(f"label_map.yaml root '{dataset_cfg.get('root')}' has no folder at {rel(dataset_root)}")

    pairs = []
    for folder_name, target in dataset_cfg["folders"].items():
        folder = dataset_root / folder_name
        if not folder.is_dir():
            raise SystemExit(f"label_map.yaml key '{folder_name}' has no folder at {rel(folder)}")
        pairs.append((folder, target))
    return pairs


def collect_images(config, active, removed_rows):
    """Return (list of image dicts {"path", "label"}, number of image files found).

    Rules for each raw folder:
      - target VERIFY               -> skip (not decided yet)
      - target in active `exclude`  -> skip (e.g. hazardous in 3-class mode)
      - target not a known class    -> stop with an error (catches typos)
    Corrupted files count as found, are dropped, and are logged in removed_rows.
    """
    items = []
    n_found = 0
    stats = Counter()

    for dataset_name, dataset_cfg in config["datasets"].items():
        pairs = folders_for_dataset(dataset_name, dataset_cfg)
        if not pairs:
            print(f"  [skip] data/raw/{dataset_name}/ not found (not downloaded yet?)")
            continue

        for folder, target in pairs:
            if target == "VERIFY":
                stats["verify"] += 1
                continue
            if target in active["exclude"]:
                stats["excluded"] += 1
                continue
            if target not in active["classes"]:
                raise SystemExit(f"Unknown class '{target}' for {folder} in label_map.yaml")

            # A `label:` dataset only reads its own top level, so the README and
            # any sub-folders are ignored. `folders:` datasets read recursively.
            files = folder.iterdir() if "label" in dataset_cfg else folder.rglob("*")
            for path in sorted(files):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                n_found += 1
                if not is_valid_image(path):
                    removed_rows.append({"file": rel(path), "kept_copy": "", "reason": "corrupted",
                                         "distance": "", "dhash_distance": ""})
                    continue
                items.append({"path": path, "label": target})

    print(f"\nFound {n_found} images. Skipped {stats['verify']} VERIFY folders "
          f"and {stats['excluded']} excluded folders.")
    return items, n_found


# ---------------------------------------------------------------------------
# Step 3-4: duplicate removal
# ---------------------------------------------------------------------------

def remove_exact_duplicates(items, removed_rows):
    """Group images by SHA-256. Keep one per group; a mixed-label group is a conflict.

    Returns (kept items, exact duplicates removed, images dropped in exact conflicts).
    """
    groups = defaultdict(list)  # sha256 -> [items with those exact bytes]
    for item in items:
        groups[file_hash(item["path"])].append(item)

    kept, n_dupes, n_conflict = [], 0, 0
    for group in groups.values():
        first = group[0]
        labels = {item["label"] for item in group}
        if len(group) == 1:
            kept.append(first)
        elif len(labels) == 1:
            # Same picture, same label: keep the first copy, drop the others.
            kept.append(first)
            for dup in group[1:]:
                removed_rows.append({"file": rel(dup["path"]), "kept_copy": rel(first["path"]),
                                     "reason": "exact_duplicate", "distance": 0, "dhash_distance": 0,
                                     "pair": dup["path"], "pair_with": first["path"]})
                n_dupes += 1
        else:
            # Same picture, different labels: we cannot trust either label, drop all.
            conflict = "label_conflict (" + " vs ".join(sorted(labels)) + ")"
            for item in group:
                note = "none (first copy of conflict group)" if item is first else f"none (exact copy of {rel(first['path'])})"
                removed_rows.append({"file": rel(item["path"]), "kept_copy": note, "reason": conflict,
                                     "distance": 0, "dhash_distance": 0})
                n_conflict += 1

    # Keep the original order so results do not depend on dictionary order.
    order = {id(item): i for i, item in enumerate(items)}
    kept.sort(key=lambda item: order[id(item)])
    return kept, n_dupes, n_conflict


def remove_near_duplicates(items, removed_rows, rules):
    """Remove near duplicates across ALL classes, with no chaining.

    Walks the images in file order. Each image that is still kept becomes an
    anchor (a "kept copy") and is compared directly with every later image
    that is still kept. A later image joins the anchor's group only if it
    matches the ANCHOR itself (see is_near_match). If A matches B and B
    matches C, C is not grouped with A unless C also matches A.
      - Every match has the anchor's label -> keep the anchor, drop the matches (near_duplicate).
      - Any match has a different label    -> drop the anchor and all its matches (label_conflict).
    Returns (kept items, near duplicates removed, conflict groups, conflict images).
    """
    for item in items:
        item["phash"], item["dhash"] = image_hashes(item["path"])
    phashes = np.array([item["phash"] for item in items], dtype=np.uint64)
    dhashes = np.array([item["dhash"] for item in items], dtype=np.uint64)
    labels = np.array([item["label"] for item in items])

    removed = np.zeros(len(items), dtype=bool)  # True once an image is dropped
    n_near, n_conflict_groups, n_conflict_images = 0, 0, 0

    for i, anchor in enumerate(items):
        if removed[i]:
            continue
        p_dist = hamming_distances(anchor["phash"], phashes)
        d_dist = hamming_distances(anchor["dhash"], dhashes)
        same = labels == anchor["label"]
        match = is_near_match(same, p_dist, d_dist, rules)
        # Only later images that are still kept can join this anchor's group.
        group = [j for j in np.nonzero(match)[0] if j > i and not removed[j]]
        if not group:
            continue

        if all(same[j] for j in group):
            for j in group:
                removed[j] = True
                removed_rows.append({"file": rel(items[j]["path"]), "kept_copy": rel(anchor["path"]),
                                     "reason": "near_duplicate", "distance": int(p_dist[j]),
                                     "dhash_distance": int(d_dist[j]),
                                     "pair": items[j]["path"], "pair_with": anchor["path"]})
                n_near += 1
        else:
            # Labels disagree: drop the anchor and every image in its group.
            n_conflict_groups += 1
            group_labels = {anchor["label"]} | {items[j]["label"] for j in group}
            conflict = "label_conflict (" + " vs ".join(sorted(group_labels)) + ")"
            # The anchor's row points at its closest match with a different label.
            cross = [j for j in group if not same[j]]
            closest = min(cross, key=lambda j: (int(p_dist[j]), int(d_dist[j]), j))
            removed[i] = True
            removed_rows.append({"file": rel(anchor["path"]),
                                 "kept_copy": f"none (anchor of conflict group, matches {rel(items[closest]['path'])})",
                                 "reason": conflict, "distance": int(p_dist[closest]),
                                 "dhash_distance": int(d_dist[closest]),
                                 "pair": anchor["path"], "pair_with": items[closest]["path"]})
            for j in group:
                removed[j] = True
                removed_rows.append({"file": rel(items[j]["path"]),
                                     "kept_copy": f"none (matches {rel(anchor['path'])})",
                                     "reason": conflict, "distance": int(p_dist[j]),
                                     "dhash_distance": int(d_dist[j]),
                                     "pair": items[j]["path"], "pair_with": anchor["path"]})
            n_conflict_images += 1 + len(group)

    kept = [item for i, item in enumerate(items) if not removed[i]]
    return kept, n_near, n_conflict_groups, n_conflict_images


# ---------------------------------------------------------------------------
# Step 5-7: split, copy, manifests
# ---------------------------------------------------------------------------

def split_list(items, ratios, seed):
    """Shuffle `items` with a fixed seed, then cut into train / validation / test."""
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["validation"])
    return {
        "train": items[:n_train],
        "validation": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],  # the remainder, so nothing is lost to rounding
    }


def copy_splits_and_write_manifests(splits_by_class):
    """Copy images into data/<split>/<class>/ and write data/splits/<split>.csv.

    Filenames get a dataset + folder prefix so two datasets that both have a
    file called 1.jpg never overwrite each other. The CSV paths point at the
    copied files and are relative to the repo root.
    """
    for d in list(SPLIT_DIRS.values()) + [MANIFEST_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for split_name in SPLIT_NAMES:
        rows = []
        for class_name in sorted(splits_by_class):
            out_dir = SPLIT_DIRS[split_name] / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for item in splits_by_class[class_name][split_name]:
                src = item["path"]
                # data/raw/garbage12_enhanced/.../battery/battery7.jpg
                #   -> garbage12_enhanced__battery__battery7.jpg
                dataset = src.relative_to(RAW_DIR).parts[0]
                dst = out_dir / f"{dataset}__{src.parent.name}__{src.name}"
                shutil.copy2(src, dst)
                rows.append({"path": rel(dst), "label": class_name})

        rows.sort(key=lambda row: row["path"])
        with open(MANIFEST_DIR / f"{split_name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Step 8: verify the output on disk
# ---------------------------------------------------------------------------

def split_files_on_disk():
    """List (split name, class name, path) for every copied image in data/<split>/<class>/."""
    files = []
    for split_name, split_dir in SPLIT_DIRS.items():
        files += [(split_name, p.parent.name, p) for p in sorted(split_dir.rglob("*")) if p.is_file()]
    return files


def check_no_overlap(rules):
    """Re-hash every copied file and confirm no exact or near duplicate spans two splits.

    This reads the files actually on disk in data/train, data/validation and
    data/test, so it checks the real output, not just our in-memory lists.
    Near duplicates use the same rules as the removal step.
    """
    files = split_files_on_disk()

    # Exact: the same SHA-256 must never show up in two splits.
    splits_per_sha = defaultdict(set)
    for split_name, _, path in files:
        splits_per_sha[file_hash(path)].add(split_name)
    exact_cross = sum(1 for s in splits_per_sha.values() if len(s) > 1)

    # Near: compare every pair; count matching pairs that sit in different splits.
    hashes = [image_hashes(path) for _, _, path in files]
    phashes = np.array([h[0] for h in hashes], dtype=np.uint64)
    dhashes = np.array([h[1] for h in hashes], dtype=np.uint64)
    splits = np.array([f[0] for f in files])
    labels = np.array([f[1] for f in files])
    near_cross = 0
    for i in range(len(files)):
        later = slice(i + 1, None)
        match = is_near_match(labels[later] == labels[i],
                              hamming_distances(phashes[i], phashes[later]),
                              hamming_distances(dhashes[i], dhashes[later]), rules)
        near_cross += int(np.sum(match & (splits[later] != splits[i])))

    if exact_cross or near_cross:
        raise SystemExit(f"FAILED: {exact_cross} exact and {near_cross} near duplicates span two splits.")
    print("Overlap check passed: no exact or near duplicate appears in more than one split.")


def check_counts(n_found, removed_counts, n_final, n_removed_rows):
    """Stop with a clear message unless: images found - every removal = images in the splits.

    Also checks that results/removed_duplicates.csv has exactly one row per removed image.
    (A plain `assert` would be skipped if Python runs with -O, so this uses SystemExit.)
    """
    n_removed = sum(removed_counts.values())
    parts = " - ".join(f"{n} {name}" for name, n in removed_counts.items())
    print(f"\nCount check: {n_found} found - {parts} = {n_found - n_removed}; "
          f"{n_final} in the splits; {n_removed_rows} rows in {rel(REMOVED_CSV)}.")
    if n_found - n_removed != n_final or n_removed_rows != n_removed:
        raise SystemExit(f"FAILED count check: {n_found} found - {n_removed} removed = {n_found - n_removed}, "
                         f"but the splits hold {n_final} images and the removed list has {n_removed_rows} rows.")
    print("Count check passed.")


# ---------------------------------------------------------------------------
# Step 9-11: class weights, removed-files log, baseline
# ---------------------------------------------------------------------------

def compute_class_weights(splits_by_class, num_classes):
    """Two sets of class weights, both from the TRAIN split only.

    inverse_frequency:      total_train / (num_classes * class_count)
        Rare classes get a weight above 1, common ones below 1.
    sqrt_inverse_frequency: sqrt(total_train / (num_classes * class_count)),
        then divided by the average so the weights average to exactly 1.
        A gentler push toward rare classes; often more stable in training.
    """
    class_order = sorted(splits_by_class)  # alphabetical = torchvision ImageFolder order
    train_counts = {c: len(splits_by_class[c]["train"]) for c in class_order}
    total_train = sum(train_counts.values())
    if 0 in train_counts.values():
        raise SystemExit(f"A class has no training images: {train_counts}")

    inverse = {c: total_train / (num_classes * n) for c, n in train_counts.items()}
    sqrt_raw = {c: math.sqrt(w) for c, w in inverse.items()}
    sqrt_mean = sum(sqrt_raw.values()) / len(sqrt_raw)
    sqrt_norm = {c: w / sqrt_mean for c, w in sqrt_raw.items()}

    def weight_set(formula, by_class):
        rounded = {c: round(by_class[c], 4) for c in class_order}
        return {"formula": formula, "by_class": rounded, "in_class_order": [rounded[c] for c in class_order]}

    return {
        "num_classes": num_classes,
        "class_order": class_order,
        "train_counts": train_counts,
        "train_total": total_train,
        "train_percent": {c: round(100 * n / total_train, 2) for c, n in train_counts.items()},
        "weights": {
            "inverse_frequency": weight_set(
                "total_train / (num_classes * class_train_count)", inverse),
            "sqrt_inverse_frequency": weight_set(
                "sqrt(total_train / (num_classes * class_train_count)), divided by the mean so the weights average to 1",
                sqrt_norm),
        },
    }


def print_class_weights(data):
    """Show train share and both weight sets, one line per class."""
    w = data["weights"]
    print(f"\nTrain split share and class weights ({rel(WEIGHTS_JSON)}):")
    print(f"  {'class':<12}{'train':>7}{'% train':>10}{'inverse':>10}{'sqrt inv':>10}")
    for c in data["class_order"]:
        print(f"  {c:<12}{data['train_counts'][c]:>7}{data['train_percent'][c]:>9.2f}%"
              f"{w['inverse_frequency']['by_class'][c]:>10.4f}{w['sqrt_inverse_frequency']['by_class'][c]:>10.4f}")


def print_majority_baseline(splits_by_class):
    """Accuracy and macro F1 on validation if a model always guessed the biggest train class.

    For the guessed class: precision = accuracy, recall = 1, so F1 = 2*acc/(acc+1).
    Every other class is never predicted, so its F1 is 0.
    Macro F1 is the plain average of the per-class F1 scores.
    """
    majority = max(splits_by_class, key=lambda c: len(splits_by_class[c]["train"]))
    n_val = sum(len(s["validation"]) for s in splits_by_class.values())
    accuracy = len(splits_by_class[majority]["validation"]) / n_val
    f1_majority = 2 * accuracy / (accuracy + 1)
    macro_f1 = f1_majority / len(splits_by_class)
    print(f"\nBaseline 'always guess {majority}' on validation ({n_val} images): "
          f"accuracy {100 * accuracy:.2f}%, macro F1 {macro_f1:.4f}")


def write_removed_csv(removed_rows):
    """Write every dropped file with the reason, sorted by file path."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(REMOVED_CSV, "w", newline="") as f:
        # extrasaction="ignore" skips the helper keys ("pair", "pair_with") used by the spot check.
        writer = csv.DictWriter(f, fieldnames=REMOVED_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(removed_rows, key=lambda row: row["file"]))


def print_table(splits_by_class):
    """Print class counts per split as a small table. Returns the grand total."""
    print(f"\n{'class':<12}{'train':>8}{'validation':>12}{'test':>8}{'total':>8}")
    print("-" * 48)
    totals = Counter()
    for class_name in sorted(splits_by_class):
        counts = {k: len(v) for k, v in splits_by_class[class_name].items()}
        total = sum(counts.values())
        totals.update(counts)
        totals["total"] += total
        print(f"{class_name:<12}{counts['train']:>8}{counts['validation']:>12}{counts['test']:>8}{total:>8}")
    print("-" * 48)
    print(f"{'TOTAL':<12}{totals['train']:>8}{totals['validation']:>12}{totals['test']:>8}{totals['total']:>8}")
    return totals["total"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_duplicates(config, active):
    """Steps 2-4. Returns (kept items, removed rows, images found, removal counts).

    Shared with src/data/review_near_dups.py so the spot check uses the exact same rules.
    """
    rules = config["near_duplicates"]
    removed_rows = []  # one dict per dropped file, written to results/removed_duplicates.csv
    items, n_found = collect_images(config, active, removed_rows)
    if not items:
        raise SystemExit("No images found. Run src/data/download.py first.")
    n_corrupted = len(removed_rows)

    items, n_exact, n_exact_conflict = remove_exact_duplicates(items, removed_rows)
    print(f"Checking near duplicates across all classes: same class pHash <= {rules['same_class_max_phash']}, "
          f"cross class pHash <= {rules['cross_class_max_phash']}, and dHash <= {rules['max_dhash']}...")
    items, n_near, n_groups, n_near_conflict = remove_near_duplicates(items, removed_rows, rules)

    removed_counts = {
        "corrupted": n_corrupted,
        "exact duplicates": n_exact,
        "near duplicates": n_near,
        "label-conflict images": n_exact_conflict + n_near_conflict,
    }
    print(f"\nRemoved {n_exact} exact duplicates, {n_near} near duplicates, and "
          f"{n_exact_conflict + n_near_conflict} images in label conflicts ({n_groups} near-duplicate groups"
          + (f" + {n_exact_conflict} exact-duplicate images" if n_exact_conflict else "")
          + f"). Dropped {n_corrupted} corrupted files. {len(items)} images remain.")
    return items, removed_rows, n_found, removed_counts


def main():
    parser = argparse.ArgumentParser(description="Build train/validation/test data from data/raw/.")
    parser.add_argument("--num-classes", type=int, choices=[3, 4], help="override num_classes from label_map.yaml")
    parser.add_argument("--dry-run", action="store_true", help="count and print, but write nothing")
    args = parser.parse_args()

    config, active, num_classes = load_config(args.num_classes)
    ratios = config["split"]["ratios"]
    seed = config["split"]["seed"]
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise SystemExit("split ratios in label_map.yaml must add up to 1.0")

    items, removed_rows, n_found, removed_counts = find_duplicates(config, active)

    by_class = defaultdict(list)
    for item in items:
        by_class[item["label"]].append(item)
    # Every active class must still have images, or the class order and the
    # weights would not line up with num_classes.
    missing = set(active["classes"]) - set(by_class)
    if missing:
        raise SystemExit(f"No images left for class(es) {sorted(missing)}. Check data/raw/ and label_map.yaml.")
    splits_by_class = {cls: split_list(by_class[cls], ratios, seed) for cls in sorted(by_class)}
    n_in_splits = print_table(splits_by_class)

    weights = compute_class_weights(splits_by_class, num_classes)
    print_class_weights(weights)
    print_majority_baseline(splits_by_class)

    if args.dry_run:
        check_counts(n_found, removed_counts, n_in_splits, len(removed_rows))
        print("\n(dry run: nothing was written)")
        return

    copy_splits_and_write_manifests(splits_by_class)
    write_removed_csv(removed_rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(WEIGHTS_JSON, "w") as f:
        json.dump(weights, f, indent=2)
        f.write("\n")
    print()
    check_no_overlap(config["near_duplicates"])

    # Final arithmetic check against what is really on disk.
    n_on_disk = len(split_files_on_disk())
    with open(REMOVED_CSV) as f:
        n_csv_rows = sum(1 for _ in csv.DictReader(f))
    check_counts(n_found, removed_counts, n_on_disk, n_csv_rows)
    print(f"Wrote data/train, data/validation, data/test, data/splits/*.csv, "
          f"{rel(REMOVED_CSV)}, {rel(WEIGHTS_JSON)}  (seed={seed}, {num_classes} classes)")


if __name__ == "__main__":
    main()
