# Midterm slides: the data section

Three slides. Every number comes from the current real split
(`Split version: d010b6a  manifest_sha256: 4d578382b730`) and from
`docs/dataset_inventory.md`. Nothing here comes from the fake sample CSVs.
Charts: `results/class_counts.png` and `results/split_counts.png`
(regenerate with `python src/data/plot_counts.py`).

---

## Slide 1: Where the data came from

- One public dataset so far: Kaggle "Garbage Classification (12 classes) ENHANCED" (huberthamelin), license ODbL for the database, DbCL for the contents. 15,870 images, web-scraped, clean backgrounds.
- We use 10 of its 12 folders, mapped to our 4 San Diego classes: 7 kinds of recycling, biological = compost, trash, and battery = hazardous. Clothes dropped, shoes undecided.
- **8,502 images** in the splits after cleaning: recycling 5,788, compost 1,025, hazardous 972, trash 717. (Chart: `results/class_counts.png`.)
- Split 70 / 15 / 15 with a fixed seed: 5,961 train, 1,271 validation, 1,270 test.
- Still to come: Amy's own photos (about 8,000) and 50 of my battery photos. Real-world shots, not web images.

**Speaker notes (about 50 seconds):**
So, where did the data come from. Right now it's one Kaggle dataset, the 12-class garbage set: about 15,900 web images with clean backgrounds, under an open database license. We map ten of its folders onto our four San Diego classes: seven kinds of recycling, food waste to compost, trash, and batteries to hazardous. After cleaning we have 8,502 images, and the chart shows recycling is way bigger than the rest. We split it 70-15-15 with a fixed seed, so everyone on the team gets the exact same train, validation and test sets. Next up is real photos: Amy has about eight thousand of her own and I'm adding fifty battery shots, because web images are too clean for a Pi camera.
---

## Slide 2: How we cleaned it

- **Exact copies:** 2 identical files found and removed.
- **Near copies:** 57 removed. Two image fingerprints (pHash and dHash) both have to agree, after we checked borderline matches by eye. Our first rule caught 125, but about half of those were different objects that just looked alike.
- **Label conflicts:** look-alikes with different labels are never deleted; they go to a review file. 0 open today (a looser rule flagged 6 groups, all false matches like bottles vs toothbrushes).
- **Leaks:** a check found 17 validation/test images that were near-copies of training images. They're out, so test scores can't be inflated by memorized photos.
- **Clothes dropped** (5,286 images, team decision Oct 18). **Before → after:** 8,578 images considered → 8,502 in the splits. Sideways iPhone photos get turned upright automatically before any of this runs.

**Speaker notes (about 50 seconds):**
Cleaning was most of the work. Exact duplicates were easy, there were two. Near-duplicates were the interesting part: we fingerprint every image two ways, and both have to agree before we call it a copy. Our first rule flagged 125, and when I checked them by eye about half were different objects with the same outline. The stricter rule gives 57. We never delete on a guess: look-alikes with different labels go into a review file for a human, and that file is empty right now. The part I'm proudest of is the leak check. Seventeen validation and test images were near-copies of training images, which would have inflated our scores, so they're gone. We also dropped clothes. Net: 8,578 images in, 8,502 out, every file accounted for.
---

## Slide 3: The imbalance problem

- Recycling is **68.1% of the training set** (4,059 of 5,961). Compost 12.0%, hazardous 11.4%, trash 8.4%.
- A model that always says "recycling" scores **68% accuracy** but only **0.20 macro F1** on validation. That's why we report macro F1, not accuracy.
- Class weights fix the pull toward recycling: inverse-frequency weights are compost 2.08, hazardous 2.19, recycling 0.37, trash 2.96 (a gentler square-root version is also saved).
- The "I'm not sure" cutoff is **tuned on validation and reported on test**. Picking it on test would mean choosing the number that flatters us.
- Real model results: pending real model results.

**Speaker notes (about 50 seconds):**
Here's the catch with our data: two thirds of the training images are recycling. A model that just says recycling every time gets 68 percent accuracy, which sounds okay, but its macro F1 is 0.20, which is terrible. Macro F1 treats all four classes equally, so that's the number we track. To stop the model leaning on recycling, training gets a weight per class: a trash mistake costs about eight times a recycling mistake. The other piece is the I'm-not-sure answer. The Pi only answers when the model is confident enough, and we pick that cutoff on the validation set, then report what it does on test. Picking it on test would mean choosing whichever number looks best. The real model numbers are pending until Amy's results are in.