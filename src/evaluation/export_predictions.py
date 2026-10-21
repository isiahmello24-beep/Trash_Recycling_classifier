"""
export_predictions.py - run a trained model over the validation and test splits
and write the prediction CSVs that src/eval/evaluate.py reads.

Output (git-ignored, under results/):
    results/val_predictions.csv    one row per image in data/validation
    results/test_predictions.csv   one row per image in data/test
Columns: path, true_label, predicted_label, confidence
confidence is the softmax probability of the predicted label (0 to 1).

The images and their true labels come from data/splits/validation.csv and
data/splits/test.csv, the manifests that build_splits.py writes, so this script
always scores exactly the stamped split (see results/class_weights.json).

Class order: the model's output number i means class_order[i], where
class_order is the active class list from configs/label_map.yaml sorted
alphabetically. That is the order torchvision's ImageFolder uses when training
on data/train, and the order in results/class_weights.json.

Model file: any of these works for --model
    1. a state_dict (torch.save(model.state_dict(), path)) - the usual form;
       needs --arch to rebuild the network: mobilenet_v3_small, mobilenet_v3_large,
       efficientnet_b0 or resnet18 from torchvision, head replaced for our classes
    2. a whole model (torch.save(model, path))  - loaded as is
    3. TorchScript (torch.jit.save)             - loaded as is (older format)

# TODO(Amy): confirm these against your training code; they are guesses until then.
#   framework    PyTorch + torchvision            (no training code is in the repo yet)
#   image size   224 x 224                        (--image-size)
#   preprocessing resize to the size, ImageNet mean/std normalization, no crop
#   model file   models/best_model.pth (state_dict) + --arch mobilenet_v3_small

Run from the repo root:
    python src/evaluation/export_predictions.py --model models/best_model.pth --arch mobilenet_v3_small
    python src/evaluation/export_predictions.py --model models/model.pt --image-size 224
"""

import argparse
import csv
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import models, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "label_map.yaml"
MANIFEST_DIR = REPO_ROOT / "data" / "splits"
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_NAMES = {"validation": "val_predictions.csv", "test": "test_predictions.csv"}

# TODO(Amy): ImageNet statistics, right for torchvision models pretrained on ImageNet.
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

ARCHITECTURES = {
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "efficientnet_b0": models.efficientnet_b0,
    "resnet18": models.resnet18,
}


def class_order():
    """Class names in the order the model's outputs use: the active classes, alphabetical."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    classes = config["class_sets"][config["num_classes"]]["classes"]
    return sorted(classes)


def build_architecture(arch, num_classes):
    """A torchvision network with its last layer swapped for our number of classes.

    No pretrained weights are downloaded here (weights=None): the trained
    weights come from --model.
    """
    net = ARCHITECTURES[arch](weights=None)
    # Each family keeps its final layer in a different place.
    if arch.startswith("mobilenet_v3") or arch.startswith("efficientnet"):
        last = net.classifier[-1]
        net.classifier[-1] = torch.nn.Linear(last.in_features, num_classes)
    else:  # resnet
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    return net


def load_model(path, arch, num_classes, device):
    """Load the model file, whichever of the three forms it is."""
    try:
        loaded = torch.load(str(path), map_location=device, weights_only=False)
    except Exception:
        loaded = None  # not a torch.save file; maybe TorchScript, tried below
    if isinstance(loaded, torch.nn.Module):  # 2. whole model (also how a TorchScript file usually loads)
        model = loaded
        print(f"Loaded {path} as a whole PyTorch model.")
    elif isinstance(loaded, dict):  # 1. state_dict
        if arch is None:
            raise SystemExit(f"{path} is a state_dict (weights only), so the network must be rebuilt: "
                             f"add --arch, one of {', '.join(ARCHITECTURES)}.")
        # Some checkpoints wrap the weights: {"state_dict": ...} or {"model_state_dict": ...}.
        state = loaded.get("state_dict", loaded.get("model_state_dict", loaded))
        model = build_architecture(arch, num_classes)
        try:
            model.load_state_dict(state)
        except RuntimeError as err:  # wrong architecture, or a head trained for another number of classes
            raise SystemExit(f"{path} does not fit {arch} with {num_classes} classes {class_order()}.\n"
                             f"Check --arch and that the split was built with the same num_classes.\nDetails: {err}")
        print(f"Loaded {path} as a state_dict into {arch}.")
    elif loaded is None:  # 3. TorchScript that torch.load could not read (older format)
        try:
            model = torch.jit.load(str(path), map_location=device)
        except Exception as err:
            raise SystemExit(f"{path} is not a model I know how to load: {err}")
        print(f"Loaded {path} as TorchScript.")
    else:
        raise SystemExit(f"{path} is not a model I know how to load ({type(loaded).__name__}).")
    model.to(device)
    model.eval()  # switch off training-only behaviour such as dropout
    return model


def read_manifest(split_name):
    """Rows of data/splits/<split>.csv as a list of {"path", "label"}."""
    path = MANIFEST_DIR / f"{split_name}.csv"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(REPO_ROOT)} not found. Run python src/data/build_splits.py first.")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def predict_split(model, rows, classes, preprocess, device, batch_size):
    """Run the model over every image in the rows. Returns a list of output dicts."""
    out = []
    with torch.no_grad():  # no gradients needed, saves memory and time
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            images = []
            for r in batch_rows:
                with Image.open(REPO_ROOT / r["path"]) as img:
                    images.append(preprocess(img.convert("RGB")))
            logits = model(torch.stack(images).to(device))
            if logits.shape[1] != len(classes):
                raise SystemExit(f"The model has {logits.shape[1]} outputs but there are {len(classes)} classes "
                                 f"{classes}. Wrong model file, or the split was built with a different num_classes.")
            probs = torch.softmax(logits, dim=1)  # turn raw scores into probabilities that add up to 1
            confidence, index = probs.max(dim=1)
            for r, c, i in zip(batch_rows, confidence.tolist(), index.tolist()):
                out.append({"path": r["path"], "true_label": r["label"],
                            "predicted_label": classes[i], "confidence": f"{c:.4f}"})
            done = min(start + batch_size, len(rows))
            if done % (batch_size * 10) == 0 or done == len(rows):
                print(f"  {done}/{len(rows)} images")
    return out


def main():
    parser = argparse.ArgumentParser(description="Write validation and test prediction CSVs from a trained model.")
    parser.add_argument("--model", type=Path, required=True, help="model file: TorchScript, whole model, or state_dict")
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), help="network to rebuild when --model is a state_dict")
    parser.add_argument("--image-size", type=int, default=224, help="TODO(Amy): the size you trained at (default 224)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"], choices=list(OUTPUT_NAMES))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    classes = class_order()
    print(f"Classes (model output 0..{len(classes) - 1}): {classes}   device: {device}")

    model = load_model(args.model, args.arch, len(classes), device)
    # TODO(Amy): make this match your training transforms exactly (size, crop, normalization).
    preprocess = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])

    RESULTS_DIR.mkdir(exist_ok=True)
    for split_name in args.splits:
        rows = read_manifest(split_name)
        print(f"\n{split_name}: {len(rows)} images from data/splits/{split_name}.csv")
        unknown = sorted({r["label"] for r in rows} - set(classes))
        if unknown:
            raise SystemExit(f"Labels {unknown} in the manifest are not in the class order {classes}. "
                             f"Was the split built with a different num_classes?")
        predictions = predict_split(model, rows, classes, preprocess, device, args.batch_size)
        out_path = RESULTS_DIR / OUTPUT_NAMES[split_name]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "true_label", "predicted_label", "confidence"])
            writer.writeheader()
            writer.writerows(predictions)
        correct = sum(p["true_label"] == p["predicted_label"] for p in predictions)
        print(f"  wrote {out_path.relative_to(REPO_ROOT)}  (plain accuracy {correct / len(predictions):.3f})")

    print("\nNext: python src/eval/evaluate.py --tune results/val_predictions.csv --report results/test_predictions.csv")


if __name__ == "__main__":
    main()
