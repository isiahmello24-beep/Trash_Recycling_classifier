# Evaluation: from model file to confusion matrix and threshold chart

Two commands take a trained model to the final numbers. The first writes a
predictions CSV per split, the second scores them and picks the "I'm not sure"
cutoff the right way: tuned on validation, reported on test.

## Before you start

- The splits are built: `python src/data/build_splits.py` ends with
  `Split version: d010b6a  manifest_sha256: 4d578382b730`. Both commands below
  read `data/splits/validation.csv` and `data/splits/test.csv`, so they score
  exactly that split.
- `pip install -r requirements.txt` plus PyTorch and torchvision for your
  computer (see https://pytorch.org/get-started/locally/).
- A model file. Three forms work: a TorchScript file, a whole model saved with
  `torch.save(model, path)`, or a state_dict saved with
  `torch.save(model.state_dict(), path)`. A state_dict also needs `--arch`.

## The 2 commands

```bash
python src/evaluation/export_predictions.py --model models/best_model.pth --arch mobilenet_v3_small
```

```bash
python src/eval/evaluate.py --tune results/val_predictions.csv --report results/test_predictions.csv
```

| Command | What it does | Writes |
|---|---|---|
| `export_predictions.py` | Runs the model over every validation and test image and records the true label, the predicted label and the model's confidence. | `results/val_predictions.csv`, `results/test_predictions.csv` |
| `evaluate.py` | Prints accuracy, macro F1 and per-class scores, sweeps cutoffs on the validation file, picks the lowest cutoff reaching 90% accuracy on answered items (`--target` to change), and applies it once to the test file. | `results/confusion_matrix.png`, `results/threshold_sweep.csv`, `results/threshold_sweep.png` |

The CSVs and charts are git-ignored. Share them in the PR or Discord.

## Settings to confirm (marked `TODO(Amy)` in the script)

These are guesses until the training code is in the repo:

| Setting | Current guess | Flag |
|---|---|---|
| Framework | PyTorch + torchvision | |
| Network | `mobilenet_v3_small` | `--arch` (only for a state_dict) |
| Image size | 224 x 224, resized without cropping | `--image-size` |
| Normalization | ImageNet mean and std | edit `NORMALIZE_MEAN` / `NORMALIZE_STD` |
| Class order | active classes from `configs/label_map.yaml`, alphabetical: compost, hazardous, recycling, trash | matches `ImageFolder` and `results/class_weights.json` |

If the training used different transforms, the predictions will be worse than
the model really is, so match them exactly.

## Reading the result

- **Macro F1 on the test file** is the headline number. Accuracy is inflated
  by the recycling class (68% of train).
- **The chosen cutoff** comes from validation only. The test numbers "with
  cutoff applied" are what the Pi demo would show: how many items get an
  answer, how accurate those answers are, and how many wrong answers became
  "I'm not sure".
- **Passing one file** to `evaluate.py` still works but prints a warning: a
  cutoff tuned and reported on the same items looks better than it is.
