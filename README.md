# Sugarcane Leaf Disease Classifier with YOLO26

A YOLO26 model that looks at a photo of a sugarcane leaf and names its disease, built for the
cane development staff of a sugar mill. When it is not sure, it says *refer to an agronomist*
instead of guessing.

**Industrial AI (IAI-26) · Week 8 assignment · Ghulam Mustafa**

Full report: [REPORT.pdf](REPORT.pdf) · Code and all results: [the notebook](sugarcane_leaf_disease_yolo26_classify.ipynb)

## The problem

A sugar mill buys its cane from hundreds of growers, and diseased cane means less sugar.
Field staff check leaves by eye, experts are few, and some diseases spread through seed cane.
This tool gives a first diagnosis from a phone photo, so staff can act sooner.

## Results

Tested on 877 photos the model never saw during training:

| Measure | Result |
| --- | --- |
| Top-1 accuracy | **99.54%** (873 of 877) |
| Macro-F1 (all 11 classes weighted equally) | 0.9948 |
| Missed disease (diseased leaf called Healthy) | 0 of 771 |
| False alarm (healthy leaf called diseased) | 0 of 60 |
| Weakest class | Smut, F1 0.980 (also the rarest: 126 training photos) |

**These numbers are for this public dataset, not for real fields.** Most photos are cut leaves
on a tiled floor, and red rot is not one of the classes. See *Limitations* below.

## What I found in the data

1. **Duplicate photos with contradicting labels.** 362 pairs of identical photos carry two
   different disease labels (most often *Pokkah Boeng* and *Smut*). I removed those 614 photos.
2. **A normal random split would have leaked.** 403 of 921 test photos (44%) would have had a
   copy in the training set. I used a *grouped* split instead, so no copy is on both sides.

## The application

Both tools use a confidence threshold of **0.9**. Below it, the photo goes to a person.

- **Field-visit survey report:** a folder of leaf photos becomes a CSV file with the condition,
  confidence and action for each photo ([example](sugarcane_project/field_visit_report.csv)).
- **Web app (Gradio):** upload one photo and see the three most likely conditions and the action.

## What is in this repository

| Path | What it is |
| --- | --- |
| `sugarcane_leaf_disease_yolo26_classify.ipynb` | The whole project: data, leak check, training, evaluation, application. All outputs are saved in it |
| `REPORT.pdf` | The assignment report in plain language (13 pages) |
| `sugarcane_project/runs/sugarcane_cls/weights/best.pt` | The trained model (3.2 MB) |
| `sugarcane_project/runs/sugarcane_cls/` | Training charts: `results.png`, confusion matrices, `results.csv` |
| `sugarcane_project/runs/summary.json` | Every key number in one file |
| `sugarcane_project/sugarcane_results.zip` | Model, tables, charts and survey report in one download |
| `sugarcane_project/field_visit_report.csv` | Example output of the survey tool |

The 6,748 photos are **not** stored here (1.2 GB). The notebook downloads them from Hugging Face.

## How to run it

**Google Colab (easiest, free GPU):**

1. Open the notebook in [Colab](https://colab.research.google.com) (*File > Upload notebook*).
2. Choose *Runtime > Change runtime type > T4 GPU*.
3. Choose *Runtime > Run all*. Allow Google Drive access when asked: the photos and the
   trained model are saved there. Takes about 15 to 25 minutes.

**On your own PC (Python 3.10 or newer):**

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install ultralytics==8.4.159 datasets scikit-learn pandas gradio jupyter
jupyter notebook sugarcane_leaf_disease_yolo26_classify.ipynb
```

Without a GPU, training takes about 2 hours. If `best.pt` is already present, the notebook reuses
it and the run takes about 10 minutes (set `RETRAIN = True` to train again).
Set `QUICK_RUN = True` to test the whole pipeline in a few minutes on a small subset.

**Use the trained model on any PC (no GPU needed):**

```python
from ultralytics import YOLO

model = YOLO("sugarcane_project/runs/sugarcane_cls/weights/best.pt")
result = model.predict("leaf_photo.jpg")[0]
print(result.names[result.probs.top1], float(result.probs.top1conf))
```

## Limitations

- The photos are not from our growers' fields; real-field accuracy must still be measured.
- Red rot is not one of the 11 classes, so the model cannot report it.
- Different photos of the same leaf can still be on both sides of the split, which may make
  the test easier than real use.
- One training run with one random seed.
- The advice text in the app is general guidance, to be checked by cane development staff.
  It is not agronomic advice.

## Credits and licence

- **Dataset:** Thite, Suryawanshi, Patil, Chumchu (2024),
  [*Sugarcane leaf dataset: a dataset for disease detection and classification for machine learning applications*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10964057/),
  *Data in Brief* 53, 110268. Used through the
  [Project-AgML copy on Hugging Face](https://huggingface.co/datasets/Project-AgML/sugarcane_leaf_disease_classification),
  licensed CC BY 4.0. The charts in this repository contain photos from this dataset.
- **Model framework:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (YOLO26), AGPL-3.0.
- **This project** (code and trained model) is released under the [GNU AGPL-3.0](LICENSE),
  because it is built on and trained with Ultralytics YOLO.
