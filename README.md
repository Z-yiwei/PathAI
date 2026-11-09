# PathAI: Image-Prefix Clinical GPT

A multimodal extension of the supplied Aesclepius event GPT. A whole-slide pathology image is represented as a prefix token, allowing subsequent clinical events to attend to image information. A pathology-report prediction task provides auxiliary supervision.

The implementation directly imports `ToyAesclea` from the original project. It supports event-only, image-only, and combined inputs for patient-level outcome prediction.

## Interactive walkthrough

Open [showcase/index.html](showcase/index.html) in a browser to explore the data, tissue patches, image-prefix architecture, report supervision, and recorded learning curves. The page works offline with the repository files and requires no build step.

## Method

1. **Pretrain the original GPT.** Train next-event prediction on 40 synthetic Mayo patients and monitor 10 validation patients. The original two-layer, two-head, 64-dimensional model reaches a validation loss of approximately 0.328 after 300 steps.
2. **Encode the slide.** Extract 256 tissue patches from each H&E whole-slide image. A frozen Phikon encoder produces 768-dimensional patch features. Mean pooling and a trainable linear projection produce one 64-dimensional image token.
3. **Condition the GPT on the image.** Prepend the image token to the event embeddings and process the sequence with the original GPT's positional embeddings, Transformer blocks, and final normalization. The last valid position supplies the patient representation.
4. **Train the downstream tasks.** A classification head predicts `death_recorded`. A second head predicts a frozen PLIP report embedding. Mayo and TCGA next-event losses continue to train the shared GPT.

```text
WSI patches -> frozen Phikon -> mean pooling -> image projection
                                                      |
                                            [image token, events]
                                                      |
                                                 Original GPT
                                                      |
                                            Patient representation
                                               /             \
                                       Outcome head      Report head
                                                              |
                                                  Frozen PLIP report target
```

The joint-training objective is:

```text
outcome_BCE(TCGA)
+ 0.1 * [next_event_CE(Mayo) + next_event_CE(TCGA)]
+ 0.1 * [report_cosine_loss(Mayo) + report_cosine_loss(TCGA)]
```

GPT parameters and the added layers are fine-tuned together. Phikon and PLIP remain frozen. The three outcome-input modes retain the language-model and report auxiliary tasks; the combined mode also conditions TCGA next-event prediction on the image prefix.

## Data

The supplied dataset contains 50 Mayo patients and 10 TCGA patients. TCGA provides clinical events, OCR pathology reports, and H&E slides in SVS format.

The classification target is the presence of `DEATH` in the complete event record. TCGA model inputs stop at the first `PROC_RESECTION`, including that event. Outcome events are excluded from the input prefix.

Report preprocessing normalizes Unicode, line endings, and whitespace. Readability checks determine whether a report contributes to the auxiliary loss. Patients remain in the outcome task when their reports are unavailable or unreadable. Of 46 supplied reports, 45 pass these checks.

## Results

The selected development-validation runs below use six TCGA training patients and four validation patients. Validation BCE determines early stopping and run selection. Each seed controls the TCGA split, new parameter initialization, and fine-tuning batch draws; Mayo pretraining uses the same fixed initialization in both runs.

| Run | Outcome input | Accuracy | BCE |
|---|---|---:|---:|
| A | Events | 50% | 0.6476 |
| A | Image | 50% | 0.6934 |
| **A** | **Image + events** | **75%** | **0.6131** |
| B | Events | 50% | 0.6096 |
| B | Image | 50% | 0.6896 |
| **B** | **Image + events** | **75%** | **0.6064** |

### Run A

![Run A training curves](results/run_a/learning_curves.png)

[Metric comparison](results/run_a/comparison.png) · [Metrics](results/run_a/metrics.json)

### Run B

![Run B training curves](results/run_b/learning_curves.png)

[Metric comparison](results/run_b/comparison.png) · [Metrics](results/run_b/metrics.json)

Dots on the validation curves mark the selected checkpoints. All modes use a 100-step training limit, validation checks at step 1 and every 10 steps, and a 20-step early-stopping patience.

## Setup

Use Python 3.10 or newer. Keep the supplied original project alongside this repository:

```text
workspace/
├── aesclepius-jr/
│   ├── model.py
│   ├── train.py
│   └── data/
└── PathAI/
```

Run commands from the `PathAI` directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[encoders,dev]'
```

For a different original-project location, set `AESCLEPIUS_PROJECT` to its directory. Feature extraction requires CUDA-enabled PyTorch; training with cached features runs on CPU.

Download the two frozen encoders into their expected local directories:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download("owkin/phikon", local_dir="artifacts/pretrained/phikon")
snapshot_download("vinid/plip", local_dir="artifacts/pretrained/plip")
PY
```

## Prepare and encode

```bash
python -m pathai.data
CUDA_VISIBLE_DEVICES=0 python -m pathai.extract_images
CUDA_VISIBLE_DEVICES=0 python -m pathai.encode_reports
```

The image extractor uses the tissue coordinates in `configs/regions.json`. Prepared data and frozen features are written to `artifacts/`. Encoding output directories must be new; use `--out` to create an alternative cache.

## Train and evaluate

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -m pathai.train --seed "$EXPERIMENT_SEED" --out artifacts/run
python -m pathai.evaluate --run artifacts/run --out results/run
```

Set `EXPERIMENT_SEED` to the random seed for your run. Each training run includes the original Mayo pretraining followed by the three outcome-input modes. `--pretrained-run` can reuse the Mayo checkpoint from an existing run. Use a new directory for each training run.

A run saves the selected models, patient split, patient-level predictions, training history, and metrics. Evaluation exports the English figures and metrics shown above.

## Predict

The demo selects the checkpoint matching the requested outcome-input mode:

```bash
python -m pathai.predict --run artifacts/run \
  --patient TCGA-50-6591 --mode both
```

Use `--mode events`, `--mode image`, or `--mode both`. The demonstration patient must belong to the run's validation split. Predictions use a probability threshold of 0.5.

## Development

```bash
python -m unittest discover -s tests
black --check src tests
```

The tests cover original-GPT weight transfer, modality inputs, gradient propagation, causal padding, outcome separation, and report preprocessing.

## Layout

```text
src/pathai/
├── original.py         # Import the supplied GPT and training helpers
├── data.py             # Prepare events, labels, and report copies
├── report_quality.py   # Report formatting and readability
├── extract_images.py   # Read SVS regions and extract Phikon features
├── encode_reports.py   # Encode complete reports with PLIP
├── model.py            # Image prefix and downstream prediction heads
├── train.py            # Pretraining and multimodal fine-tuning
├── evaluate.py         # Metrics and training figures
├── predict.py          # Validation-patient prediction demo
└── common.py           # JSON output helpers
configs/regions.json    # Fixed tissue-sampling coordinates
results/                # Selected metrics and training curves
```

Original data, local documents, pretrained weights, feature caches, and checkpoints remain local and are excluded by `.gitignore`.
