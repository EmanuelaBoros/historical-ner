# Historical NER
Experiments on named entity recognition for historical documents using HIPE-style data and historical BERT models.
This repository trains and evaluates token-classification models for historical NER, focusing on the `NE-COARSE-LIT` column. The goal is to compare a standard historical BERT baseline with two architecture variants that add extra Transformer layers and temporal information.
## Task
The task is sequence labelling for named entity recognition.
Input data follows the HIPE TSV format, for example:
```text
TOKEN       NE-COARSE-LIT   NE-COARSE-METO
Le          O               O
public      O               O
Charlotte   B-pers          O
née         I-pers          O
Bourgoin    I-pers          O
```

Only the following column is used for training and evaluation: `NE-COARSE-LIT`.

## Data

The experiments use HIPE-style historical NER data stored under: `data/`. The script recursively reads all .tsv files in `data/`. The test set is fixed to: `data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-test-fr.tsv`.  All other TSV files found under `data/` are used for training.

### Label normalization

The HIPE data contains labels from slightly different annotation schemes. Before training, labels are normalized to reduce inconsistencies such as:
```text
B-PER       -> B-pers
I-PER       -> I-pers
B-LOC       -> B-loc
I-ORG       -> I-org
B-STREET    -> B-loc
B-BUILDING  -> B-loc
B-HumanProd -> B-prod
B-object    -> B-prod
B-work      -> B-prod
B-date      -> B-time
```
After normalization, the main entity types are:
```text
loc
org
pers
prod
time
```

## Models

The experiments compare three models.

| Model | Base encoder | Additional layers | Temporal encoding | Description |
|---|---|---:|---|---|
| `baseline` | `dbmdz/bert-base-historic-multilingual-cased` | 0 | No | Historical BERT with a token-classification head |
| `stacked` | `dbmdz/bert-base-historic-multilingual-cased` | 2 | No | Historical BERT followed by two additional Transformer encoder layers |
| `time` | `dbmdz/bert-base-historic-multilingual-cased` | 2 | Yes | Historical BERT with document year encoding added before the extra Transformer layers |

The temporal model encodes the document year from the HIPE metadata: `# hipe2022:date = 1798-01-04`.
The year is normalized as: `(year - 1800) / 100.0` and projected into the hidden space with a small MLP.

## Training

Install dependencies:

```bash
pip install torch transformers datasets evaluate seqeval scikit-learn accelerate safetensors
```

Run all three models:
```bash
python train_ner_hipe.py \
  --data_dir data \
  --test_file data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-test-fr.tsv \
  --variant all \
  --epochs 5 \
  --batch_size 8 \
  --max_length 256 \
  --fp16
```
Run only one variant:
```bash
python train_ner_hipe.py \
  --data_dir data \
  --test_file data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-test-fr.tsv \
  --variant baseline \
  --epochs 5 \
  --batch_size 8 \
  --max_length 256 \
  --fp16
```
Available variants:
```text
baseline
stacked
time
all
```

## Results

| Model | Entity type | Precision | Recall | F1 | Loss | Support | Epochs |
|---|---|---:|---:|---:|---:|---:|---:|
| `baseline` | **overall / micro avg** | 0.7879 | 0.7422 | 0.7644 | 0.1448 | 1567 | 5 |
| `baseline` | `loc` | 0.86 | 0.86 | 0.86 | 0.1448 | 797 | 5 |
| `baseline` | `org` | 0.71 | 0.52 | 0.60 | 0.1448 | 128 | 5 |
| `baseline` | `pers` | 0.70 | 0.68 | 0.69 | 0.1448 | 530 | 5 |
| `baseline` | `prod` | 0.84 | 0.61 | 0.71 | 0.1448 | 59 | 5 |
| `baseline` | `time` | 0.54 | 0.28 | 0.37 | 0.1448 | 53 | 5 |
| `baseline` | **macro avg** | 0.73 | 0.59 | 0.65 | 0.1448 | 1567 | 5 |
| `baseline` | **weighted avg** | 0.78 | 0.74 | 0.76 | 0.1448 | 1567 | 5 |
| `stacked` | **overall / micro avg** | TBD | TBD | TBD | TBD | TBD | 5 |
| `time` | **overall / micro avg** | TBD | TBD | TBD | TBD | TBD | 5 |


The combined results are saved to: `outputs/ner_hipe/all_results.json`.