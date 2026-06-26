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
The results are saved to: `outputs/ner_hipe/all_results.json`.

## Results

| Model | Overall P | Overall R | Overall F1 | Loss ↓ | loc F1 | org F1 | pers F1 | prod F1 | time F1 | Macro F1 | Weighted F1 | Epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | **0.7879** | **0.7422** | **0.7644** | 0.1448 | 0.86 | **0.60** | **0.69** | **0.71** | **0.37** | **0.65** | **0.76** | 5 |
| `stacked` | 0.7675 | 0.7288 | 0.7476 | **0.1307** | **0.87** | 0.57 | 0.65 | 0.65 | 0.24 | 0.60 | 0.74 | 5 |
| `time` | 0.7657 | 0.7173 | 0.7407 | 0.1445 | 0.85 | 0.57 | 0.66 | 0.57 | 0.31 | 0.59 | 0.73 | 5 |
