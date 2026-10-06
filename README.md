# CPRG-MoE

CPRG-MoE classifies emotions in conversational utterances and extracts the utterances that cause those emotions.

This repository provides the original CPRG-MoE implementation with a prebuilt Docker image. Configure your server’s GPU and data paths, then use Docker Compose to run training and evaluation.

- **Paper:** [Enhancing Emotion-Cause Pair Extraction in Conversation with Contextual Information](https://doi.org/10.1145/3748522.3779710)
- **Original implementation:** [JaehyeokLee-119/CPRG-MoE](https://github.com/JaehyeokLee-119/CPRG-MoE)
- **Docker image:** `ghcr.io/bbobburi/cprg-moe:1.0.0`

## 1. Server Requirements

The execution server must provide:

- Linux AMD64
- An NVIDIA GPU
- An NVIDIA driver compatible with the CUDA 11.8 runtime
- Docker Engine
- Docker Compose v2
- NVIDIA Container Toolkit configured for GPU access in Docker

Internet access is required to download the image and pretrained BERT weights on the first run.

Python, PyTorch, model dependencies, and the CPRG-MoE code are included in the image. You do not need to install them in the server’s Python environment.

## 2. Quick Start

### Clone the repository

```bash
git clone https://github.com/bbobburi/CPRG-MoE.git
cd CPRG-MoE
```

### Create your configuration file

```bash
cp .env.example .env
nano .env
```

Set your GPU, dataset paths, and output directory in `.env`. See the next section for configuration details.

### Create the output directory

Create the directory specified by `OUTPUT_DIR` in `.env`.

```bash
mkdir -p /absolute/path/to/your/results
```

Replace the example path with your actual output directory. The execution user must have write permission for this directory.

### Download the image and run

Run the following commands from the directory containing `compose.yaml` and `.env`:

```bash
docker compose config --quiet
docker compose pull train
docker compose run --rm train
```

After training, the best validation checkpoint is automatically evaluated on the test set.

## 3. Configuration — `.env`

| Variable | Description |
|---|---|
| `CPRG_IMAGE` | Container image address. You can keep the supplied default. |
| `GPU_ID` | Physical GPU index on your server. Check it with `nvidia-smi`. |
| `LOCAL_UID` | Execution user’s UID, returned by `id -u`. |
| `LOCAL_GID` | Execution user’s GID, returned by `id -g`. |
| `DATA_DIR` | Absolute path to the dataset directory on your server. |
| `OUTPUT_DIR` | Absolute path for checkpoints and logs. Use a separate directory for each run. |
| `TRAIN_FILE` | Training file path within `DATA_DIR`. |
| `VALID_FILE` | Validation file path within `DATA_DIR`. |
| `TEST_FILE` | Test file path within `DATA_DIR`. |
| `DATASET_TYPE` | Emotion-label scheme. Currently supports `ConvECPE` and `RECCON`. |
| `DATA_LABEL` | A name used to identify the run in logs. |
| `EPOCHS` | Number of training epochs. Default: `20`. |
| `BATCH_SIZE` | Training batch size. Default: `4`. |
| `NUM_WORKER` | Additional CPU processes for loading data. Default: `0`. |
| `CKPT_NAME` | Checkpoint filename without the `.ckpt` extension. |

For example, if your data is stored as follows:

```text
/home/user/my_data/
└── split/
    ├── train.json
    ├── valid.json
    └── test.json
```

Use:

```dotenv
DATA_DIR=/home/user/my_data
OUTPUT_DIR=/home/user/results/experiment_01

TRAIN_FILE=split/train.json
VALID_FILE=split/valid.json
TEST_FILE=split/test.json
```

Do not repeat the full server path in `TRAIN_FILE`, `VALID_FILE`, or `TEST_FILE`. Specify each file’s location within `DATA_DIR`.

`DATA_DIR` is mounted read-only at `/data` inside the container. `OUTPUT_DIR` is mounted at `/outputs`. Checkpoints and logs are saved on your server and remain available after the container exits.

## 4. Input Data Format

Prepare separate JSON files for training, validation, and testing. All three files must follow the input structure required by the current CPRG-MoE implementation.

### Input JSON example

The following example is an excerpt from dialogue `tr_4466` in a **RECCON JSON file prepared for CPRG-MoE input**. It shows the first two utterances and selected fields, representing the input structure **before model preprocessing**.

```json
{
  "tr_4466": [
    [
      {
        "turn": 1,
        "speaker": "A",
        "utterance": "Hey , you wanna see a movie tomorrow ?",
        "emotion": "happiness",
        "expanded emotion cause evidence": [1]
      },
      {
        "turn": 2,
        "speaker": "B",
        "utterance": "Sounds like a good plan . What do you want to see ?",
        "emotion": "happiness",
        "expanded emotion cause evidence": [1]
      }
    ]
  ]
}
```

Each dialogue contains an ordered list of utterances. Wrap this utterance list in an additional list, as shown above.

| Field | Description |
|---|---|
| `Dialogue ID` | A unique identifier for the dialogue. |
| `turn` | Utterance number within the dialogue, starting at 1 in the example. |
| `speaker` | Speaker identifier. The current implementation assumes two speakers: `A` and `B`. |
| `utterance` | Utterance text. |
| `emotion` | Emotion label for the utterance. |
| `expanded emotion cause evidence` | A list of utterance indices annotated as causes of the target utterance’s emotion. |

### Emotion and cause annotations

The second utterance in the example has:

```json
"emotion": "happiness",
"expanded emotion cause evidence": [1]
```

This means that **the second utterance expresses happiness, and the first utterance is its annotated cause**.

Cause indices are **1-based** and refer to positions in the ordered utterance list. An utterance may reference itself as a cause. Multiple causes are represented by multiple indices in the list.

Emotion and cause annotations are required for training and evaluation.

### Supported emotion-label schemes

`DATASET_TYPE` determines the emotion-label mapping and the number of emotion classes.

| DATASET_TYPE | Emotion labels |
|---|---|
| `ConvECPE` | `happy`, `sad`, `neutral`, `angry`, `excited`, `frustrated` |
| `RECCON` | `anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise`, `neutral` |

The original implementation applies the following mappings for RECCON:

| Input labels | Emotion class |
|---|---|
| `angry`, `anger` | anger |
| `disgust` | disgust |
| `fear` | fear |
| `happy`, `happines`, `happiness`, `excited` | happiness |
| `sad`, `sadness`, `frustrated` | sadness |
| `surprise`, `surprised` | surprise |
| `neutral` | neutral |

ConvECPE treats `excited` and `frustrated` as separate emotion classes. See the [preprocessing code](https://github.com/bbobburi/CPRG-MoE/blob/main/module/preprocessing.py) for the complete mapping.

A new dataset can be used if it follows the supported input structure and emotion-label scheme. Changing the value of `DATASET_TYPE` alone does not add support for different emotion categories.

> 📩 **Using a new dataset**
> If you provide a data sample or a description of its structure, along with the emotion-label list, we will provide code to convert your data into the CPRG-MoE input format. Where necessary, we will adapt the emotion-label mapping and model output classes and provide an updated image.

Your data is supplied from your execution server and is not included in the image.

## 5. Training and Evaluation

```bash
docker compose run --rm train
```

Following the original public implementation, this command:

1. Loads the data and BERT model.
2. Trains for the configured number of epochs.
3. Evaluates the validation set after each epoch.
4. **Saves the checkpoint with the highest validation PECA F1.**
5. Loads that checkpoint and evaluates it on the test set.

The default settings follow the paper’s training configuration: **20 epochs, batch size 4, learning rate `5e-5`, maximum sequence length 128, and dropout 0.5**.

The `train` service performs training followed by test evaluation. A separate evaluation-only Compose service is not currently provided.

## 6. Outputs and Evaluation Metrics

Outputs are saved under the directory specified by `OUTPUT_DIR`:

```text
OUTPUT_DIR/
├── model/
├── logs/
├── lightning_logs/
└── hf_cache/
```

| Directory | Contents |
|---|---|
| `model/` | Best validation checkpoint |
| `logs/` | Training, validation, test, and sample logs |
| `lightning_logs/` | CSV metrics and run configuration |
| `hf_cache/` | Downloaded BERT-related files |

The best checkpoint is saved to `OUTPUT_DIR/model/CKPT_NAME.ckpt`.

Final test results are displayed in the terminal and recorded in the test log.

| Metric | Description |
|---|---|
| EBC | ROC-AUC for binary emotion classification |
| EMC | Weighted F1 for emotion multiclass classification |
| ECPD | F1 for emotion–cause pair detection |
| PECA | F1 for joint assessment of emotion–cause pairs and their emotion labels |

## 7. Attribution and License

This repository is based on the [original CPRG-MoE implementation](https://github.com/JaehyeokLee-119/CPRG-MoE). Code usage is governed by the repository’s `LICENSE.txt`.

The original study uses:

- [RECCON](https://github.com/declare-lab/RECCON) — [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- [ConvECPE](https://github.com/Maxwe11y/JointEC/tree/main/Dataset)

When using these datasets, follow their respective usage terms. User-provided data remains subject to its own usage policies.

## 8. Citation

```bibtex
@inproceedings{10.1145/3748522.3779710,
author = {Lee, Jaehyeok and Bak, JinYeong},
title = {Enhancing Emotion-Cause Pair Extraction in Conversation with Contextual Information},
year = {2026},
isbn = {9798400722943},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3748522.3779710},
doi = {10.1145/3748522.3779710},
abstract = {To generate more emotionally human-like responses, it is important for the chatbot to not only infer the emotions of interlocutors, but also consider the underlying causes of these emotions. Emotion-cause pair extraction in conversation (ECPEC) is a task that aims to identify emotions and their corresponding causes as pairs in a conversation. The use of conversational context is important in predicting emotions and their causes for the ECPEC task. However, prior studies have been limited in fully utilizing contextual information. In this study, we propose the contextualized pair-relationship guided mixture-of-experts (CPRG-MoE) model for ECPEC, which incorporates contextual information along with dialogue features such as speaker information. Furthermore, prior evaluations of ECPEC have focused only on detecting emotion-cause pairs without considering the type of emotion evoked by each cause. This limited focus leads to an insufficient understanding of the dependency relationship between emotions and their causes. Therefore, we also propose a novel evaluation metric, emotion-cause pair emotion combined assessment (PECA), to jointly assess emotion-cause pairs and their corresponding emotions. Our proposed CPRG-MoE model outperforms the baselines in terms of ECPEC on two datasets, RECCON and ConvECPE, utilizing both existing metrics and the novel PECA metric1.},
booktitle = {Proceedings of the 41st ACM/SIGAPP Symposium on Applied Computing},
pages = {883–891},
numpages = {9},
keywords = {conversations, emotion analysis, emotion-cause pair extraction, dialogue systems, multi-task learning},
location = {Grand Hotel Palace, Thessaloniki, Greece},
series = {SAC '26}
}
```
