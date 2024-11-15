# Reproducibility Study for "Beneath the Surface: Unveiling Harmful Memes with Multimodal Reasoning Distilled from Large Language Models"
Modification Official PyTorch implementation for the paper - **Beneath the Surface: Unveiling Harmful Memes with Multimodal Reasoning Distilled from Large Language Models**.

(**EMNLP 2023**: *The 2023 Conference on Empirical Methods in Natural Language Processing (Findings), Dec 2023, Singapore*.) [[`paper`](https://aclanthology.org/2023.findings-emnlp.611/)]

# Results

## Reproduced Results

In our study, we successfully reproduced the results from the original paper using the provided methodology. The results of our reproduced experiments were very close to the published results in terms of Accuracy and Macro-F1 scores, with minor discrepancies attributed to the following factors:

- **Random Initialization**: Variability in model weights due to the random initialization process.
- **Preprocessing Variations**: Small differences in tokenization or image augmentation techniques.
- **Environment Differences**: Variations in hardware (GPU) and software (CUDA) configurations.

| Dataset | Metric      | Published | Reproduced |
|---------|-------------|-----------|------------|
| Harm-C  | Accuracy    | 86.16     | 85.90      |
|         | Macro-F1    | 85.43     | 85.10      |
| Harm-P  | Accuracy    | 89.58     | 89.40      |
|         | Macro-F1    | 89.57     | 89.20      |
| FHM     | Accuracy    | 75.40     | 74.80      |
|         | Macro-F1    | 75.10     | 74.50      |

## Challenges and Observations

- **Preprocessing Consistency**: Minor preprocessing discrepancies during text-tokenization and image augmentation caused slight variations in results. We recommend further standardizing preprocessing steps for improved reproducibility.
- **Computational Constraints**: Limited GPU access and variability in hardware configurations impacted training time and some hyperparameter tuning.
- **Environment Setup**: Ensuring consistent setup across machines can be challenging, especially with distributed setups.

## Install

```bash
# Create a conda environment and activate it
conda create -n meme python=3.8
conda activate meme

# Install dependencies
pip install -r requirements.txt
```

## Data

Please refer to [Data](https://github.com/Bhavana0810-dev/HarmfulMemeDetection/tree/main/Data).

## Training
- To train the model using the pretrained LLMs and fine-tune it for meme classification:

```bash
export DATA="/path/to/data/folder"
export LOG="/path/to/save/ckpts/name"

rm -rf $LOG
mkdir $LOG

CUDA_VISIBLE_DEVICES=0 python run.py with data_root=$DATA \
    num_gpus=1 num_nodes=1 task_train per_gpu_batchsize=32 batch_size=32 \
    clip32_base224 text_t5_base image_size=224 vit_randaug mode="rationale" \
    log_dir=$LOG precision=32 max_epoch=10 learning_rate=5e-5
```

- Learn from Labels
```bash
export DATA="/path/to/data/folder"
export LOG="/path/to/save/ckpts/name"

rm -rf $LOG
mkdir $LOG

CUDA_VISIBLE_DEVICES=0 python run.py with data_root=$DATA \
    num_gpus=1 num_nodes=1 task_train per_gpu_batchsize=32 batch_size=32 \
    clip32_base224 text_t5_base image_size=224 vit_randaug mode="label" \
    log_dir=$LOG precision=32 max_epoch=30 learning_rate=5e-5 \
    load_path="/path/to/distill_LLMs.ckpt"
```

## Inference

```bash
export DATA="/path/to/data/folder"
export LOG="/path/to/log/folder"

CUDA_VISIBLE_DEVICES=0 python run.py with data_root=$DATA \
    num_gpus=1 num_nodes=1 task_train per_gpu_batchsize=1 batch_size=1 \
    clip32_base224 text_t5_base image_size=224 vit_randaug \
    log_dir=$LOG precision=32 test_only=True \
    load_path="/path/to/label_learn.ckpt" \
    out_path="/path/to/save/label_pred.json"
```
Then, you can use the `/path/to/save/label_pred.json` and the gold labels to get the scores.

## Citation

```
@inproceedings{lin-etal-2023-beneath,
    title = "Beneath the Surface: Unveiling Harmful Memes with Multimodal Reasoning Distilled from Large Language Models",
    author = "Lin, Hongzhan  and
      Luo, Ziyang  and
      Ma, Jing  and
      Chen, Long",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.611",
    doi = "10.18653/v1/2023.findings-emnlp.611",
    pages = "9114--9128",
}
```
## Workload Clarification
```
The reproduction study was a collaborative effort,
with each member contributing to key aspects:
• Dataset Preparation (Bhavana, Priya): Ensured
datasets matched the original study, replicat-
ing preprocessing steps like tokenization, im-
age resizing, and splits.
• Model Implementation (Bhavana, Priya): Reproduced
the training pipeline, fine-tuned hyperparam-
eters, and validated performance metrics.
• Robustness Evaluation (Bhavana): Designed per-
turbation tests, analyzed model resilience un-
der adversarial and noisy conditions.
• Error Analysis (Priya): Investigated failure
cases, compared predictions, and identified
root causes for discrepancies.
```
## Acknowledgements

The code is based on [HKBUNLP](https://github.com/HKBUNLP/Mr.Harm-EMNLP2023)
