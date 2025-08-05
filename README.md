# Domain Adaptation of BERT: CoNLL -> WNUT

This repository contains experiments on domain adaptation of a BERT model, initially fine-tuned on the CoNLL dataset (formal news text), to the WNUT dataset (noisy, user-generated content).

The main goal is to improve NER (Named Entity Recognition) performance in informal text domains using various data-centric adaptation techniques, including continued pretraining and task-specific fine-tuning.

---

## Project Structure

- `loss_plots/` – Training loss plots from continued pretraining of BERT on the MLM task using Reddit data  
- `data_prep.py` – Script for preprocessing CoNLL and WNUT datasets  
- `mlm.ipynb` – Notebook for continued pretraining of BERT on a Reddit-based dataset with the masked language modeling (MLM) objective  
- `model_utils.py` – Helper functions for model training, evaluation, and metric calculation  
- `ner.ipynb` – Notebook for domain adaptation of BERT-base to the WNUT dataset  
- `training_log.csv` – Training log from the MLM pretraining phase