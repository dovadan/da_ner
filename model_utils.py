import datasets
import typing as tp

from transformers import (pipeline, 
        AutoModelForTokenClassification, AutoTokenizer, 
        BertForTokenClassification, BertTokenizer)

from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

from torch.nn.utils import clip_grad_norm_ # выполняет обрезку градиентов для избежания их взрыва при обучении
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

import seqeval
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

import importlib
import data_prep
importlib.reload(data_prep)


class NamedEntityPredictor:
    """
    Предсказывает NER-метки в батче, метки соответствуют id2label
    """
    def __init__(self,
                 model: BertForTokenClassification,
                 tokenizer: BertTokenizer,
                 id2label: tp.Optional[tp.Dict[int, str]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = model.config.id2label if id2label is None else id2label
        self.label2id = {v : k for k, v in self.id2label.items()}

    def predict(self, batch: tp.Dict[str, tp.Any]):
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(input_ids=batch["input_ids"],
                                      token_type_ids=batch["token_type_ids"],
                                      attention_mask=batch["attention_mask"],
                                      labels=batch["labels"],
                                      return_dict=True)
        indices = torch.argmax(model_output.logits, axis=2) # здесь индексы лейблов, на которых была обучена модель
        indices = indices.detach().cpu().numpy()

        label2id = self.label2id
        model_id2label = self.model.config.id2label
        index_map = {
            model_idx: label2id[label]
            for model_idx, label in model_id2label.items()
        }
        indices = np.vectorize(index_map.get)(indices)
        
        attention_mask = batch["attention_mask"].cpu().numpy()
        batch_size = len(batch["input_ids"])
        predicted_labels = []

        for i in range(batch_size):
            predicted_labels.append([self.id2label[id_] for id_ in indices[i][attention_mask[i]==1]])

        return {
            'predicted_labels': predicted_labels,
            'loss' : model_output.loss,
            'logits' : model_output.logits
        }

def get_sentence_embeddings(model, batch):
    """
    Выдает эмбеддинги предложений из датасетов по токену [CLF]
    """
    model.eval()
    with torch.no_grad():
        return model.bert(input_ids=batch["input_ids"],
                          token_type_ids=batch["token_type_ids"],
                          attention_mask=batch["attention_mask"],
                          return_dict=True)['last_hidden_state'].cpu().numpy()[:, 0] # 0 соответствует токену [CLF]


def train_eval_ner(model_ft, tokenizer, device, optimizer, num_epochs, lr_scheduler, loss_fn,
                   ft_train_dataset, ft_test_dataset, base_test_dataset, batch_size,
                   id2label):
    """
    Сначала оценивает Precision, Recall, F1 для задачи NER на модели без файн-тюна,
    делает файн-тюн с логированием микро усредненных метрик по эпохам, затем сравнивает эти метрики до и после всего файн-тюна. 

    Возвращает дообученную модель, репорты по классам до и после файн-тюна, метрики по эпохам
    """
    num_labels = 9

    # словари для перевода конлл-ных меток в модельные и обратно
    label2id = {v : k for k, v in id2label.items()}
    model_id2label = model_ft.config.id2label
    model_to_conll = {
        model_idx: label2id[label]
        for model_idx, label in model_id2label.items()
    }
    conll_to_model = {v : k for k, v in model_to_conll.items()}
        
    """
    Создаем даталоадеры
    """
    input_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    
    ft_train_dataloader = torch.utils.data.DataLoader(ft_train_dataset, 
                                                        batch_size=batch_size, 
                                                        collate_fn=data_prep.PadSequence(input_keys))
    ft_test_dataloader = torch.utils.data.DataLoader(ft_test_dataset, 
                                                        batch_size=batch_size, 
                                                        collate_fn=data_prep.PadSequence(input_keys))
    base_test_dataloader = torch.utils.data.DataLoader(base_test_dataset, 
                                                        batch_size=batch_size, 
                                                        collate_fn=data_prep.PadSequence(input_keys))
    
    model_ft.to(device)
    
    """
    Оценка модели без файн-тюна
    """
    ner_base = NamedEntityPredictor(model_ft, tokenizer, id2label)
    predicted_labels = {"ft_test": [], "base_test": []}

    print(f"Evaluating model without fine-tuning")
    for batch in tqdm(base_test_dataloader):
        predicted_labels["base_test"].extend(ner_base.predict(batch)["predicted_labels"]) # получаем conll-ные метки до файн-тюна
        
    for batch in tqdm(ft_test_dataloader):
        predicted_labels["ft_test"].extend(ner_base.predict(batch)["predicted_labels"])
    print(f"")

    # здесь в example["text_labels"] conll-ные метки
    base_report_before = classification_report(y_true=[list(example["text_labels"]) for example in base_test_dataset],
                                                         y_pred=predicted_labels["base_test"], output_dict=True)

    y_true_ft = [list(example["text_labels"]) for example in ft_test_dataset]
    y_pred_ft = predicted_labels["ft_test"]

    ft_report_before = classification_report(y_true=y_true_ft, y_pred=y_pred_ft, output_dict=True)

    ft_f1_before = f1_score(y_true=y_true_ft, y_pred=y_pred_ft)
    ft_pres_before = precision_score(y_true=y_true_ft, y_pred=y_pred_ft)
    ft_rec_before = recall_score(y_true=y_true_ft, y_pred=y_pred_ft)
    ft_acc_before = accuracy_score(y_true=y_true_ft, y_pred=y_pred_ft)
    ft_metrics_before = {'f1' : ft_f1_before, 'pr' : ft_pres_before, 'rec' : ft_rec_before,'acc' : ft_acc_before}

    df_base_before = pd.DataFrame(base_report_before).T.round(2)
    df_ft_before = pd.DataFrame(ft_report_before).T.round(2)
    
    comparison_before = pd.concat([df_base_before[['precision', 'recall', 'f1-score']], 
                            df_ft_before[['precision', 'recall', 'f1-score']]], axis=1)
    comparison_before.columns = ['pr_base', 'rec_base', 'f1_base', 'pr_ft', 'rec_ft', 'f1_ft']

    
    """
    Файн-тюн модели
    """
    print(f"Fine-tuning model")

    # создание словаря для логирования метрик
    ft_metrics_after = {'train_loss' : [], 'test_loss' : [], 'f1' : [], 'pr' : [], 'rec' : [], 'acc' : []}
    
    for epoch in range(num_epochs):
        # тренируем
        model_ft.train()
        loop = tqdm(ft_train_dataloader, desc=f"Epoch {epoch+1}")
        
        # в batch["labels"] conll-ные метки, а модели на вход нужно подавать её собственные
        for batch in loop:
            conll_labels = batch["labels"]
            # создаем mapping_array: в нем индекс - метка из conll, значение - модельная метка
            labels_model = torch.full_like(conll_labels, fill_value=-100)
            for conll_id, model_id in conll_to_model.items():
                labels_model[conll_labels == conll_id] = model_id

            outputs = model_ft(input_ids=batch["input_ids"],
                         token_type_ids=batch["token_type_ids"],
                         attention_mask=batch["attention_mask"],
                         labels=labels_model, return_dict=True)
            logits = outputs.logits.view(-1, num_labels) # логиты соответствуют собственным меткам
            labels_model = labels_model.view(-1)
            loss = loss_fn(logits, labels_model)
            loss.backward()
    
            # clip_grad_norm_(model_ft.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    
        # оцениваем
        model_ft.eval()
        train_losses=[]
        test_losses=[]
        with torch.no_grad():
            for batch in ft_test_dataloader:
                conll_labels = batch["labels"]
                # создаем mapping_array: в нем индекс - метка из conll, значение - модельная метка
                labels_model = torch.full_like(conll_labels, fill_value=-100)
                for conll_id, model_id in conll_to_model.items():
                    labels_model[conll_labels == conll_id] = model_id
                            
                outputs = model_ft(input_ids=batch["input_ids"],
                             token_type_ids=batch["token_type_ids"],
                             attention_mask=batch["attention_mask"],
                             labels=labels_model, return_dict=True)
                loss = outputs['loss'].item()
                test_losses.append(loss)
    
            for batch in ft_train_dataloader:
                conll_labels = batch["labels"]
                # создаем mapping_array: в нем индекс - метка из conll, значение - модельная метка
                labels_model = torch.full_like(conll_labels, fill_value=-100)
                for conll_id, model_id in conll_to_model.items():
                    labels_model[conll_labels == conll_id] = model_id
                
                outputs = model_ft(input_ids=batch["input_ids"],
                             token_type_ids=batch["token_type_ids"],
                             attention_mask=batch["attention_mask"],
                             labels=labels_model, return_dict=True)
                loss = outputs['loss'].item()
                train_losses.append(loss)
    
        train_losses=np.array(train_losses)
        test_losses=np.array(test_losses)

        ner = NamedEntityPredictor(model_ft, tokenizer, id2label) # возвращает conll-ные метки
        predicted_labels = {"ft_test": [], "base_test": []}
        
        for batch in ft_test_dataloader:
            predicted_labels["ft_test"].extend(ner.predict(batch)["predicted_labels"])
        y_true_ft = [list(example["text_labels"]) for example in ft_test_dataset]
        y_pred_ft = predicted_labels["ft_test"]
        ft_f1_after = f1_score(y_true=y_true_ft, y_pred=y_pred_ft)
    
        print(f"Epoch: {epoch+1}\t Train: {train_losses.mean()} \t Test: {test_losses.mean()}")
        print(f"F1: {ft_f1_after}")
        print(f"LR: {lr_scheduler.get_last_lr()}")
        # print(f"")
    
        """
        Оценка модели с файн-тюном
        """
        # print(f"Evaluating fine-tuned model")
        ner = NamedEntityPredictor(model_ft, tokenizer, id2label) # возвращает conll-ные метки
        predicted_labels = {"ft_test": [], "base_test": []}
        
        for batch in base_test_dataloader:
            predicted_labels["base_test"].extend(ner.predict(batch)["predicted_labels"])
            
        for batch in ft_test_dataloader:
            predicted_labels["ft_test"].extend(ner.predict(batch)["predicted_labels"])
    
        y_true_ft = [list(example["text_labels"]) for example in ft_test_dataset]
        y_pred_ft = predicted_labels["ft_test"]
        
        ft_f1_after = f1_score(y_true=y_true_ft, y_pred=y_pred_ft)
        ft_pres_after = precision_score(y_true=y_true_ft, y_pred=y_pred_ft)
        ft_rec_after = recall_score(y_true=y_true_ft, y_pred=y_pred_ft)
        ft_acc_after = accuracy_score(y_true=y_true_ft, y_pred=y_pred_ft)

        ft_metrics_after['train_loss'].append(train_losses.mean())
        ft_metrics_after['test_loss'].append(test_losses.mean())
        ft_metrics_after['f1'].append(ft_f1_after)
        ft_metrics_after['pr'].append(ft_pres_after)
        ft_metrics_after['rec'].append(ft_rec_after)
        ft_metrics_after['acc'].append(ft_acc_after)

        print(f"")


    ft_report_after = classification_report(y_true=y_true_ft, y_pred=y_pred_ft, output_dict=True)
    base_report_after = classification_report(y_true=[list(example["text_labels"]) for example in base_test_dataset],
                                                         y_pred=predicted_labels["base_test"], output_dict=True)
    
    df_base_after = pd.DataFrame(base_report_after).T.round(2)
    df_ft_after = pd.DataFrame(ft_report_after).T.round(2)
    
    comparison_after = pd.concat([df_base_after[['precision', 'recall', 'f1-score']], 
                            df_ft_after[['precision', 'recall', 'f1-score']]], axis=1)
    comparison_after.columns = ['pr_base', 'rec_base', 'f1_base', 'pr_ft', 'rec_ft', 'f1_ft']

    """
    Сравнение метрик до и после файн-тюна
    """
    diff = comparison_after-comparison_before
    
    # функция форматирования
    def format_metric(current, delta):
        sign = "+" if delta >= 0 else "-"
        return f"{current:.2f} ({sign}{abs(delta):.2f})"
    
    formatted = comparison_after.copy()
    
    for col in comparison_after.columns:
        formatted[col] = [
            format_metric(curr, d) 
            for curr, d in zip(comparison_after[col], diff[col])
        ]

    print(formatted)
    print(f"")

    ft_metrics_diff = {}

    for k in ft_metrics_before:
        before = ft_metrics_before[k]
        after = ft_metrics_after[k][-1] # берем последние измеренные метрики
        delta = after - before
        sign = "+" if delta >= 0 else "-"
        ft_metrics_diff[k] = f"{after:.2f} ({sign}{abs(delta):.2f})"

    print(f"{'metric':<6} | {'value':>15}")
    print("-" * 24)
    for k, v in ft_metrics_diff.items():
        print(f"{k:<6} | {v:>15}")
    print(f"")

    return model_ft, ft_report_before, ft_report_after, ft_metrics_before, ft_metrics_after


def train_eval_mlm(model_ft, tokenizer, device, optimizer, num_epochs, lr_scheduler, loss_fn,
                   pre_train_dataset, batch_size):
    """
    Сначала оценивает Precision, Recall, F1 для задачи NER на модели без файн-тюна,
    делает файн-тюн с логированием микро усредненных метрик по эпохам, затем сравнивает эти метрики до и после всего файн-тюна. 

    Возвращает дообученную модель, репорты по классам до и после файн-тюна, метрики по эпохам
    """
    
    """
    Создаем даталоадер
    """
    input_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    
    pre_train_dataloader = torch.utils.data.DataLoader(pre_train_dataset, 
                                                        batch_size=batch_size, 
                                                        collate_fn=data_prep.PadSequence(input_keys))

    model_ft.to(device)
    
    """
    Оценка модели без файн-тюна
    """
    ner_base = NamedEntityPredictor(model_ft, tokenizer, id2label)
    predicted_labels = {"ft_test": [], "base_test": []}

    print(f"Evaluating model without fine-tuning")
    for batch in tqdm(base_test_dataloader):
        predicted_labels["base_test"].extend(ner_base.predict(batch)["predicted_labels"])
        
    for batch in tqdm(ft_test_dataloader):
        predicted_labels["ft_test"].extend(ner_base.predict(batch)["predicted_labels"])
    print(f"")
    
    base_report_before = classification_report(y_true=[list(example["text_labels"]) for example in base_test_dataset],
                                                         y_pred=predicted_labels["base_test"], output_dict=True)

    y_true_ft = [list(example["text_labels"]) for example in ft_test_dataset]
    y_pred_ft = predicted_labels["ft_test"]

    ft_report_before = classification_report(y_true=y_true_ft, y_pred=y_pred_ft, output_dict=True)

    ft_f1_before = f1_score(y_true=y_true_ft, y_pred=y_pred_ft)
    ft_pres_before = precision_score(y_true=y_true_ft, y_pred=y_pred_ft)
    ft_rec_before = recall_score(y_true=y_true_ft, y_pred=y_pred_ft)
    ft_acc_before = accuracy_score(y_true=y_true_ft, y_pred=y_pred_ft)
    ft_metrics_before = {'f1' : ft_f1_before, 'pr' : ft_pres_before, 'rec' : ft_rec_before,'acc' : ft_acc_before}

    df_base_before = pd.DataFrame(base_report_before).T.round(2)
    df_ft_before = pd.DataFrame(ft_report_before).T.round(2)
    
    comparison_before = pd.concat([df_base_before[['precision', 'recall', 'f1-score']], 
                            df_ft_before[['precision', 'recall', 'f1-score']]], axis=1)
    comparison_before.columns = ['pr_base', 'rec_base', 'f1_base', 'pr_ft', 'rec_ft', 'f1_ft']

    
    """
    Файн-тюн модели
    """
    print(f"Fine-tuning model")

    # создание словаря для логирования метрик
    ft_metrics_after = {'train_loss' : [], 'test_loss' : [], 'f1' : [], 'pr' : [], 'rec' : [], 'acc' : []}
    
    for epoch in range(num_epochs):
        # тренируем
        model_ft.train()
        loop = tqdm(ft_train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            outputs = model_ft(input_ids=batch["input_ids"],
                         token_type_ids=batch["token_type_ids"],
                         attention_mask=batch["attention_mask"],
                         labels=batch["labels"], return_dict=True)
            logits = outputs.logits.view(-1, num_labels)
            labels = batch["labels"].view(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
    
            clip_grad_norm_(model_ft.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    
        # оцениваем
        model_ft.eval()
        train_losses=[]
        test_losses=[]
        with torch.no_grad():
            for batch in ft_test_dataloader:
                outputs = model_ft(input_ids=batch["input_ids"],
                             token_type_ids=batch["token_type_ids"],
                             attention_mask=batch["attention_mask"],
                             labels=batch["labels"], return_dict=True)
                loss = outputs['loss'].item()
                test_losses.append(loss)
    
            for batch in ft_train_dataloader:
                outputs = model_ft(input_ids=batch["input_ids"],
                             token_type_ids=batch["token_type_ids"],
                             attention_mask=batch["attention_mask"],
                             labels=batch["labels"], return_dict=True)
                loss = outputs['loss'].item()
                train_losses.append(loss)
    
        train_losses=np.array(train_losses)
        test_losses=np.array(test_losses)
    
        print(f"Epoch: {epoch+1}\t Train: {train_losses.mean()} \t Test: {test_losses.mean()}")
        # print(f"")
    
        """
        Оценка модели с файн-тюном
        """
        # print(f"Evaluating fine-tuned model")
        ner = NamedEntityPredictor(model_ft, tokenizer, id2label)
        predicted_labels = {"ft_test": [], "base_test": []}
        
        for batch in base_test_dataloader:
            predicted_labels["base_test"].extend(ner.predict(batch)["predicted_labels"])
            
        for batch in ft_test_dataloader:
            predicted_labels["ft_test"].extend(ner.predict(batch)["predicted_labels"])
    
        y_true_ft = [list(example["text_labels"]) for example in ft_test_dataset]
        y_pred_ft = predicted_labels["ft_test"]
        
        ft_f1_after = f1_score(y_true=y_true_ft, y_pred=y_pred_ft)
        ft_pres_after = precision_score(y_true=y_true_ft, y_pred=y_pred_ft)
        ft_rec_after = recall_score(y_true=y_true_ft, y_pred=y_pred_ft)
        ft_acc_after = accuracy_score(y_true=y_true_ft, y_pred=y_pred_ft)

        ft_metrics_after['train_loss'].append(train_losses.mean())
        ft_metrics_after['test_loss'].append(test_losses.mean())
        ft_metrics_after['f1'].append(ft_f1_after)
        ft_metrics_after['pr'].append(ft_pres_after)
        ft_metrics_after['rec'].append(ft_rec_after)
        ft_metrics_after['acc'].append(ft_acc_after)

        print(f"")


    ft_report_after = classification_report(y_true=y_true_ft, y_pred=y_pred_ft, output_dict=True)
    base_report_after = classification_report(y_true=[list(example["text_labels"]) for example in base_test_dataset],
                                                         y_pred=predicted_labels["base_test"], output_dict=True)
    
    df_base_after = pd.DataFrame(base_report_after).T.round(2)
    df_ft_after = pd.DataFrame(ft_report_after).T.round(2)
    
    comparison_after = pd.concat([df_base_after[['precision', 'recall', 'f1-score']], 
                            df_ft_after[['precision', 'recall', 'f1-score']]], axis=1)
    comparison_after.columns = ['pr_base', 'rec_base', 'f1_base', 'pr_ft', 'rec_ft', 'f1_ft']

    """
    Сравнение метрик до и после файн-тюна
    """
    diff = comparison_after-comparison_before
    
    # функция форматирования
    def format_metric(current, delta):
        sign = "+" if delta >= 0 else "-"
        return f"{current:.2f} ({sign}{abs(delta):.2f})"
    
    formatted = comparison_after.copy()
    
    for col in comparison_after.columns:
        formatted[col] = [
            format_metric(curr, d) 
            for curr, d in zip(comparison_after[col], diff[col])
        ]

    print(formatted)
    print(f"")

    ft_metrics_diff = {}

    for k in ft_metrics_before:
        before = ft_metrics_before[k]
        after = ft_metrics_after[k][-1] # берем последние измеренные метрики
        delta = after - before
        sign = "+" if delta >= 0 else "-"
        ft_metrics_diff[k] = f"{after:.2f} ({sign}{abs(delta):.2f})"

    print(f"{'metric':<6} | {'value':>15}")
    print("-" * 24)
    for k, v in ft_metrics_diff.items():
        print(f"{k:<6} | {v:>15}")
    print(f"")

    return model_ft, ft_report_before, ft_report_after, ft_metrics_before, ft_metrics_after


