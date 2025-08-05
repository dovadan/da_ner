import datasets
import typing as tp

def convert_label_sequence(example: tp.Dict[str, tp.Any], label_mapping: tp.Dict[str, str]) -> tp.Dict[str, tp.Any]:
    """
    Переделывает метки для NEP по шаблону label_mapping
    """
    converted_example = dict(**example)
    converted_example['ner_tags'] = [label_mapping[label] for label in example['ner_tags']]
    return converted_example


from transformers import (pipeline, 
        AutoModelForTokenClassification, AutoTokenizer, 
        BertForTokenClassification, BertTokenizer)

def tokenize_and_preserve_tags(example: tp.Dict[str, tp.Any],
                               tokenizer: BertTokenizer,
                               label2id: tp.Dict[str, int],
                               tokenizer_params={}) -> tp.Dict[str, tp.Any]:

    """
    Функция для расширения тегов на все токены
    """

    encoded = tokenizer(example["tokens"], is_split_into_words=True, **tokenizer_params)
    encoded.update(example)

    encoded['labels'] = []
    encoded['text_labels'] = []
    original_tags = encoded["ner_tags"]

    id2label = {v : k for k, v in label2id.items()}

    prev_idx = None
    for word_idx in encoded.word_ids():
        if word_idx is None:
            encoded['labels'].append(0)
            encoded['text_labels'].append("O")
        else:
            tag = int(original_tags[word_idx]) # conll tag
            text_tag = id2label[tag]
            
            if word_idx != prev_idx: # 1) prev_idx==None 2) prev_idx!=None
                encoded['labels'].append(tag)
                encoded['text_labels'].append(text_tag)
            else:  # word_idx == prev_idx (внутри того же слова)
                if text_tag.startswith("B-") or text_tag.startswith("I-"):
                    suffix = text_tag[2:]
                    continued_tag = "I-" + suffix
                else:
                    continued_tag = "O"
                encoded['text_labels'].append(continued_tag)
                encoded['labels'].append(label2id.get(continued_tag, label2id["O"]))
        prev_idx = word_idx

    assert len(encoded["labels"]) == len(encoded["input_ids"])
    return encoded


from collections import defaultdict
import torch


class PadSequence:
    """
    Паддит предложения и делает батчи, ставит -100 в labels для паддинга (по умолчанию в ce-lossе игнорируются токены со значением -100)
    """
    def __init__(self, padded_columns, label_column="labels", pad_token_id=0, device='cuda'):
        self.padded_columns = set(padded_columns)
        self.label_column = label_column
        self.pad_token_id = pad_token_id
        self.device = device

    def __call__(self, batch):
        padded_batch = defaultdict(list)
        for example in batch:
            for key, tensor in example.items():
                padded_batch[key].append(tensor)

        for key, vals in padded_batch.items():
            if key in self.padded_columns:
                padded = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True)

                if key == self.label_column:
                    lengths = [len(t) for t in vals]
                    max_len = padded.size(1)
                    mask = torch.zeros_like(padded, dtype=torch.bool)

                    for i, l in enumerate(lengths):
                        mask[i, :l] = True # настоящие токены (не паддиннг)

                    padded[~mask] = -100

                padded_batch[key] = padded.to(self.device)

            # else:
            #     padded_batch[key] = torch.stack(val).to(self.device)

        return padded_batch
                    


        