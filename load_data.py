import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import os
from collections import defaultdict

import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


os.environ['CURL_CA_BUNDLE'] = ''

def get_corpus(file):
    with open(file, 'r', encoding="utf-8") as f:
        item_list = [json.loads(line) for line in f.readlines()]
        random.shuffle(item_list)
    f.close()
    return item_list



class NLPCustomDataset(Dataset):
    def __init__(self, data,neg_sample_num):
        self.data = data
        self.label_to_samples = defaultdict(list)
        self.neg_sample_num = neg_sample_num


        for entry in self.data:
            label = entry['label']
            self.label_to_samples[label].append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        set_seed(1)
        current_example = self.data[index]
        label = current_example['label']

        # Randomly select a positive sample with the same label
        positive_samples = self.label_to_samples[label]
        if len(positive_samples) > 1:
            positive_samples.remove(current_example)  # Remove the current example
        positive_sample = random.choice(positive_samples)['text_pos']

        # Sample neg_sample_num negative samples with different labels
        negative_samples = []
        negative_labels = [] 
        for i in range(self.neg_sample_num):
            #random.seed(i)
            negative_label = random.choice(list(self.label_to_samples.keys()))
            while negative_label == label:
                negative_label = random.choice(list(self.label_to_samples.keys()))
            negative_sample = random.choice(self.label_to_samples[negative_label])['text_pos']
            negative_samples.append(negative_sample)
            negative_labels.append(negative_label)

        return {
            'text_gt': current_example['text_pos'],
            'text_pos': positive_sample,
            'text_neg': negative_samples,
            'label': label,
            'negative_labels': negative_labels,#[negative_label for _ in range(self.neg_sample_num)],
        }

class NLPCollateFn:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts_pos = [item['text_pos'] for item in batch]
        texts_gt = [item['text_gt'] for item in batch]
        texts_neg = [neg_sample for item in batch for neg_sample in item['text_neg']]
        #texts_neg = [item['text_neg'] for item in batch]
        label= [item['label'] for item in batch]
        #negative_label=[item['negative_label'] for item in batch]
        negative_labels = [item['negative_labels'] for item in batch]

        encoded_inputs_pos = self.tokenizer(
            texts_pos,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoded_inputs_gt = self.tokenizer(
            texts_gt,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoded_inputs_neg = self.tokenizer(
            texts_neg,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids_pos': encoded_inputs_pos['input_ids'],
            # 'attention_mask_pos': encoded_inputs_pos['attention_mask'],
            'input_ids_gt': encoded_inputs_gt['input_ids'],
            # 'attention_mask_gt': encoded_inputs_gt['attention_mask'],
            'input_ids_neg': encoded_inputs_neg['input_ids'],
            # 'attention_mask_neg': encoded_inputs_neg['attention_mask'],
            'label': torch.tensor(label)  ,
            'negative_label' :torch.tensor(negative_labels),

            'gt_att_mask' :  encoded_inputs_gt['attention_mask'],
            'pos_att_mask' : encoded_inputs_pos['attention_mask'], 
            'neg_att_mask' : encoded_inputs_neg['attention_mask']
            }
