import pickle 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification




if __name__ == "__main__":

    model = AutoModelForTokenClassification.from_pretrained(
        "./pretrained_models/bert-base-chinese",
        num_labels=7
    )

    # config label mapping
    model.config.id2label = {
        'O':0,
        'B-LOC':1, 
        'I-LOC':2,
        'B-ORG':3, 
        'I-ORG':4, 
        'B-PER':5, 
        'I-PER':6, 
    }

    print(model)

    pass