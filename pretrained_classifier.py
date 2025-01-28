from transformers import AutoModel, PreTrainedTokenizerFast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class BertClassifier(nn.Module):

    def __init__(self, pretrained_model_path, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 特征提取 使用正常lr = 1e-3
        # 全量微调 使用较小lr = le-5 ~ 1e-4
        with torch.no_grad():
            # bert
            result = self.bert(input_ids, attention_mask, token_type_ids)
        # classifier
        output = self.classifier(result.pooler_output)
        return output
    
    
    