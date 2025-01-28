"""
构建bert模型
    模型结构
    创建多层堆叠encoder
"""

from transformers import BertConfig, BertModel, AutoModel, AutoConfig
import torch




if __name__ == "__main__":

    config = AutoConfig.from_pretrained("pretrained_models/bert-base-chinese")

    # 加载模型预训练权重
    # 默认加载config.json文件
    model = AutoModel.from_pretrained("pretrained_models/bert-base-chinese")

    print(model)