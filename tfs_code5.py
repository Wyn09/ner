from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

if __name__ == "__main__":

    model = AutoModelForSequenceClassification.from_pretrained(
        "./pretrained_models/roberta_dianping",
        num_labels=5,
        ignore_mismatched_sizes=True
    )

    # print(model)
    # 模型调用
    input_ids = torch.tensor([[1,2],[4,5]])
    label = torch.tensor([1,0])
    # 模型推理，带有损失返回
    # result = model(input_ids)
    result = model(input_ids, labels=label)
    print(result.logits.shape)
    print(result.loss)