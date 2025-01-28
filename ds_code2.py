import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def batch_process(batch_data):
    inputs, padding_mask, type_ids, labels = [], [], [], []

    for item in batch_data:
        inputs.append(item["input_ids"][:512])
        padding_mask.append(item["attention_mask"][:512])
        type_ids.append(item["token_type_ids"][:512])
        labels.append(item["label"])
    
    inputs = pad_sequence(inputs, batch_first=True)
    padding_mask = pad_sequence(padding_mask, batch_first=True)
    type_ids = pad_sequence(type_ids, batch_first=True)
    return {
        "input_ids":inputs,
        "attention_mask":padding_mask,
        "token_type_ids":type_ids,
        "labels":torch.tensor(labels)
    }
    


if __name__ == "__main__":

    batch_size = 16
    lr = 1e-5
    epoch = 10
    device = "cuda"



    tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/roberta_dianping")

    # 加载数据集
    dataset = load_dataset("./datasets/clue/iflytek")

    # 数据预处理的回调函数
    # model_data_cov = lambda item : tokenizer(item["sentence"], padding=True, truncation=True, max_length=512) # 第2种写法
    model_data_cov = lambda item : tokenizer(item["sentence"], truncation=True) # 第1种写法


    # 通过回调函数 逐一转换原有的dataset样本记录
    # 在字典中增加三个key:input_ids, token_type_ids, attention_mask
    # 如果回调函数tokenizer没有padding=True，并且不指定batch_size，默认为1000，与DataLoader的batch_size不一致，那么后续的DataLoader要写collate_fn进行padding
    # cov_dataset = dataset.map(model_data_cov, batched=True, batch_size=batch_size) # 第2种写法
    cov_dataset = dataset.map(model_data_cov, batched=True) # 第1种写法

    # 删除字段
    cov_dataset = cov_dataset.remove_columns(["sentence", "idx"])
    # cov_dataset = cov_dataset.rename_column("label", "labels")# 第2种写法#
    # 映射类型
    cov_dataset = cov_dataset.with_format("torch")

    # dl = DataLoader(cov_dataset["train"], batch_size=batch_size, shuffle=False) # 第2种写法
    dl = DataLoader(cov_dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=batch_process) # 第1种写法


    # for item in cov_dataset["train"]:
    #     print(item)
    #     # print(item["sentence"])
    #     # print(item["label"])
    #     break

    # for item in dl:
    #     print(item)
    #     break


    # model
    model = AutoModelForSequenceClassification.from_pretrained(
        "./pretrained_models/roberta_dianping",
        num_labels=119,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    print(device)
    for e in range(epoch):
        bar = tqdm(dl)
        update_count = 1
        for item in bar:
            item = {k:v.to(device) for k, v in item.items()}
            result = model(**item)
            optimizer.zero_grad()
            result.loss.backward()
            optimizer.step()
            bar.set_description(f"epoch:{e+1}, loss:{result.loss.item():.4f}")

            if best_loss < result.loss.item():
                best_loss = result.loss.item()
                torch.save(model.state_dict(), "model_seqcls.bin")
                bar.set_postfix_str(f"Model Saved Successfully!, update_count: {update_count}, best loss: {result.loss.item():.4f}")
                update_count += 1

