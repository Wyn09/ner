"""
1.准备Dataset
2.Dataset数据预处理
3.创建TrainingArguments对象，封装模型训练参数
4.创建Trainer对象，封装模型训练控制参数、Dataset、评估方法、优化器、学习率调度器等
5.调用Trainer对象的train方法开始训练
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
import evaluate
import numpy as np

def entities_tags_proc(item):
    data_size = len(item["text"])
    ner_tags = []

    # for i in range(data_size):
    #     text_len = len(item["text"][i])
    #     entities = item["entities"][i]
    #     tags = torch.tensor([entity_index["O"]] * text_len)
    #     for entity in entities:
    #         start, end = entity["start_offset"], entity["end_offset"]
    #         label = entity["label"]
    #         tags[start] = entity_index[label] * 2 - 1
    #         tags[start + 1 : end] = entity_index[label] * 2
    #     ner_tags.append(tags)
    # return {"ner_tags" : ner_tags}

    for i in range(data_size):
        text_len = len(item["text"][i])
        entities = item["entities"][i]
        tags = torch.tensor([tags_t2i["O"]] * text_len)
        for entity in entities:
            start, end = entity["start_offset"], entity["end_offset"]
            label = entity["label"].upper()
            tags[start] = tags_t2i["B-"+label]
            tags[start + 1 : end] = tags_t2i["I-"+label]
        ner_tags.append(tags)
    return {"ner_tags" : ner_tags}



# def corpus_porc(item):
#     # 语料中文本转换成为模型输入项
#     input_data = tokenizer(
#         item["text"],
#         truncation=True,
#         add_special_tokens=False,
#         max_length=512,
#         return_offsets_mapping=True
#     )

#     # **生成token_index与tags对齐**

#     total_adjusted_labels = []
#     # 遍历每个样本
#     for k in range(len(input_data["input_ids"])):
#         # toekn和输入字符之间的映射关系
#         word_ids = input_data.word_ids(k)
#         exits_label_ids = item["ner_tags"][k] # 输入项每个字符的tag
#         adjust_label_ids = [] # 修正后每个token的tag

#         prev_wid = -1
#         i = -1

#         for wid in word_ids:
#             if wid != prev_wid:
#                 prev_wid = wid
#                 i += 1
#             adjust_label_ids.append(exits_label_ids[i])
#         total_adjusted_labels.append(adjust_label_ids)
#     input_data["labels"] = total_adjusted_labels
#     return input_data


def corpus_porc(item):
    # 语料中文本转换成为模型输入项
    
    # 因为分词不区分大小写，CSOL会被映射为[UNK]，所以要转换为csol
    item["text"] = np.vectorize(str.lower)(item["text"]).tolist()
    input_data = tokenizer(
        item["text"],
        truncation=True,
        add_special_tokens=False,
        max_length=512,
        return_offsets_mapping=True
    )

    # **生成token_index与tags对齐**

    total_adjusted_labels = []
    # 遍历每个样本
    for k in range(len(input_data["input_ids"])):
        exits_label_ids = item["ner_tags"][k] # 输入项每个字符的tag
        adjust_label_ids = [] # 修正后每个token的tag
        for start, _ in input_data["offset_mapping"][k]:
            adjust_label_ids.append(exits_label_ids[start])
        total_adjusted_labels.append(adjust_label_ids)
      
    input_data["labels"] = total_adjusted_labels
    return input_data




def compute_metrics(result):

    # 获取推理结果和labels
    predicts, labels = result
    predicts = np.argmax(predicts, axis=-1)
    # 去除填充项并将tag_index转换为tag标签
    truncated_predicts = [[tags_i2t[p] for p, l in zip(pred, label) if l!= -100] 
                         for pred, label in zip(predicts, labels)]

    truncated_labels = [[tags_i2t[l] for p, l in zip(pred, label) if l!= -100] 
                         for pred, label in zip(predicts, labels)]

    result = seqeval.compute(predictions=truncated_predicts,references=truncated_labels)
    
    return {
        "precision": result["overall_precision"],
        "recall": result["overall_recall"],
        "f1": result["overall_f1"],
        "acc": result["overall_accuracy"],
    }


if __name__ == "__main__":
    datasets = load_dataset("./datasets/clue-ner")
    # 映射 entities 映射为 tag_index结构
    # 0 -> 0
    # PER:B-PER -> 1 I-PER -> 2  (1*2)-1 = 1  (1*2)=2
    # LOC:B-LOC -> 3 I-LOC -> 4  (2*2)-1 = 3  (2*2)=4
    # ORG:B-ORG -> 5 I-ORG -> 6  (3*2)-1 = 5  (3*2)=6

    # entity_index {'PER':1, 'LOC':2, 'ORG':3}
    # tag_index = entity_index * 2 - 1


    """
    enetity_set = set()
    for item in datasets['train']:
        entity_list = item['entities']
        for entity in entity_list:
            enetity_set.add(entity['label'])
    print(enetity_set)
   
    {'game', 'scene', 'name', 'organization', 'movie', 'book', 'position', 'address', 'company', 'government'}
    """

    labels = ['O', 'game', 'scene', 'name', 'organization', 'movie', 'book', 'position', 'address', 'company', 'government']
    # entity_index = {e:i for i, e in enumerate(labels)}
    
    # 构建字典
    tags_t2i = {
            labels[i] if i == 0 
            else 
                "B-"+labels[int((i+1)/2)].upper() if i % 2 != 0 
                else 
                "I-"+labels[int((i/2))].upper()
            : i
        for i in range(0, 2 * (len(labels) - 1)+1)
    }
    # 反向字典
    tags_i2t = {v: k for k, v in tags_t2i.items()}

    tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/bert-base-chinese")

    ds1 = datasets.map(entities_tags_proc, batched=True)
    ds2 = ds1.map(corpus_porc, batched=True, batch_size=3)
    ds2.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
    # print(next(iter(ds2["train"])))

    model = AutoModelForTokenClassification.from_pretrained("./pretrained_models/bert-base-chinese", num_labels=len(labels) * 2 -1)

    training_args = TrainingArguments(
        output_dir="example_trainer",
        num_train_epochs=3,
        save_safetensors=False,
        save_only_model=True,   # 只保存模型，不包含训练参数
        per_device_train_batch_size=32,
        eval_strategy="epoch", # 默认"no"，可选值:["no", "steps", "epoch"]。每隔一定步数评估，每个epoch结束评估
        # eval_steps=0.05  # [int, float]
        
    )

    # datacollator负责label矩阵填充
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100
    )
    
     # 加载评估对象
    seqeval = evaluate.load("./eval/seqeval") # 要开VPN下载，不然会卡住

    # 创建训练对象
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds2["train"],
        tokenizer=tokenizer,
        data_collator=data_collator, #Trainer负责对input填充，collator负责对label填充
        eval_dataset=ds2["validation"],
        compute_metrics=compute_metrics,

    )
    trainer.train()
    

    predict = trainer.predict(ds2["test"])
    print(predict)
