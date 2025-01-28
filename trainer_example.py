"""
1.准备Dataset
2.Dataset数据预处理
3.创建TrainingArguments对象，封装模型训练参数
4.创建Trainer对象，封装模型训练控制参数、Dataset、评估方法、优化器、学习率调度器等
5.调用Trainer对象的train方法开始训练
"""

import torch
from datasets import load_dataset





def entities_tags_proc(item):
    data_size = len(item["text"])
    ner_tags = []
    for i in range(data_size):
        text_len = len(item["text"][i])
        entities = item["entities"][i]
        tags = torch.tensor([entity_index["O"]] * text_len)
        for entity in entities:
            start, end = entity["start_offset"], entity["end_offset"]
            label = entity["label"]
            tags[start] = entity_index[label] * 2 - 1
            tags[start + 1 : end] = entity_index[label] * 2
        ner_tags.append(tags)
    return {"ner_tags" : ner_tags}


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
    entity_index = {e:i for i, e in enumerate(labels)}

    ds = datasets.map(entities_tags_proc, batched=True)
    print(next(iter(ds["train"])))
