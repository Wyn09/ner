import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoTokenizer


if __name__ == "__main__":
    
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/bert-base-chinese")

    tags_t2i = {
        'O':0,
        'B-LOC':1, 
        'I-LOC':2,
        'B-ORG':3, 
        'I-ORG':4, 
        'B-PER':5, 
        'I-PER':6, 
    }

    tags_i2t = {v : k for k ,v in tags_t2i.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        "./pretrained_models/bert-base-chinese",
        num_labels=7
    ).to(device)

    state_dict = torch.load("./best_ner_model.pth", weights_only=False)
    model.load_state_dict(state_dict)

    inputs_text = ["馆长赛伯先生邀请代表团团长为博物馆题字当翻译刚刚译出熔古铸今个大字的含义时他连连点头称赞这个字正好道出了博物馆信奉的宗旨所在", 
                   "回到阡陌八十路的餐厅里把苹果放在了窗台上的盆里。"]
    inputs = tokenizer(inputs_text, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt").to(device)
    outputs = model(**inputs)
    y_hat = outputs.logits.argmax(-1) # (N, T)

    pred = []
    for seq, mask in zip(y_hat, inputs["attention_mask"]):
        seq = torch.masked_select(seq, mask.bool()).cpu().tolist()
        pred.append(seq)

    entities = [] # [{"LOC":{"start":3, "end":5}, "PER":{"start":0, "end":5}},{...}, ...]
    for idx, seq in enumerate(pred):
        start = -1
        end = -1
        ent_dic = {}
        for i, tk in enumerate(seq):
            if tk == 0:
                if start != -1:
                    end = i
                    entity = inputs_text[idx][start : end]
                    ent_dic[label] = {
                        "entity":entity,
                        "start":start,
                        "end":end,
                    }
                    start = -1

            else:
                label = tags_i2t[tk].split("-")[-1]
                if start == -1:
                    start = i
           
        entities.append(ent_dic)

    print(entities)

