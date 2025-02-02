import pickle 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


def read_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
        tokens, tags = [], []
        corpus = []
        for line in lines:
            line = line.strip()
            if line == "" :
                if len(tokens) <= 512:
                    corpus.append((tokens, tags))
                tokens, tags = [] ,[]
                continue
            token, tag = line.split()
            # if token == "":
            #     print("EEEE")
            # if tag == "":
            #     print("TTTT")
            tokens.append(token)
            tags.append(tag)
    return corpus

def build_dataloader(corpus, tags_t2i, batch_size=4, shuffle=True, sampler=None):
    tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/bert-base-chinese")
    
    def batch_data_proc(batch_data):
        tokens, tags = [], []
        for tks, tgs in batch_data:
            tokens.append(tks)
            tags.append(torch.tensor([tags_t2i[t] for t in tgs]))

        # 统一转换收集toekn集合
        data_input = tokenizer(tokens, padding=True, truncation=True, return_tensors="pt",
                               is_split_into_words=True, add_special_tokens=False)
        
        return data_input, pad_sequence(tags, batch_first=True, padding_value=-100)
    if sampler is not None:
        return DataLoader(corpus, batch_size=batch_size, sampler=sampler, collate_fn=batch_data_proc)
    
    return DataLoader(corpus, batch_size=batch_size, shuffle=shuffle, collate_fn=batch_data_proc)



if __name__ == "__main__":

    train_file = "./example.train"
    dev_file = "./example.dev"
    test_file = "./example.test"

    train_ds = read_corpus(train_file)
    dev_ds = read_corpus(dev_file)
    test_ds = read_corpus(test_file)

    train_ds += dev_ds

    print("语料记录总数:", len(train_ds))
    print(train_ds[10][0])
    print(train_ds[10][1])


    # 保存加载预料集合
    with open("corpus.dat", 'wb') as f:
        pickle.dump((train_ds, test_ds), f)

    
    # dataset: corpus
    # dataloader: batch, collate_fn 数据格式转换
    tags_t2i = {
        'O':0,
        'B-LOC':1, 
        'I-LOC':2,
        'B-ORG':3, 
        'I-ORG':4, 
        'B-PER':5, 
        'I-PER':6, 
    }

    dl_train = build_dataloader(train_ds, tags_t2i)

    for item, tags in dl_train:
        print(item)
        print(tags)
        break
    pass