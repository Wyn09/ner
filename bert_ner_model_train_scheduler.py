import pickle 
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification
from data_proc import build_dataloader
from seqeval.metrics import accuracy_score, classification_report, f1_score
from transformers import get_linear_schedule_with_warmup


"""
    动态学习率
    差分学习率
"""

if __name__ == "__main__":

    batch_size = 16
    lr = 1e-5
    epoch = 3
    device = "cuda"


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

    # load data
    with open("./corpus.dat", "rb") as f:
        train_ds, test_ds = pickle.load(f)

    # create DataLoader
    train_dl = build_dataloader(train_ds, tags_t2i=tags_t2i, batch_size=batch_size, shuffle=True)
    test_dl = build_dataloader(test_ds, tags_t2i=tags_t2i, batch_size=batch_size, shuffle=False)



    model = AutoModelForTokenClassification.from_pretrained(
        "./pretrained_models/bert-base-chinese",
        num_labels=7
    )

    # config label mapping
    model.config.label2id = tags_t2i
    model.config.id2label = tags_i2t

    # 差分学习率
    # model parameters grouping
    params = list(model.named_parameters())
    bert_params, classifier_params = [], []
    for name, params in params:
        if "bert" in name:
            bert_params.append(params)
        else:
            classifier_params.append(params)

    # weight_decay 不适用于normalization
    param_groups = [
        {"params":bert_params, "lr":1e-5},
        {"params":classifier_params, "weight_decay":0.1, "lr":1e-3}
    ]

    optimizer = torch.optim.AdamW(param_groups)

    # 动态学习率
    # 学习率调度器 warmup
    train_steps = epoch * len(train_dl)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=train_steps, num_warmup_steps=100)

    model = model.to(device)
    print(device)
    for e in range(epoch):
        bar = tqdm(train_dl)
        for item, tgs in bar:
            item, tgs = item.to(device), tgs.to(device)
            outputs = model(**item, labels=tgs)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            bar.set_description(f"epoch: {e+1}/{epoch}, bert_lr: {scheduler.get_last_lr()[0]}, classifier_lr: {scheduler.get_last_lr()[1]}, loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            max_f1 = 0
            total_loss = 0
            total_count = 0
            total_pred, total_tags = [], []
            bar = tqdm(test_dl, desc=f"Test: ")
            for item, tgs in bar:
                item, tgs = item.to(device), tgs.to(device)
                outputs = model(**item, labels=tgs)
                total_loss += outputs.loss.item()
                total_count += 1

                # seqeval estimator
                pred = outputs.logits.argmax(-1)

                tk_pred = pred.masked_select(item["attention_mask"].bool())
                tk_true = tgs.masked_select(item["attention_mask"].bool())
                pred_tags = [tags_i2t[tk.item()] for tk in tk_pred]
                true_tags = [tags_i2t[tk.item()] for tk in tk_true]

                total_pred += pred_tags
                total_tags += true_tags



            print("Accuracy:", accuracy_score([total_tags], [total_pred]))

            f1 = f1_score([total_tags], [total_pred])
            print("F1 score:", f1)

            print(classification_report([total_tags], [total_pred]))

            print(f"Test loss: {(total_loss / total_count):.4f}")

            # save model
            if f1 > max_f1:
                max_f1 = f1
                # torch.save(model.state_dict(), "best_ner_model.pth")
                # print("Model Saved Successfully!")
    pass