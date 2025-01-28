import pickle 
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification
from data_proc import build_dataloader
from seqeval.metrics import accuracy_score, classification_report, f1_score


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
    model.config.id2label = {
        'O':0,
        'B-LOC':1, 
        'I-LOC':2,
        'B-ORG':3, 
        'I-ORG':4, 
        'B-PER':5, 
        'I-PER':6, 
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

            bar.set_description(f"epoch: {e+1}/{epoch}, loss: {loss.item():.4f}")

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
                torch.save(model.state_dict(), "best_ner_model.pth")
                print("Model Saved Successfully!")
    pass