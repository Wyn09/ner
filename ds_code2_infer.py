import torch
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset 
from torch.utils.data import DataLoader



if __name__ =="__main__":

    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/roberta_dianping")

    dataset = load_dataset("./datasets/clue/iflytek")

    # callback func
    data_conv = lambda item : tokenizer(item["sentence"], padding=True, truncation=True, max_length=512)
    
    conved_data = dataset.map(data_conv, batched=True)
    # delete columns
    conved_data = conved_data.remove_columns(["sentence", "idx"])
    conved_data = conved_data.rename_column("label", "labels")
    conved_data = conved_data.with_format("torch")

    dl = DataLoader(conved_data["validation"], batch_size=16, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(\
        "./pretrained_models/roberta_dianping",
        num_labels=119,
        ignore_mismatched_sizes=True
        ).to(device)
    state_dict = torch.load("./model_seqcls.bin")
    model.load_state_dict(state_dict)


    model.eval()
    with torch.no_grad():
        bar = tqdm(dl)
        for item in bar:
            item = {k:v.to(device) for k,v in item.items()}
            result = model(**item)
            y_hat = result.logits.argmax(-1)
            acc = (y_hat == item["labels"]).sum() / len(y_hat)
            
            bar.set_description(f"accuracy: {acc * 100:.4f}%")
    pass

