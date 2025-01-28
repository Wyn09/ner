import pickle 
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification
from data_proc import build_dataloader
from seqeval.metrics import accuracy_score, classification_report, f1_score


"""
    分布式训练DDP
"""
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# 定义训练
def train(rank, model, epoch, batch_size, lr, train_ds, test_ds, tags_i2t, tags_t2i, world_size):
    setup(rank, world_size)

    # 分布式训练采样器
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    train_dl = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size)

    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank)

    train_dl = build_dataloader(train_ds, tags_t2i=tags_t2i, batch_size=batch_size, shuffle=True, sampler=train_sampler)
    test_dl = build_dataloader(test_ds, tags_t2i=tags_t2i, batch_size=batch_size, shuffle=False, sampler=test_sampler)


    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        bar = tqdm(train_dl)
        for item, tgs in bar:
            item, tgs = item.to(rank), tgs.to(rank)
            outputs = ddp_model(**item, labels=tgs)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bar.set_description(f"epoch: {e+1}/{epoch}, rank:{rank}, loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            max_f1 = 0
            total_loss = 0
            total_count = 0
            total_pred, total_tags = [], []
            bar = tqdm(test_dl, desc=f"Test: ")
            for item, tgs in bar:
                item, tgs = item.to(rank), tgs.to(rank)
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


    cleanup()


def main(model, epoch, batch_size, lr, train_ds, test_ds, tags_i2t, tags_t2i):
    world_size = torch.cuda.device_count()
    """
        nprocs: 调用train()时要创建进程数量
        join: 
            当join设置为True时，它会阻塞主进程，直到由spawn启动的所有子进程都执行完毕。也就是说，主进程会等待所有子进程完成其任务后，才会继续执行后续代码。如果设置为False，主进程不会等待子进程，子进程会在后台继续运行，主进程则会继续执行后续代码，这种情况下，需要开发者自行处理子进程的生命周期管理等问题。 
    """
    mp.spawn(train, args=(model, epoch, batch_size, lr, train_ds, test_ds, tags_i2t, tags_t2i, world_size), nprocs=world_size, join=True)

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



    model = AutoModelForTokenClassification.from_pretrained(
        "./pretrained_models/bert-base-chinese",
        num_labels=7
    )

    # config label mapping
    model.config.label2id = tags_t2i
    model.config.id2label = tags_i2t

    main(model, epoch, batch_size, lr, train_ds, test_ds, tags_i2t, tags_t2i)


    pass