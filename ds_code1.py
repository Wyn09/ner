from datasets import load_dataset

if __name__ == "__main__":

    # 加载数据集
    dataset = load_dataset("./datasets/clue/iflytek")

    # print(dataset)
    print(dataset["train"])

    for item in dataset["train"]:
        print(item["sentence"])
        print(item["label"])
        break
