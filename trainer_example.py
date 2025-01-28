"""
1.准备Dataset
2.Dataset数据预处理
3.创建TrainingArguments对象，封装模型训练参数
4.创建Trainer对象，封装模型训练控制参数、Dataset、评估方法、优化器、学习率调度器等
5.调用Trainer对象的train方法开始训练
"""
from datasets import load_dataset

if __name__ == "__main__":
    datasets = load_dataset("./datasets/clue-ner")
    
