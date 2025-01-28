from transformers import pipeline

# 流水线对象创建模型从输入到输出
classifier = pipeline(
    "sentiment-analysis",
    model="pretrained_models/roberta_dianping"
)
output = classifier(
    [
        "东西不错， 分量手感， 价格都没得说，点赞。关节阻尼合适，细节到位。多图，大家自己看。",
        "买了都没拆，这外包装就让人想要退货的冲动。"
    ]
)
print(output)