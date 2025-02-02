from transformers import AutoTokenizer
import numpy as np

def align_labels(text, char_labels, tokenizer):
    """
    将字符级标签映射到分词后的token级标签
    :param text: 原始文本（未分字的完整字符串）
    :param char_labels: 字符级标签列表（BIO格式）
    :param tokenizer: 分词器对象
    :return: (tokens, token_labels)
    """
    # Step 1: 字符级别预处理
    chars = list(text)
    assert len(chars) == len(char_labels), "文本与标签长度不一致"
    
    # Step 2: 获取分词映射关系
    encoding = tokenizer(
        text, 
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    token_offset = encoding["offset_mapping"]
    
    # Step 3: 生成token级标签
    token_labels = []
    for (start, end) in token_offset:
        # 获取当前token对应的第一个字符的标签
        first_char_idx = start
        token_labels.append(char_labels[first_char_idx])
    
    # 转换为token列表（可选，用于验证）
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    
    return tokens, token_labels

# 测试示例 ==============================================
if __name__ == "__main__":
    # 示例1：处理中文+英文混合文本
    text1 = "生生不息CSOL生化狂潮让你填弹狂扫"
    char_labels1 = ["O"]*4 + ["B-GAME","I-GAME","I-GAME","I-GAME"] + ["O"]*10
    
    # 示例2：处理含特殊符号的文本organization
    text2 = "那不勒斯vs锡耶纳以及桑普vs热那亚之上呢？"
    char_labels2 = ["B-ORG","I-ORG","I-ORG","I-ORG","O","O","B-ORG","I-ORG","I-ORG","O","O","B-ORG","I-ORG","O","O","B-ORG","I-ORG","I-ORG","I-ORG","I-ORG","O","O"]
    
    # 初始化分词器（使用适合中文的BERT分词器）
    tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/bert-base-chinese")
    
    # 执行转换
    for text, labels in [(text1, char_labels1), (text2, char_labels2)]:
        tokens, new_labels = align_labels(text, labels, tokenizer)
        
        # 可视化结果
        print("\n原始文本:", text)
        print("字符标签:", labels)
        print("分词结果:", tokens)
        print("Token标签:", new_labels)
        print("对齐验证:")
        for t, l in zip(tokens, new_labels):
            print(f"{t}({l})", end=" ")