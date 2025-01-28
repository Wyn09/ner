from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch

if __name__ == "__main__":
    # 分词器
    toeknizer = AutoTokenizer.from_pretrained("./pretrained_models/roberta_dianping")

    # # 分词(中文每个字符都是一个token)
    # tokens = toeknizer.tokenize(["人工智能应用", "未来技术方向"])
    # print(tokens)

    # # 分词+转码
    # token_idx = toeknizer.encode(["人工智能应用", "未来技术方向"])
    # print(token_idx)

    # # 解码
    # words = toeknizer.decode(token_idx)
    # print(words)

    # 根据输入的数据生成模型训练用的tensor集合
    # 结果字典包含模型调用的3个必要参数 
    
    # result = toeknizer.batch_encode_plus(["人工智能应用", "未来技术方向"])
    result = toeknizer([["人工智能应用", "未来技术发展的方向！"],
                        ["你是谁？","我是艾芬"]], 
                       return_tensors="pt",
                       padding=True,
                    #    truncation=True
                       )
    
    print(result.input_ids)
    print(result.token_type_ids)
    print(result.attention_mask)

    # 加载预训练模型
    model = AutoModel.from_pretrained("./pretrained_models/roberta_dianping")

    # 模型调用
    output = model(**result)
    print(output.last_hidden_state.shape)
    print(output.pooler_output.shape)