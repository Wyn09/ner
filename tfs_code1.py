"""
构建bert模型
    模型结构
    创建多层堆叠encoder
"""

from transformers import BertConfig, BertModel
import torch




if __name__ == "__main__":

    # 模型结构配置属性对象
    config = BertConfig(num_hidden_layers=10)

    # 模型对象
    model = BertModel(config=config)
    
    token_inputs = torch.tensor([[6378, 2526, 1345], [7563, 8765, 6343]])
    # last_hidden_state, pooler_output
    result = model(token_inputs)
    print(model)
    print(result["last_hidden_state"].shape)
    print(result.pooler_output.shape)
    