import torch

model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-chinese')
# 获得对应的字符映射器，它将中文的每个字映射成一个数字
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-chinese')

def get_bert_encode_for_single(text):
    #bert的tokenizer映射后为结果前后添加开始（101）和结束表示符（102），对长文本有效，在这里无效
    indexed_tokens = tokenizer.encode(text)[1:-1]
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        outputs = model(tokens_tensor)
    #[batch_size, sequence_length, hidden_size]
    return outputs.last_hidden_state[0]

if __name__ == '__main__':
    text = '你好，周杰伦'
    outputs = get_bert_encode_for_single(text)
    # print(outputs)