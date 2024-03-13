import torch
import torch.nn as nn

def sentence_map(sentence_list, char_to_id, max_length):
    """
    将句子中的每个字符映射到码表中
    :param sentence_list:待映射的句子
    :param char_to_id:（dict）码表
    :return:（tensor）每个字对应的编码
    """
    sentence_list.sort(key=lambda c: len(c), reverse=True)
    sentence_map_list = []
    for sentence in sentence_list:
        sentence_id_list = [char_to_id[c] for c in sentence]
        padding_list = [0] * (max_length-len(sentence))
        sentence_id_list.extend(padding_list)
        sentence_map_list.append(sentence_id_list)
    return torch.tensor(sentence_map_list, dtype=torch.long)
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, tag_to_id, input_feature_size, hidden_size,
                 batch_size, sentence_length, num_layers=1, batch_first=True):
        """
        :param vocab_size:词典大小
        :param tag_to_id:标签与id对应
        :param input_feature_size:词嵌入维度
        :param hidden_size:隐藏层向量维度
        :param batch_size:批训练大小
        :param sentence_length:句子长度
        :param num_layers:堆叠LSTM层数
        :param batch_first:是否将batch_size放在第一维
        """
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.tag_size = len(tag_to_id)
        self.embedding_size = input_feature_size
        self.hidden_size = hidden_size // 2
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.BiLSTM = nn.LSTM(input_size=input_feature_size, hidden_size=self.hidden_size, num_layers=num_layers,
                              bidirectional=True, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, self.tag_size)

    def forward(self, sentence_sequence):
        h0 = torch.randn(self.num_layers * 2, self.batch_size, self.hidden_size)
        c0 = torch.randn(self.num_layers * 2, self.batch_size, self.hidden_size)
        input_features = self.embedding(sentence_sequence)

        output, (hn, cn) = self.BiLSTM(input_features, (h0, c0))
        sequence_feature = self.linear(output)

        return sequence_feature

if __name__ == '__main__':
    sentence_list = [
        "确诊弥漫大b细胞瘤1年",
        "反复咳嗽、咳痰40年,再发伴气促5天。",
        "生长发育迟缓9年。",
        "右侧小细胞肺癌第三次化疗入院",
        "反复气促、心悸10年，加重伴胸痛3天",
        "反复胸闷、心悸、气促2多月,加重3天",
        "咳嗽、胸闷1月余，加重1周",
        "右上肢无力3年,加重伴肌肉萎缩半年"]

    char_to_id = {'<PAD>': 0}
    tag_to_id = {'0': 0, 'B-dis': 1, 'I-dis': 2, 'B-sym': 3, 'I-sym': 4}
    embedding_dim = 200
    hidden_dim = 100
    batch_size = 8
    num_layers = 1

    sentence_length = 20

    for sentence in sentence_list:
        for _char in sentence:
            if _char not in char_to_id:
                char_to_id[_char] = len(char_to_id)
    sentence_sqeuence = sentence_map(sentence_list, char_to_id, sentence_length)

    model = BiLSTM(vocab_size=len(char_to_id), tag_to_id=tag_to_id, input_feature_size=embedding_dim, hidden_size=hidden_dim,
                   batch_size=batch_size, sentence_length=sentence_length, num_layers=num_layers)

    sentence_feature = model(sentence_sqeuence)
    print(sentence_feature)