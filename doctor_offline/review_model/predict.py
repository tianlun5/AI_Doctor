import torch
from doctor_offline.review_model.bert_chinese_encode import get_bert_encode_for_single
from Model_RNN import RNN
import os


def _test(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)
    return output

def predict(input_line):
    with torch.no_grad():
        output = _test(get_bert_encode_for_single(input_line))
        _, topi = output.topk(1, 1)
        return topi.item()

def batch_predict(input_path, output_path):
    csv_list = os.listdir(input_path)
    for csv in csv_list:
        with open(os.path.join(input_path, csv), "r", encoding='utf-8') as fr:
            with open(os.path.join(output_path, csv), "w", encoding='utf-8') as fw:
                input_line = fr.readline()
                res = predict(input_line)
                if res == 1:
                    fw.write(input_line + '\n')
                else:
                    pass

if __name__  == '__main__':
    input_size = 768
    hidden_size = 128
    n_categories = 2
    MODEL_PATH = 'BERT_RNN.pth'

    rnn = RNN(input_size, hidden_size, n_categories)
    rnn.load_state_dict(torch.load(MODEL_PATH))
    #
    # input_line = '点瘀样尖针性发多'
    # print((predict(input_line)))

    input_path = '../structured/noreviewed'
    out_path = '../structured/reviewed'
    batch_predict(input_path, out_path)