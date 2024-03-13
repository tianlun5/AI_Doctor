import torch
from torch import nn
import time
import math
from doctor_offline.review_model.bert_chinese_encode import get_bert_encode_for_single
import random
import pandas as pd
from Model_RNN import RNN
import matplotlib.pyplot as plt

train_data = pd.read_csv("../ner_model/data/train_data.csv", header=None, sep='\t')
train_data1 = train_data.values.tolist()
rnn = RNN(768, 128, 2)
criterion = nn.NLLLoss()
learning_rate = 0.005

def randomTrainingExample(train_data):
    category, line = random.choice(train_data)
    line_tensor = get_bert_encode_for_single(line)
    category_tensor = torch.tensor([int(category)])
    return category, line, category_tensor, line_tensor

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item()

def valid(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    with torch.no_grad():
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)
    loss = criterion(output, category_tensor)
    return output, loss.item()

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m * 60
    return '%dm%ds' % (m, s)

if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_iters = 50000
    plot_every = 1000

    train_current_loss = 0
    train_current_acc = 0
    valid_current_loss = 0
    valid_current_acc = 0

    all_train_loss = []
    all_train_acc = []
    all_valid_loss = []
    all_valid_acc = []

    start = time.time()
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample(train_data1)
        category_, line_, category_tensor_, line_tensor_ = randomTrainingExample(train_data1)

        train_output, train_loss = train(category_tensor, line_tensor)
        valid_output, valid_loss = valid(category_tensor_, line_tensor_)

        train_current_loss += train_loss
        train_current_acc += (train_output.argmax(1) == category_tensor).sum().item()
        valid_current_loss += valid_loss
        valid_current_acc += (valid_output.argmax(1) == category_tensor_).sum().item()

        if iter % plot_every == 0:
            train_average_loss = train_current_loss / plot_every
            train_average_acc = train_current_acc / plot_every
            valid_average_loss = valid_current_loss / plot_every
            valid_average_acc = valid_current_acc / plot_every

            print("Iter:", iter, "| TimeSince", timeSince(start))
            print("Train Loss:", train_average_loss, "| Train Acc:", train_average_acc)
            print("Valid Loss", valid_average_loss, "| Valid Acc:", valid_average_acc)

            all_train_loss.append(train_average_loss)
            all_train_acc.append(train_average_acc)
            all_valid_loss.append(valid_average_loss)
            all_valid_acc.append(valid_average_acc)

            train_current_loss = 0
            train_current_acc = 0
            valid_current_loss = 0
            valid_current_acc = 0

    plt.figure(0)
    plt.plot(all_train_loss, label="Train Loss")
    plt.plot(all_valid_loss, color='red', label='Valid Loss')
    plt.legend(loc="upper left")
    plt.savefig('./loss.png')

    plt.figure(1)
    plt.plot(all_train_acc, label="Train Acc")
    plt.plot(all_valid_acc, color='red', label='Valid Acc')
    plt.legend(loc="upper left")
    plt.savefig('./acc.png')

    MODEL_PATH = 'BERT_RNN.pth'
    torch.save(rnn.state_dict(), MODEL_PATH)
