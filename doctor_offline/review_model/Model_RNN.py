import torch
import torch.nn as nn
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input1, hidden1):
        combined = torch.cat((input1, hidden1), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

if __name__ == "__main__":
    input_size = 768
    hidden_size = 128
    n_categories = 2
    input = torch.rand(1, input_size)
    hidden = torch.rand(1, hidden_size)
    rnn = RNN(input_size, hidden_size, n_categories)
    outputs, hidden = rnn(input, hidden)
    print("outputs:", outputs)
    print("hidden:", hidden)