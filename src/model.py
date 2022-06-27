
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

#https://github.com/pytorch/examples/blob/main/mnist/main.py
class Cnn(nn.Module):
    def __init__(self, channels: int, output_size: int):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class ImageSeqClassifier(nn.Module):
    def __init__(self, channels, hidden_size, output_size):
        super(ImageSeqClassifier, self).__init__()
        self.cnn = Cnn(channels, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, seq_lengths=None) -> torch.Tensor:
        batch_size = input.size(0)
        input = input.flatten(0, 1).unsqueeze(1)
        features = self.cnn(input)
        features = features.view(batch_size, -1, features.shape[-1])
        padded_seq_embedded = pack_padded_sequence(
            features, seq_lengths.cpu().numpy(), batch_first=True,  enforce_sorted=False)
        self.gru.flatten_parameters()
        output, hidden = self.gru(padded_seq_embedded)
        fc_output = self.fc(hidden[-1])
        fc_output = F.sigmoid(fc_output)
        return fc_output
