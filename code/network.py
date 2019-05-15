import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        r_out, _ = self.lstm(x, None)
        out = self.out(r_out)
        return out


if __name__ == '__main__':
    lstm = LSTM(30)
    print(lstm)
