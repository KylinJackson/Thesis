import torch.nn as nn
import torch


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


class DA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, origin):
        da = torch.tensor([0.], requires_grad=True)

        if targets > origin and outputs > origin:
            da = da + 1
        elif targets < origin and outputs < origin:
            da = da + 1
        elif targets == origin and -0.01 < outputs - origin < 0.01:
            da = da + 1
        return da
