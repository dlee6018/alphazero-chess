import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Input/Output: (B, C, 8, 8)
    """
    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ConvBlock(nn.Module):
    """
    (B, 18, 8, 8) -> (B, C, 8, 8)
    """
    def __init__(self, in_channels: int = 18, channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PolicyHead(nn.Module):
    """
    (B, C, 8, 8) -> (B, n_moves)
    """
    def __init__(self, channels: int = 64, n_moves: int = 4672):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2 * 8 * 8, n_moves)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits


class ValueHead(nn.Module):
    """
    (B, C, 8, 8) -> (B, 1) scalar value in [-1, 1]
    """
    def __init__(self, channels: int = 64, head_channels: int = 32, hidden: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(channels, head_channels, kernel_size=1, bias=False)
        self.ln = nn.LayerNorm([head_channels, 8, 8])
        self.fc1 = nn.Linear(head_channels * 8 * 8, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.ln(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.tanh(x)  # (B, 1) in [-1, 1]


class AlphaZeroChessNet(nn.Module):
    """
    Full model:
      Input:  (B, 18, 8, 8)
      Output: policy logits (B, n_moves), value (B, 1) in [-1, 1]
    """
    def __init__(self, channels: int = 64, n_blocks: int = 19, n_moves: int = 4672, value_hidden: int = 256):
        super().__init__()
        self.stem = ConvBlock(in_channels=18, channels=channels)
        self.residuals = nn.ModuleList([ResidualBlock(channels) for _ in range(n_blocks)])
        self.policy_head = PolicyHead(channels=channels, n_moves=n_moves)
        self.value_head = ValueHead(channels=channels, hidden=value_hidden)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Small init for value head output to keep tanh in linear regime at start
        nn.init.uniform_(self.value_head.fc2.weight, -0.01, 0.01)
        nn.init.zeros_(self.value_head.fc2.bias)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        for block in self.residuals:
            x = block(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AlphaZeroChessNet(channels=128, n_blocks=19, n_moves=4672)
    dummy = torch.randn(4, 18, 8, 8)
    p, v = model(dummy)
    print("policy:", p.shape)  # (4, 4672)
    print("value:", v.shape)   # (4, 1)
    print("value range:", v.min().item(), v.max().item())
    print("Total parameters:", model.count_parameters())
