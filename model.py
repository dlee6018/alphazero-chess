import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Standard ResNet block (AlphaZero-style):
      x -> conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> add(skip) -> ReLU
    Input/Output: (B, C, 8, 8)
    """
    def __init__(self, channels: int = 64):
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
    Initial stem: (B, 18, 8, 8) -> (B, C, 8, 8)
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
    Policy head (simple flat move head):
      (B, C, 8, 8) -> conv1x1 to 2 channels -> BN -> ReLU -> flatten -> Linear -> logits
    Output: (B, n_moves) logits (mask illegal moves outside)
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
        x = x.view(x.size(0), -1)  # (B, 128)
        logits = self.fc(x)        # (B, n_moves)
        return logits


class ValueHead(nn.Module):
    """
    Value head:
      (B, C, 8, 8) -> conv1x1 to 1 channel -> BN -> ReLU -> flatten -> FC -> ReLU -> FC -> tanh
    Output: (B, 1) in [-1, 1]
    """
    def __init__(self, channels: int = 64, hidden: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(1 * 8 * 8, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # (B, 64)
        x = self.fc1(x)            # (B, hidden)
        x = self.relu(x)
        x = self.fc2(x)            # (B, 1)
        v = self.tanh(x)           # (B, 1) in [-1, 1]
        return v


class AlphaZeroChessNet(nn.Module):
    """
    Full model:
      Input:  (B, 18, 8, 8)
      Trunk:  ConvBlock -> N residual blocks, all at 'channels'
      Heads:  policy logits (B, n_moves), value (B, 1)
    """
    def __init__(self, channels: int = 64, n_blocks: int = 19, n_moves: int = 4672, value_hidden: int = 256):
        super().__init__()
        self.stem = ConvBlock(in_channels=18, channels=channels)
        self.residuals = nn.ModuleList([ResidualBlock(channels) for _ in range(n_blocks)])
        self.policy_head = PolicyHead(channels=channels, n_moves=n_moves)
        self.value_head = ValueHead(channels=channels, hidden=value_hidden)

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
    # Quick sanity check
    model = AlphaZeroChessNet(channels=64, n_blocks=19, n_moves=4672)
    dummy = torch.randn(4, 18, 8, 8)
    p, v = model(dummy)
    print("policy:", p)  # (4, 4672),(Batch, total_moves_possible)
    print("value:", v)   # (4, 1) - (batch, prob of winning in curr board)
    print("Total parameters:", model.count_parameters())