"""
wandb tutorial: MNIST classification with gradient tracking,
system utilization, memory monitoring, and model artifacts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

PROJECT = "wandb-tutorial-mnist"


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B, 32, 14, 14)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B, 64, 7, 7)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    config = {
        "epochs": 10,
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "device": str(device),
    }

    run = wandb.init(project=PROJECT, config=config)

    # --- Data ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST("./data", train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # --- Model ---
    model = ConvNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # Auto-log gradients + weights every 100 forward passes
    wandb.watch(model, log="all", log_freq=100)

    global_step = 0

    for epoch in range(config["epochs"]):
        # ====== Training ======
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()

            # Gradient norms
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            conv_params = [p for n, p in model.named_parameters() if "conv" in n and p.grad is not None]
            conv_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in conv_params]))
            fc_params = [p for n, p in model.named_parameters() if "fc" in n and p.grad is not None]
            fc_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in fc_params]))

            optimizer.step()
            global_step += 1

            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            # Per-step logging
            wandb.log({
                "train/loss": loss.item(),
                "gradients/total_norm": total_norm.item(),
                "gradients/conv_layers_norm": conv_norm.item(),
                "gradients/fc_layers_norm": fc_norm.item(),
                "lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)

        train_acc = correct / total
        avg_loss = epoch_loss / len(train_loader)

        # ====== Validation ======
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_loss += F.cross_entropy(logits, y_batch).item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Per-epoch logging
        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": avg_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
        }, step=global_step)

        # MPS memory
        if device.type == "mps":
            wandb.log({
                "memory/mps_allocated_mb": torch.mps.current_allocated_memory() / 1024**2,
                "memory/mps_driver_mb": torch.mps.driver_allocated_memory() / 1024**2,
            }, step=global_step)

        scheduler.step()

        print(f"Epoch {epoch:2d} | Train Loss: {avg_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # --- Save model artifact ---
    checkpoint_path = "mnist_model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    artifact = wandb.Artifact("mnist-convnet", type="model", description="ConvNet trained on MNIST")
    artifact.add_file(checkpoint_path)
    run.log_artifact(artifact)

    wandb.summary["final_val_acc"] = val_acc
    wandb.summary["final_val_loss"] = val_loss
    wandb.summary["total_steps"] = global_step
    wandb.summary["total_params"] = sum(p.numel() for p in model.parameters())

    wandb.finish()
    print(f"\n✓ Done. View run at: {run.url}")


if __name__ == "__main__":
    main()
