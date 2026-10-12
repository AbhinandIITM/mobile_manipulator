"""
Train a lightweight MLP Policy Network via Behavioral Cloning
=============================================================
Trains a tiny 3-layer MLP on the expert dataset collected by collect_expert_data.py.
Maps 6 state features -> 4 action outputs (dx_cm, dy_cm, dz_cm, close_gripper).

Architecture:
    Input(6) -> Linear(256) -> ReLU -> Linear(256) -> ReLU -> Linear(128) -> ReLU
    Regression head: Linear(128) -> Linear(3)   [dx, dy, dz]
    Classification head: Linear(128) -> Linear(1) -> Sigmoid  [gripper]

Output:
    grasp_policy.pt   (trained model weights)
    training_curves.png

Usage:
    python train_policy.py
    python train_policy.py --epochs 30 --batch-size 512 --lr 1e-3
"""

import os, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "laya_expert_dataset.jsonl")
MODEL_PATH   = os.path.join(SCRIPT_DIR, "grasp_policy.pt")


# ─── Dataset ──────────────────────────────────────────────────────────────────
class GraspDataset(Dataset):
    """
    Loads the JSONL dataset. Each sample:
        state:  [dist_xy, height_above_block, gripper_open, target_dx, target_dy, target_dz]
        action: [dx_cm, dy_cm, dz_cm, close_gripper]
    """
    def __init__(self, path):
        self.samples = []
        print(f"Loading dataset from {path} ...")
        with open(path, "r") as f:
            for line in f:
                obj = json.loads(line.strip())
                s = obj["state"]
                a = obj["action"]

                state = np.array([
                    s["dist_xy_meters"],
                    s["height_above_block_meters"],
                    float(s["gripper_is_open"]),
                    float(s["target_dx_cm"]),
                    float(s["target_dy_cm"]),
                    float(s["target_dz_cm"]),
                ], dtype=np.float32)

                action_xyz = np.array([
                    float(a["dx_cm"]),
                    float(a["dy_cm"]),
                    float(a["dz_cm"]),
                ], dtype=np.float32)

                action_gripper = np.array([float(a["close_gripper"])], dtype=np.float32)

                self.samples.append((state, action_xyz, action_gripper))

        print(f"  Loaded {len(self.samples):,} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, xyz, grip = self.samples[idx]
        return (torch.from_numpy(state),
                torch.from_numpy(xyz),
                torch.from_numpy(grip))


# ─── Model ────────────────────────────────────────────────────────────────────
class GraspPolicyMLP(nn.Module):
    """
    Tiny MLP policy: 6 state features -> (3 delta cm floats + 1 gripper bool).
    ~200k parameters, runs in microseconds on CPU.
    """
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(6, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.xyz_head     = nn.Linear(128, 3)    # Regression: dx, dy, dz in cm
        self.gripper_head = nn.Linear(128, 1)    # Classification: open/close

    def forward(self, x):
        feat   = self.backbone(x)
        xyz    = self.xyz_head(feat)
        gripper = torch.sigmoid(self.gripper_head(feat))
        return xyz, gripper


# ─── Training ─────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Dataset
    dataset    = GraspDataset(DATASET_PATH)
    val_size   = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0)

    print(f"  Train: {train_size:,}  Val: {val_size:,}")

    # Model
    model = GraspPolicyMLP().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Optimiser & losses
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_xyz_fn = nn.MSELoss()
    loss_grip_fn = nn.BCELoss()

    history = {"train_loss": [], "val_loss": [], "val_xyz_mae": [], "val_grip_acc": []}

    print(f"\nTraining for {args.epochs} epochs ...")
    print("-" * 60)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for states, xyz_tgt, grip_tgt in train_loader:
            states   = states.to(device)
            xyz_tgt  = xyz_tgt.to(device)
            grip_tgt = grip_tgt.to(device)

            pred_xyz, pred_grip = model(states)

            loss = loss_xyz_fn(pred_xyz, xyz_tgt) + loss_grip_fn(pred_grip, grip_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * states.size(0)

        scheduler.step()
        train_loss /= train_size

        # --- Validate ---
        model.eval()
        val_loss  = 0.0
        xyz_mae   = 0.0
        grip_correct = 0

        with torch.no_grad():
            for states, xyz_tgt, grip_tgt in val_loader:
                states   = states.to(device)
                xyz_tgt  = xyz_tgt.to(device)
                grip_tgt = grip_tgt.to(device)

                pred_xyz, pred_grip = model(states)

                loss = loss_xyz_fn(pred_xyz, xyz_tgt) + loss_grip_fn(pred_grip, grip_tgt)
                val_loss  += loss.item() * states.size(0)
                xyz_mae   += (pred_xyz - xyz_tgt).abs().mean(dim=1).sum().item()
                grip_correct += ((pred_grip > 0.5) == grip_tgt.bool()).sum().item()

        val_loss  /= val_size
        xyz_mae   /= val_size
        grip_acc   = grip_correct / val_size * 100

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_xyz_mae"].append(xyz_mae)
        history["val_grip_acc"].append(grip_acc)

        elapsed = time.time() - t0
        print(f"  Ep {epoch:>3}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"xyz_mae={xyz_mae:.3f}cm  grip_acc={grip_acc:.1f}%  "
              f"({elapsed:.1f}s)")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "xyz_mae": xyz_mae,
                "grip_acc": grip_acc,
            }, MODEL_PATH)

    print(f"\nBest model saved to: {MODEL_PATH}")
    print(f"Best val loss: {best_val_loss:.4f}")

    # ─── Training curves plot ──────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    epochs_range = range(1, args.epochs + 1)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("MLP Policy Network - Training Curves", fontsize=14, fontweight="bold")

    axs[0].plot(epochs_range, history["train_loss"], label="Train Loss")
    axs[0].plot(epochs_range, history["val_loss"],   label="Val Loss")
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss")
    axs[0].set_title("Total Loss (MSE + BCE)"); axs[0].legend(); axs[0].grid(True)

    axs[1].plot(epochs_range, history["val_xyz_mae"], color="orange")
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("MAE (cm)")
    axs[1].set_title("XYZ Delta MAE"); axs[1].grid(True)

    axs[2].plot(epochs_range, history["val_grip_acc"], color="green")
    axs[2].set_xlabel("Epoch"); axs[2].set_ylabel("Accuracy (%)")
    axs[2].set_title("Gripper Classification Accuracy"); axs[2].grid(True)

    plt.tight_layout()
    curves_path = os.path.join(SCRIPT_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=150); plt.close()
    print(f"Training curves saved to: {curves_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=1024)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
