# train_swin_se_sam.py
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from swin_se import SwinSEFER
from sam import SAM


# -----------------------------
# Dataset class (Four4All)
# -----------------------------
class Four4All(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = int(self.labels.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label


# -----------------------------
# Transforms
# -----------------------------
def build_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAutocontrast(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])


# -----------------------------
# Training / evaluation helpers
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 1st SAM step
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.first_step()

        # 2nd SAM step
        logits2 = model(images)
        loss2 = criterion(logits2, targets)
        loss2.backward()
        optimizer.second_step()

        optimizer.zero_grad()

        _, preds = torch.max(logits2, dim=1)
        running_loss += loss2.item() * images.size(0)
        running_correct += (preds == targets).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        _, preds = torch.max(logits, dim=1)

        running_loss += loss.item() * images.size(0)
        running_correct += (preds == targets).sum().item()
        total += images.size(0)

        all_preds.append(preds.cpu())
        all_labels.append(targets.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc, macro_f1, all_labels, all_preds


def plot_learning_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"))
    plt.close()


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # Text in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    # ===========================================
    # CONFIGURATION - Set your values here
    # ===========================================

    # Dataset paths (CSVs and image directories for each split)
    train_csv = "path/to/train.csv"           # TODO: Set your train CSV path
    # TODO: Set your train images directory
    train_img_dir = "path/to/train/images"
    val_csv = "path/to/val.csv"               # TODO: Set your validation CSV path
    # TODO: Set your validation images directory
    val_img_dir = "path/to/val/images"
    test_csv = "path/to/test.csv"             # TODO: Set your test CSV path
    # TODO: Set your test images directory
    test_img_dir = "path/to/test/images"

    # Training hyperparameters
    epochs = 25
    batch_size = 64
    lr = 1e-3
    rho = 0.05
    momentum = 0.9
    weight_decay = 1e-4

    # Model configuration
    backbone = "swin_tiny_patch4_window7_224"
    num_classes = 8                           # TODO: Set number of classes

    # Device and output
    device_str = "cuda"
    out_dir = "./runs_swin_se_sam"

    # ===========================================
    # END CONFIGURATION
    # ===========================================

    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Datasets & loaders
    train_tfms = build_transforms(train=True)
    val_tfms = build_transforms(train=False)
    test_tfms = build_transforms(train=False)

    train_ds = Four4All(train_csv, train_img_dir, transform=train_tfms)
    val_ds = Four4All(val_csv, val_img_dir, transform=val_tfms)
    test_ds = Four4All(test_csv, test_img_dir, transform=test_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = SwinSEFER(
        backbone_name=backbone,
        num_classes=num_classes,
        reduction=16,
        pretrained=True,
    ).to(device)

    # Optimizer & scheduler (SAM + SGD + StepLR every 10 epochs). [file:1]
    base_opt = torch.optim.SGD
    optimizer = SAM(
        model.parameters(),
        base_optimizer=base_opt,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        rho=rho,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer.base_optimizer, step_size=10, gamma=0.1
    )
    criterion = nn.CrossEntropyLoss()

    # Training history for learning curves
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(out_dir, "best_model.pt")

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_macro_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_acc": val_acc,
                },
                best_ckpt_path,
            )
            print(f"  -> New best model saved to {best_ckpt_path}")

    # Plot and save learning curves
    plot_learning_curves(history, out_dir)

    # -----------------------------
    # Final TEST evaluation
    # -----------------------------
    print("\nLoading best model for final test evaluation...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    test_loss, test_acc, test_macro_f1, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )

    prec_macro = precision_score(y_true, y_pred, average="macro")
    rec_macro = recall_score(y_true, y_pred, average="macro")
    prec_weighted = precision_score(y_true, y_pred, average="weighted")
    rec_weighted = recall_score(y_true, y_pred, average="weighted")

    print("\n=== Test Results ===")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro Precision: {prec_macro:.4f}")
    print(f"Macro Recall: {rec_macro:.4f}")
    print(f"Macro F1: {test_macro_f1:.4f}")
    print(f"Weighted Precision: {prec_weighted:.4f}")
    print(f"Weighted Recall: {rec_weighted:.4f}")

    # Detailed classification report
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"),
               cm, delimiter=",", fmt="%d")

    # If you have fixed label names, define them here, e.g.:
    # class_names = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    # Make sure length == num_classes and order matches labels.
    class_names = [str(i) for i in range(num_classes)]

    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    print(f"\nConfusion matrix saved to: {cm_path}")
    print(f"Curves saved to: {out_dir}")


if __name__ == "__main__":
    main()
