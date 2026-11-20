import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from dataset.loader import OccludedGTSRB
from models.cnn_model import RoadSignGuardCNN

SEED, BATCH_SIZE, EPOCHS, VAL_SPLIT, LIMIT = 42, 32, 3, 0.15, 2000

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cpu")
    print(f"Training device: {device}")
    dataset = OccludedGTSRB(augment_occlusion=True)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(SEED)).tolist()[:LIMIT]
    val_size = int(VAL_SPLIT * len(indices))
    train_set, val_set = Subset(dataset, indices[val_size:]), Subset(dataset, indices[:val_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = RoadSignGuardCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    sign_loss, occ_loss = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    best_val = float("inf")
    os.makedirs("saved_models", exist_ok=True)
    for epoch in range(EPOCHS):
        model.train(); train_total = 0.0
        for images, labels, occ in train_loader:
            images, labels, occ = images.to(device), labels.to(device), occ.to(device)
            optimizer.zero_grad()
            sign_logits, occ_logits = model(images)
            loss = sign_loss(sign_logits, labels) + occ_loss(occ_logits, occ)
            loss.backward(); optimizer.step()
            train_total += loss.item()
        model.eval(); val_total = 0.0
        with torch.no_grad():
            for images, labels, occ in val_loader:
                sign_logits, occ_logits = model(images.to(device))
                val_total += (sign_loss(sign_logits, labels.to(device)) + occ_loss(occ_logits, occ.to(device))).item()
        train_avg, val_avg = train_total / len(train_loader), val_total / len(val_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_avg:.4f} | Validation Loss: {val_avg:.4f}")
        if val_avg < best_val:
            best_val = val_avg
            torch.save({"model_state": model.state_dict(), "num_classes": 43, "epoch": epoch + 1, "val_loss": best_val}, "saved_models/roadsignguard.pth")
            print("Best model checkpoint saved.")
    print(f"\nTraining complete. Best validation loss: {best_val:.4f}")
    print("Model saved in saved_models/roadsignguard.pth")


if __name__ == "__main__":
    main()
