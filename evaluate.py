import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix
from dataset.loader import OccludedGTSRB
from models.cnn_model import RoadSignGuardCNN
from evaluation.metrics import classification_metrics, occlusion_metrics


SEED, TRAIN_LIMIT, EVAL_LIMIT, BATCH_SIZE = 42, 2000, 2000, 32

def main():
    device = torch.device("cpu")
    dataset = OccludedGTSRB(augment_occlusion=True)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(SEED)).tolist()
    eval_indices = indices[TRAIN_LIMIT:TRAIN_LIMIT + EVAL_LIMIT]
    if not eval_indices:
        raise RuntimeError("No held-out evaluation images available.")
    loader = DataLoader(Subset(dataset, eval_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    checkpoint = torch.load("saved_models/roadsignguard.pth", map_location=device, weights_only=False)
    model = RoadSignGuardCNN(checkpoint.get("num_classes", 43)).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    sign_true, sign_pred, occ_true, occ_pred = [], [], [], []
    with torch.no_grad():
        for images, labels, occ in loader:
            sign_logits, occ_logits = model(images.to(device))
            sign_true.extend(labels.numpy().tolist())
            sign_pred.extend(sign_logits.argmax(1).cpu().numpy().tolist())
            occ_true.extend(occ.numpy().astype(int).tolist())
            occ_pred.extend((torch.sigmoid(occ_logits).cpu().numpy() >= 0.5).astype(int).tolist())
    print("\n=== ROADSIGNGUARD HELD-OUT EVALUATION ===")
    print(f"Evaluation images: {len(eval_indices)} (separate from the 2,000-image training sample)")
    print("Sign classification:", classification_metrics(sign_true, sign_pred))
    print("Occlusion detection:", occlusion_metrics(occ_true, occ_pred))
    import os
    os.makedirs("outputs", exist_ok=True)
    print("Sign confusion matrix saved to outputs/sign_confusion_matrix.csv")
    import numpy as np
    np.savetxt("outputs/sign_confusion_matrix.csv", confusion_matrix(sign_true, sign_pred), delimiter=",", fmt="%d")

if __name__ == "__main__":
    main()
