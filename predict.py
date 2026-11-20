import torch
from PIL import Image
from torchvision import transforms

from models.cnn_model import RoadSignGuardCNN
from utils.labels import CLASS_NAMES


DEVICE = torch.device("cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


def load_model():
    checkpoint = torch.load(
        "saved_models/roadsignguard.pth",
        map_location=DEVICE,
        weights_only=False
    )

    model = RoadSignGuardCNN(
        checkpoint.get("num_classes", 43)
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model


def predict_image(model, image):
    image = image.convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        sign_logits, occ_logits = model(tensor)

        sign_prob = torch.softmax(sign_logits, dim=1)
        confidence, predicted_class = sign_prob.max(dim=1)

        occ_prob = torch.sigmoid(occ_logits).item()

    return {
        "sign": CLASS_NAMES[predicted_class.item()],
        "sign_confidence": confidence.item() * 100,
        "occlusion": "Occluded" if occ_prob >= 0.5 else "Clear",
        "occlusion_confidence": occ_prob * 100
    }