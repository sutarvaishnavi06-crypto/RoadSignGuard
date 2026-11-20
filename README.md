# RoadSignGuard🚦

**RoadSignGuard: A Multi-Task CNN for Traffic Sign Classification and Occlusion Detection**

A compact CNN with two outputs: 43-class traffic-sign classification and binary synthetic-occlusion detection.

## Features
- GTSRB traffic sign images
- Synthetic leaf, mud, graffiti and shadow-like occlusion
- Shared CNN backbone with two prediction heads
- Accuracy, macro precision/recall/F1 and occlusion metrics
- Streamlit image-inference interface

## Setup (Windows / VS Code)
Use the Python installation already configured for this project.

```powershell
python -m pip install -r requirements.txt
python download_dataset.py
python test_dataset.py
```

## Train
```powershell
python train.py
```
Training uses a reproducible 2,000-image sample for a CPU-friendly pilot run (1,700 train / 300 validation, 3 epochs). Best checkpoint is written to `saved_models/roadsignguard.pth`.

## Evaluate
```powershell
python evaluate.py
```
Evaluation uses a reproducible held-out sample outside the 2,000 images selected by the training script. This is a held-out split from the training dataset, not the official GTSRB test benchmark. Synthetic occlusions are generated at runtime, so occlusion metrics can vary between runs.
## Run the interface
```powershell
streamlit run app.py
```
Upload a road-sign image to see the predicted class, confidence and occlusion probability. The inference interface requires a trained checkpoint.
## Dataset
The dataset is not bundled in this ZIP due to its size. `download_dataset.py` downloads and extracts GTSRB training images. Dataset folder detection supports nested extraction paths.
## Project layout
```
RoadSignGuard/
├── app.py
├── train.py
├── evaluate.py
├── download_dataset.py
├── test_dataset.py
├── requirements.txt
├── dataset/       # loader and occlusion generator
├── models/        # CNN
├── evaluation/    # metric functions
├── utils/         # class names
├── data/          # dataset downloaded locally
├── saved_models/  # trained checkpoint
└── outputs/        # evaluation outputs
```
## Limitations
Synthetic occlusion is only a proxy for real occlusion. This educational prototype is not validated for safety-critical or autonomous-driving use.
