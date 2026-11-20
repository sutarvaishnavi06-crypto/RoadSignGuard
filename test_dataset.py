from dataset.loader import OccludedGTSRB

dataset = OccludedGTSRB(augment_occlusion=True)
print("\n--- DATASET VERIFICATION ---")
print("Total images:", len(dataset))
print("Classes found:", len(set(label for _, label in dataset.samples)))
image, label, occlusion = dataset[0]
print("Image shape:", image.shape)
print("First label:", label)
print("Occlusion label:", occlusion)
print("Dataset test successful!")
