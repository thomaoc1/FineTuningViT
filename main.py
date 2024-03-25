from transformers import AutoImageProcessor, ViTForImageClassification
import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

print(image_processor)


inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
label = model.config.id2label[predicted_label]

# Show the image with caption of predicted label
plt.imshow(np.array(image))
plt.title(f'Prediction: {label}')
plt.show()
