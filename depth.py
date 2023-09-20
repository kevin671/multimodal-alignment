# %%
import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, DPTFeatureExtractor, DPTForDepthEstimation

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model_name = "Intel/dpt-hybrid-midas"

feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)
model = DPTForDepthEstimation.from_pretrained(model_name)
image_processor = AutoImageProcessor.from_pretrained(model_name)

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

encoding = feature_extractor(image, return_tensors="pt")
print(encoding.keys())

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

# visualize the prediction
output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)
depth.shape
# %%
