# image_classification_utils.py
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

def predict_class(image_path):
    custom_image = Image.open(image_path)
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    inputs = processor(images=custom_image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]
