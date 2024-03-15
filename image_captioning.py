# image_captioning.py
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def generate_captions(img_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    raw_image = Image.open(img_path).convert('RGB')

    # Conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt")
    out = model.generate(**inputs)
    conditional_caption = processor.decode(out[0], skip_special_tokens=True)

    # Unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    unconditional_caption = processor.decode(out[0], skip_special_tokens=True)

    return {
        "ConditionalCaption": conditional_caption,
        "UnconditionalCaption": unconditional_caption
    }
