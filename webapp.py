import io
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
import base64
from yolo_model import yolo_model
from image_classification import predict_class
from image_captioning import generate_captions

# Initialize Flask app
app = Flask(__name__)

# Global variables to store the status messages
model_status = "Model not running"
results_status = ""
individual_download_links = []

font_pat = "arial.ttf"  # Change this path to the appropriate font file on your system
font_size=22
font_large = ImageFont.truetype(font_pat, font_size)  # Adjust font size as needed

@app.route("/", methods=["GET", "POST"])
def index():
    global model_status, results_status, individual_download_links

    if request.method == 'POST' and 'file' in request.files:
        # Update the model status message on the server interface
        model_status = "Model is running..."

        # Get uploaded images
        images = request.files.getlist('file')

        # Define list to store results
        results = []

        # Process each uploaded image
        for idx, img_file in enumerate(images):
            # Load image
            img = Image.open(img_file).convert('RGB')

            # Run YOLO detection and classification
            detections = yolo_model(img)
            class_labels = detections.names
            bounding_boxes = detections.xyxy[0].tolist()

            # Draw bounding boxes, classification label, and confidence on the image
            draw = ImageDraw.Draw(img)

            for box, class_label in zip(bounding_boxes, class_labels):
                box = [int(coord) for coord in box]
                # Draw bounding box
                draw.rectangle(box[0:4], outline="red", width=2)

                # Draw classification label on top of the box
                label_text = f"{class_label}"
                label_size = draw.textlength(label_text, font=font_large)
                font_metrics = font_large.getmetrics()  # Get font metrics
                text_height = sum(font_metrics)  # Approximate height based on metrics
                draw.rectangle([(box[0], box[1] - text_height), (box[0] + label_size, box[1])], fill="red")
                draw.text((box[0], box[1] - text_height), label_text, fill="white", font=font_large)

                # Draw confidence value beside the box
                confidence_text = f"Confidence: {box[4]:.2f}"
                draw.text((box[2], box[3]), confidence_text, fill="red", font=font_large)

            # Run image classification using the utility method
            predicted_class_label = predict_class(img_file)

            # Run image captioning
            captions = generate_captions(img_file)

            draw.text((10, 10), f"Caption: {captions}", fill="green", font=font_large)
            draw.text((10, 40), f"Class Label: {predicted_class_label}", fill="green", font=font_large)

            # Save the image with detection, classification, and caption
            img_path = f"static/uploaded_image{idx + 1}_result.jpg"
            img.save(img_path)


            # Store results
            result = {
                "ImageName": f"uploaded_image{idx + 1}",
                "Results": {
                    "Detection": {
                        "BoundingBoxes": bounding_boxes,
                        "ImageWithBoundingBoxes": img_path
                    },
                    "Classification": {
                        "ClassLabel": predicted_class_label,
                    },
                    "Captioning": captions
                }
            }
            results.append(result)

        individual_download_links = [url_for('download_result', filename=f'uploaded_image{i + 1}_result.jpg') for i in range(len(images))]

        results_status = "Results obtained!"

        return render_template('index_o.html', message=model_status, download_links=individual_download_links, results=results)

    return render_template('index_o.html', message=model_status, download_links=individual_download_links)


@app.route('/download_result/<filename>')
def download_result(filename):
    return send_file(f'static/{filename}', as_attachment=True)

@app.route("/status")
def get_status():
    return jsonify({"ModelStatus": model_status, "ResultsStatus": results_status})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
