from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import io

# Load DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

db = client["Food"]  # Replace with your database name
collection = db["items"]  # Replace with your collection name

@app.route('/lost_and_found', methods=['GET', 'POST'])
def lost_and_found():
    if request.method == 'POST':
        name = request.form['name']
        phone = request.form['phone']

        # Check if an image file is uploaded
        if 'image' not in request.files:
            return redirect(request.url)

        image_file = request.files['image']

        # Check if the file is empty
        if image_file.filename == '':
            return redirect(request.url)

        # Check if the file is an image
        if image_file and allowed_file(image_file.filename):
            # Read and process the uploaded image
            image = Image.open(image_file.stream)
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            # Process the detection results
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            # Prepare the results for rendering
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                detection_info = {
                    "label": model.config.id2label[label.item()],
                    "score": round(score.item(), 3),
                    "box": box
                }
                detections.append(detection_info)

                # Check if label exists in MongoDB
                existing_record = collection.find_one({"label": detection_info["label"]})

                if existing_record:
                    # If label exists, update the record
                    updated_record = {
                        "$push": {
                            "items": {
                                "name": name,
                                "phone": phone,
                                "images": [image_file.filename]
                            }
                        }
                    }
                    collection.update_one({"_id": existing_record["_id"]}, updated_record)
                else:
                    # If label doesn't exist, create a new record
                    new_record = {
                        "label": detection_info["label"],
                        "items": [
                            {
                                "name": name,
                                "phone": phone,
                                "images": [image_file.filename]
                            }
                        ]
                    }
                    collection.insert_one(new_record)

            return render_template('result.html', image=image, detections=detections)

    return render_template('lost_and_found.html')

@app.route('/list_labels')
def list_labels():
    labels = collection.distinct("label")
    return render_template('list_labels.html', labels=labels)

@app.route('/items/<label>')
def items_with_label(label):
    items = []
    records = collection.find({"label": label})
    for record in records:
        for item in record.get("items", []):
            items.append(item)
    return render_template('items_with_label.html', label=label, items=items)

# Helper function to check file extensions
def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

