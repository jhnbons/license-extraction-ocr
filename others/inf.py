import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(config.yaml, "runs\detect\train6\weights\best.pt")  # Replace with actual paths

# Load and preprocess input image
image = cv2.imread("datasets\license\images\inference\mmda-drivers-license-restriction-codes-apprehension-1578284319.jpg")  # Replace with the actual image path
resized_img = cv2.resize(image, (model.input_width, model.input_height))
preprocessed_img = model.preprocess_image(resized_img)

# Perform inference
predictions = model.predict(preprocessed_img)

# Process predictions and draw bounding boxes on the original image
for pred in predictions:
    class_id, confidence, bbox = pred
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f"Class: {class_id}, Conf: {confidence:.2f}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display or save the annotated image
cv2.imshow("YOLOv8 Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()