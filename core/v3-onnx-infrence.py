import cv2
import numpy as np
import onnxruntime as ort

# Function to preprocess each frame
def preprocess_frame(frame, img_size):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_shape = frame.shape[:2]  # (height, width)
    resized_frame = cv2.resize(frame_rgb, (img_size, img_size))
    frame_data = resized_frame / 255.0  # Normalize to [0, 1]
    frame_data = np.transpose(frame_data, (2, 0, 1))  # Change to (channels, height, width)
    frame_data = np.expand_dims(frame_data, axis=0).astype(np.float32)  # Add batch dimension
    return frame_data, original_shape

# Function to post-process the output
def postprocess_output(output, original_shape, img_size, conf_threshold=0.5, iou_threshold=0.5):
    output = output[0]
    output = np.squeeze(output)  # Remove batch dimension if present

    # Check if the output is a 2D array (detections x 85), else reshape if necessary
    if len(output.shape) == 1:
        output = output.reshape(-1, 85)

    boxes, scores, class_ids = [], [], []

    for detection in output:
        if detection[4] >= conf_threshold:  # Confidence score
            box = detection[:4]
            scores.append(detection[4])
            class_ids.append(np.argmax(detection[5:]))
            boxes.append(box)

    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)

    # Ensure boxes array is 2D
    if len(boxes.shape) == 1:
        boxes = boxes.reshape(-1, 4)

    # Scale boxes back to original frame shape
    boxes[:, 0] *= original_shape[1] / img_size  # x_min
    boxes[:, 1] *= original_shape[0] / img_size  # y_min
    boxes[:, 2] *= original_shape[1] / img_size  # x_max
    boxes[:, 3] *= original_shape[0] / img_size  # y_max

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
        boxes = boxes[indices]
        scores = scores[indices]
        class_ids = class_ids[indices]
    
    return boxes, scores, class_ids

# Load ONNX model with specified providers
onnx_model_path = "83_map_best.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(onnx_model_path, providers=providers)

# Set input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Open video file or capture device
video_path = "cropped.mp4"  # Use 0 for webcam
cap = cv2.VideoCapture(0)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img_size = 416  # The size YOLOv5 expects
    frame_data, original_shape = preprocess_frame(frame, img_size)

    # Run inference
    output = session.run([output_name], {input_name: frame_data})

    # Post-process the output
    boxes, scores, class_ids = postprocess_output(output, original_shape, img_size)
    print(boxes, scores, class_ids)
    # Draw boxes on the frame
    for box, score, class_id in zip(boxes, scores, class_ids):
        print(score, class_id,"?????????")
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"Class {class_id}: {score:.2f}"
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLOv5 Inference', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
