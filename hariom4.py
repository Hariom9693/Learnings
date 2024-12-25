# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame
    ret, frame = cap.read()

    # Detect objects using YOLOv3
    outputs = net.forward(frame)

    # Extract detections
    detections = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                # Extract bounding box coordinates
                x, y, w, h = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                detections.append((x, y, w, h))

    # Track objects using Kalman filter
    tracks = []
    for detection in detections:
        # Predict object state
        x, y, w, h = detection
        state = np.array([x, y, w, h])
        prediction = kalman_filter.predict(state)

        # Update object state
        measurement = np.array([x, y, w, h])
        updated_state = kalman_filter.update(prediction, measurement)

        # Correct object state
        corrected_state = kalman_filter.correct(updated_state)

        tracks.append(corrected_state)

    # Visualize detections and tracks
    for detection in detections:
        x, y, w, h = detection
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for track in tracks:
        x, y, w, h = track
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display output
    cv2.imshow("Object Detection and Tracking", frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()