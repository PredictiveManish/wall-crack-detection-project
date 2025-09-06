import cv2
import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model("crack_detector.h5")

# Use HTTP instead of HTTPS
url = "<place-url-video>"
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Check IP/Port and make sure phone/laptop are on same WiFi.")
        break
    
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32")/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0][0]
    label = "No Crack" if pred<0.9 else "Crack Detected"

    color = (0,255,0) if label=="No Crack" else (0,0,255)
    cv2.putText(frame, label, (20,40), cv2.FONT_HERSHEY_SIMPLEX,1, color, 2)

    cv2.imshow("Wall crack detection", frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
