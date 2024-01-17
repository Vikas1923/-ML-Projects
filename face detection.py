import cv2

# Specify the Haar cascade file path
alg = "haarcascade_frontalface_default.xml"

# Initialize the video capture object (you may need to change the index to 0 if it's the default camera)
cam = cv2.VideoCapture(0)

# Load the Haar cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + alg)

while True:
    # Read frames from the camera
    ret, img = cam.read()

    if not ret:
        print("Error reading frame")
        break

    # Convert the frame to grayscale for faster processing
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use the Haar cascade classifier to detect faces
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=4)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the image with rectangles
    cv2.imshow("Face Detection", img)

    # Check for the 'Esc' key to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
