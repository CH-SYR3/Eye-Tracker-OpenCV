#Imports
import cv2
import time
import datetime


#Capture Webcam Video
capture = cv2.VideoCapture(0)

#Detect Faces & Eye Classifiers (requires Grayscale Image/Frame)
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")



while True:
    _, frame = capture.read()

    # #Covert Frame to a Grayscale Image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)

    # #Draws a Blue Rectangle on Faces Using BGR not RGB
    #for (x, y, width, height) in faces:                        #Color    Line Thickness
        #cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)

    for (x, y, width, height) in eyes:                        
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

    #Open Webcam
    cv2.imshow("Camera", frame)

    #Close Program When 'q' is Clicked
    if cv2.waitKey(1) == ord('q'):
        break

#Close Window
capture.release()
cv2.destroyAllWindows()

