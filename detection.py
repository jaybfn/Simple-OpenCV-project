
"""
Created on Thu Oct 14 11:17:41 2021

@author: jayesh
"""
#To download Haar Cascade for OpenCV 
# https://github.com/opencv/opencv/tree/master/data/haarcascades

# Importing the libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # haarcascade for frontalface
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')                  # haarcascade for eye
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')              # haarcascade for smile

# Defining a function that will do the detections    
def detect(gray, frame):
    """ This function takes an input as grey and original frame and returns 
    with a rectangular shape box for face, eye and smile """
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)                     # function DetectMultiFace detects one or several faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)            # A rectangular face is drawn across the face in grey image
        roi_gray = gray[y:y+h, x:x+w]                                       # we extract the important region in grey scale
        roi_color = frame[y:y+h, x:x+w]                                     # we extract the important region in color scale
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)              # this function detects eyes in the rectangular frame drawn across the face
        for (ex, ey, ew, eh) in eyes:                                       # draws a box across both the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)          # this function detects smile in the region where the face is detected
        for (sx, sy, sw, sh) in smiles:                                     # draws a rectangal across the lips when you smile
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()                                         # extract the last frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                          # converting the color BGR to gray frame
    canvas = detect(gray, frame)                                            # detect function which is defined above is called to detect all the feature of face
    cv2.imshow('Video', canvas)                                             # shows face live detection with eyes and smile
    if cv2.waitKey(1) & 0xFF == ord('q'):                                   # key to exit the webcam
        break
video_capture.release()
cv2.destroyAllWindows()