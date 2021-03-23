# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 17:44:57 2020

@author: ijkan
"""

#Importing the libraries
import cv2

#Loading the cascades 
#cascading is series of filters that will apply one after the other to detect the face

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')

#Defining a function that will do the detection
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #faces are detected using this. faces are tuples of four elements x and y that are upperleft coordinates adn w and h are width and height
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),2) #first argument is the frame on which we want to draw rectangles, second argument is upper left corner coordinates, third argument is the bottom right corner of the rectangle
        roi_gray=gray[y:y+h,x:x+w] #zone of interest the [...] gives the rectangle
        roi_color=frame[y:y+h,x:x+w] #zone in colored image
        eyes=eye_cascade.detectMultiScale(roi_gray, 1.1, 3) #roi_gray is the region where we apply the cascade, first number is the scale factor, second number is minimum number of excepted neighbors
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh),(0,255,0),2) #fourth argument is the color of the rectangle, fifth argument is the thickness of the rectangle borders
    return frame
        
#Doing some Face Recognition with the Webcam  
video_capture=cv2.VideoCapture(0) # 0 if internal webcam and 1 if external webcam

while True:
    _, frame = video_capture.read() #gets us the last frame of the webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#cv2.COLOR_BGR2GRAY tells to average BGR to get the contrast
    canvas = detect(gray, frame)   #call the detect function 
    cv2.imshow('Video', canvas)  #display all the processed images in an animated way
    if cv2.waitKey(1) & 0xFF == ord('q'):  #if we press q quit the process
        break
    
video_capture.release()
cv2.destroyAllWindows()