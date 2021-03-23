# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 16:16:41 2021

@author: ijkan
"""

#importing the libraries
import cv2

#loading the cascades
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade=cv2.CascadeClassifier("haarcascade_smile.xml")
eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")


def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        eye=eye_cascade.detectMultiScale(roi_gray,1.1,20)
        for(ex,ey,ew,eh) in eye:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
        smile=smile_cascade.detectMultiScale(roi_gray,1.7,22)
        for(sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh), (0,0,255), 2)
    return frame

#Doing some Face Recognition with the Webcam  
video_capture=cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read() #gets us the last frame of the webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#cv2.COLOR_BGR2GRAY tells to average BGR to get the contrast
    canvas = detect(gray, frame)   #call the detect function 
    cv2.imshow('Video', canvas)  #display all the processed images in an animated way
    if cv2.waitKey(1) & 0xFF == ord('q'):  #if we press q quit the process
        break
video_capture.release()
cv2.destroyAllWindows()
