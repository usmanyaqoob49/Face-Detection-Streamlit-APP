#Streamlit is used to make a web application arround your Machine learning model
import streamlit as st
#We will use components to enject html code to show on the websit
import streamlit.components.v1 as components
import cv2
import logging as log
import datetime as dt
from time import sleep


#so Haarcascade is basically the pretrained model for the face detection and we will use it to detect the face 
#the following xml file is trained on that image model so we can directly use it
cascPath = "haarcascade_frontalface_default.xml"
#Now we will use CascadClassifier function of cv2 for pointing out the location where we the have the file, I have downloaded it
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename = "webcam.log", level= log.INFO)


video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    #if does not able to capture video
    #isOpened() tell wheather video is being captured or not
    if not video_capture.isOpened():
        print('Unable to load camera')
        sleep(5)
        pass


            #As the model we are using is built on detecting the faces from image so to use videos we have to convert them in frames first
    ret, frame = video_capture.read()

            #when OpenCV reads the RGB image, it usually stores the image in BGR (Blue, Green, Red) "
            #channel. For the purposes of image recognition, we need to convert this BGR channel to gray

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



            #Now we will try to locate the exact location of face
            #detectMultiScale() function will help us to find the features/locations of the new image
            #it will use all the features from the faceCascade object to detect the features of the new image
    faces = faceCascade.detectMultiScale(
                #actually our image from which the face will be detected
            gray,

                #how much the image size is reduced at each image scale
                #basically xml file is trained on specific sized faces so we are reducing the size of image and we are 
                #zooming in the face 110%
            scaleFactor = 1.1,

                #how many neighbourse each candidate should have
            minNeighbors = 5,
                
                #object detected smaller than this size will be ignored
            minSize = (30,30)
            )


            #now we have to draw a rectangle arround the face


            #Function detectMultiScale() returns 4 values which are:
            #x co-ordinate, y co-ordinate, width (w), height (h)

    for (x, y, w, h) in faces:
                            #image         rectangle       green color
        cv2.rectangle(frame,(x,y), (x + w, y + h), (0, 255, 0), 2)

            
    if anterior != len(faces):
            #stroing the number of faces present in the image
        anterior = len(faces)
            #putting how many face were detected at which time
        log.info("faces: " + str(len(faces)) + " at " + str(dt.datetime.now()))

        # Display the resulting frame
    cv2.imshow('Video', frame)



            #Lastly we are waiting till the user inputs 'q', then exit all processes, releasing all captures
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


                 # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()