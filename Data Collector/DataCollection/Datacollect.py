import cv2
import random
import math
import numpy as np
from libs.utils.utils import show_bboxes
from libs.utils.align import get_aligned_faces
from libs.mtcnn.mtcnn import MTCNN
from beepy import beep
import time

dataDIR = "C:\\Users\\Xafold\\Desktop\\Major_Project\\DataCollection\\Data\\"

detector =MTCNN()

print("********** Please enter your Name and Type: Ex: Utsav Masked/Unmasked *********")
name,category = input("Enter a two value: ").split()
image_no = random.randint(0, 1000000)


cap = cv2.VideoCapture(0)
counter = 0
############################################
beep(sound='ready')

while cap.isOpened():
    image_no += 1 
    ret, frame = cap.read()
    frame  = cv2.flip(frame,1)
    landmarks, bboxs = detector(frame)

    faces = get_aligned_faces(frame, bboxs, landmarks)
    frame = show_bboxes(frame , bboxs , landmarks,'.')
    try:

        for face in faces:
            
            if category == "Masked":
                counter += 1
                
                x = dataDIR+str(category)+'\\'+str(category)+str(name)+str(image_no)+".JPG"
                print(x)
                cv2.imwrite(x,face)
                counter += 1
                time.sleep(0.1)
                
            if category == "Unmasked":
            
                x = dataDIR+str(category)+'\\'+str(category)+str(name)+str(image_no)+".JPG"
                cv2.imwrite(x,face)
                counter += 1   
                time.sleep(0.1)
            
            #if category == "Improper":
                #x = dataDIR+str(category)+'\\'+str(category)+str(name)+str(image_no)+".JPG"
                #cv2.imwrite(x,face)
                #counter += 1   
                #time.sleep(0.1)

            #cv2.imshow("faces",face)
            #cv2.setWindowProperty("faces", cv2.WND_PROP_TOPMOST, 1)


    except:
        print("No face Deteceted")
    cv2.putText(frame, "Please Wait unitil 100", (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        
    cv2.putText(frame, str(counter), (5, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    cv2.imshow("Frame", frame)
    cv2.setWindowProperty("Frame", cv2.WND_PROP_TOPMOST, 1)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    if counter == 150:
        break
 
cap.release()
cv2.destroyAllWindows()