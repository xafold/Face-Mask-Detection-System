import cv2
import math
import numpy as np
from tensorflow.keras import models
from libs.utils.utils import show_bboxes
from libs.utils.align import get_aligned_faces
from libs.mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
detector =MTCNN()

font_letter = cv2.FONT_HERSHEY_PLAIN

facemodel = load_model(r'April17.h5')
cap = cv2.VideoCapture(0)


def detect_mask(image):
    category = {'Masked': 0, 'Unmasked': 1}
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
    # print(grayscale.shape)
    cropped_grayscale = cv2.resize(grayscale, (130, 130), interpolation = cv2.INTER_AREA) 
    resized = cropped_grayscale.reshape((1, 130, 130, 1)).astype(np.float32) / 255.
    
    opt = facemodel.predict(resized)
    idx = np.argmax(opt)
    accuracy = list(opt.flatten())[idx]
    print(accuracy)
    mask_class = list(category.keys())[list(category.values()).index(idx)]
    return mask_class,accuracy



#########################FPS#################
import datetime
fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0
############################################
while cap.isOpened():
    face_mask = ''
    accuracy = int()
    output = np.zeros((800,1000,3), dtype="uint8")

    ret, frame = cap.read()
    frame  = cv2.flip(frame,1)
    landmarks, bboxs = detector(frame)
    faces = get_aligned_faces(frame, bboxs, landmarks)
    frame = show_bboxes(frame , bboxs , landmarks,'.')

    try:

        face_mask,accuracy = detect_mask(faces[0])
        accuracy = accuracy * 100
        accuracy = float("{:.2f}".format(accuracy))
        print(face_mask)
    except:
        pass
    
    #################################FPS##########################################################
    total_frames = total_frames + 1
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    ###############################################################################################
    
    cv2.putText(output,"REAL TIME FACE MASK DETECTION ",(100,50), font_letter,3, (0,0,255),2)
    cv2.putText(output,str(face_mask),(400,150), font_letter,2, (255,255,51),2)
    cv2.putText(output,'Accuracy : '+str(accuracy)+'%',(400,180), font_letter,2, (0,0,255),2)




    output[320:800, 180:820] = frame
    cv2.imshow("Frame", output)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()