import cv2
import dlib
from math import hypot
import numpy as np

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        #face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        
        for face in faces:
            landmarks = predictor(gray, face)
            print(type(face))
            #r_pt = (landmarks.part(27).x, landmarks.part(27).y)
            #r_pt_1 = (landmarks.part(30).x, landmarks.part(30).y)
            #r_pt_line = cv2.line(image, r_pt, r_pt_1, (0, 255, 0), 2)
           # print(r_pt_line)
           # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            #Gaze detection
            full_face_region = np.array([(landmarks.part(17).x,landmarks.part(17).y),
                                         (landmarks.part(26).x,landmarks.part(26).y),
                                       (landmarks.part(11).x,landmarks.part(11).y),
                                       (landmarks.part(5).x,landmarks.part(5).y)],np.int32)
                                       
                                       
        
            left_eye_region = np.array([(landmarks.part(36).x,landmarks.part(36).y),
                                       (landmarks.part(37).x,landmarks.part(37).y),
                                       (landmarks.part(38).x,landmarks.part(38).y),
                                       (landmarks.part(39).x,landmarks.part(39).y),
                                       (landmarks.part(40).x,landmarks.part(40).y),
                                       (landmarks.part(41).x,landmarks.part(41).y)], np.int32)
            print(left_eye_region)
            cv2.polylines(image,[left_eye_region],True,(0,0,255),2)
            cv2.polylines(image,[full_face_region],True,(0,255,0),2)
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
