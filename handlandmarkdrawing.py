import cv2 
import mediapipe as mp
import time

model_path = "hand_landmarker.task"

# Setup MediaPipe Task API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,  # Specifies running in single frame mode
    num_hands=2
)

cam = cv2.VideoCapture(2)#Change accordingly try 1 or 2 if 0 didnt work(or howevermany camera soruces you have)
cam.set(3, 1280)#setting width
cam.set(4, 720)#setting height
tx = 0
ty = 0
lx = 0
ly = 0
i = 0
is_touching = True

with HandLandmarker.create_from_options(options) as landmarker:
    while cam.isOpened(): #Making sure the camera is open
        success, frame = cam.read()
        if not success: break

        # Pass frame to landmarker (converts to RGB internally with mp.Image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)
        if result.hand_world_landmarks:#makes sure our code only works when we actually get the data
            
                jointA = result.hand_world_landmarks[0][8] #Getting the point 08 which is the index tip
                jointB = result.hand_world_landmarks[0][4] #Getting the point 04 which is the thumb tip
                x1,y1,z1 = jointA.x,jointA.y,jointA.z#assigning the values we get to shorter variables
                x2,y2,z2 = jointB.x,jointB.y,jointB.z#assigning the values we get to shorter variables
                print(abs(x1-x2),abs(y1-y2))

                jAnormal = result.hand_landmarks[0][8]
                jBnormal = result.hand_landmarks[0][4]
                
                h,w,_=frame.shape
                bx = int(jBnormal.x*w)
                by = int(jBnormal.y*h)
                cx = int(jAnormal.x*w)
                cy = int(jAnormal.y*h)
                if abs(x1 - x2) <0.015 and abs(y1 - y2) <0.008 :
                    if tx == 0: 
                        tx,ty = bx,by
                        is_touching = False
                        
                    elif tx != 0 and lx == 0 and is_touching == True :
                         lx,ly = cx,cy
                    else:
                         cv2.line((frame),pt1=(tx,ty),pt2=(lx,ly),color=(0,255,0),thickness=10)

                        
                elif tx != 0 and lx == 0:
                    cv2.line((frame),pt1=(tx,ty),pt2=(cx,cy),color=(0,255,0),thickness=10)
                    is_touching = True
                elif tx != 0 and lx != 0:
                    cv2.line((frame),pt1=(tx,ty),pt2=(lx,ly),color=(0,255,0),thickness=10)
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        tx = 0
                        ty = 0
                        lx = 0
                        ly = 0
  
                else:
                    cv2.line((frame),pt1=(bx,by),pt2=(cx,cy),color=(0,255,0),thickness=10)
                    

        cv2.imshow("Show Video", cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cam.release()
cv2.destroyAllWindows()