import cv2 
import mediapipe as mp
import math

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
i = 10
cam = cv2.VideoCapture(2)#Change accordingly try 1 or 2 if 0 didnt work(or howevermany camera soruces you have)
cam.set(3, 1280)#setting width
cam.set(4, 720)#setting height

with HandLandmarker.create_from_options(options) as landmarker:
    while cam.isOpened(): #Making sure the camera is open
        success, frame = cam.read()
        if not success: break

        # Pass frame to landmarker (converts to RGB internally with mp.Image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)
        ##detected_hand = result.hand_world_landmarks[0]
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
            

            if abs(x1 - x2) < 0.015 and abs(y1 - y2) < 0.008:
                print("Finger tips are close together")
                i = i + 1
                angle = angle + 0.05  #controls the rotation speed in radians
                #specifying the 4 points needed for a rectangle(without the turning multiplier)
                dx_tr, dy_tr = (40 + i * 2), (50 + i * 2)#top right 
                dx_br, dy_br = (40 + i * 2), (-50 - i * 2)#bottom right
                dx_tl, dy_tl = (-40 - i * 2), (50 + i * 2)#top left
                dx_bl, dy_bl = (-40 - i * 2), (-50 - i * 2)#bottom left

                #Adding the turning multiplier with sine cosine so the math is easier
                ptr = (
                    int(bx + (dx_tr * math.cos(angle) - dy_tr * math.sin(angle))),
                    int(by + (dx_tr * math.sin(angle) + dy_tr * math.cos(angle)))
                )
                pbr = (
                    int(bx + (dx_br * math.cos(angle) - dy_br * math.sin(angle))),
                    int(by + (dx_br * math.sin(angle) + dy_br * math.cos(angle)))
                )
                ptl = (
                    int(bx + (dx_tl * math.cos(angle) - dy_tl * math.sin(angle))),
                    int(by + (dx_tl * math.sin(angle) + dy_tl * math.cos(angle)))
                )
                pbl = (
                    int(bx + (dx_bl * math.cos(angle) - dy_bl * math.sin(angle))),
                    int(by + (dx_bl * math.sin(angle) + dy_bl * math.cos(angle)))
                )

                cv2.line(frame, pt1=ptr, pt2=ptl, color=(0, 255, 0), thickness=10)
                cv2.line(frame, pt1=ptr, pt2=pbr, color=(0, 255, 0), thickness=10)
                cv2.line(frame, pt1=pbl, pt2=pbr, color=(0, 255, 0), thickness=10)
                cv2.line(frame, pt1=pbl, pt2=ptl, color=(0, 255, 0), thickness=10)
                a = (
                    int((ptr[0] + ptl[0]) / 2),
                    int((ptr[1] + ptl[1]) / 2)
                )  

                b = (
                    int((ptr[0] + pbr[0]) / 2),
                    int((ptr[1] + pbr[1]) / 2)
                )  

                c = (
                    int((pbl[0] + pbr[0]) / 2),
                    int((pbl[1] + pbr[1]) / 2)
                )  

                d = (
                    int((pbl[0] + ptl[0]) / 2),
                    int((pbl[1] + ptl[1]) / 2)
                )  
                cv2.line(frame, pt1=a, pt2=b, color=(0, 255, 0), thickness=10)
                cv2.line(frame, pt1=a, pt2=c, color=(0, 255, 0), thickness=10)
                cv2.line(frame, pt1=d, pt2=c, color=(0, 255, 0), thickness=10)
                cv2.line(frame, pt1=d, pt2=b, color=(0, 255, 0), thickness=10)
                
            else:
                cv2.line(frame, pt1=(bx, by), pt2=(cx, cy), color=(0, 255, 0), thickness=10)
                i = 10
                angle = 0.0  
        cv2.imshow("Show Video", cv2.flip(frame, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'): break

cam.release()
cv2.destroyAllWindows()