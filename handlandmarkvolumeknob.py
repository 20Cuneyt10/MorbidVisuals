import cv2 
import mediapipe as mp


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

cam = cv2.VideoCapture(2) 
cam.set(3, 1280)#setting width
cam.set(4, 720)#setting height
i = 0
with HandLandmarker.create_from_options(options) as landmarker:
    while cam.isOpened(): #Making sure the camera is open
        success, frame = cam.read()
        if not success: break

        # Pass frame to landmarker (converts to RGB internally with mp.Image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)
   
        if result.hand_world_landmarks:#makes sure our code only works when we actually get the data

            index_mcp = result.hand_world_landmarks[0][5]
            index_pip = result.hand_world_landmarks[0][6]
            index_tip = result.hand_world_landmarks[0][8]

            middle_pip = result.hand_world_landmarks[0][10]
            middle_tip = result.hand_world_landmarks[0][12]

            ring_pip = result.hand_world_landmarks[0][14]
            ring_tip = result.hand_world_landmarks[0][16]


            for hand in result.hand_landmarks:
                for lm in hand:
                    h, w, _ = frame.shape # Getting the exact values of the frame
                    cx, cy = int(lm.x * w), int(lm.y * h)# Helped with accuracy a lot but why do we use this? It's because mediapipe doesnt give a pixel coordinate for the hands it gives a value between 1-0 1 being one edge and 0 being the other so we multiply these values by our frames size so we can get the exact coordinates to draw on
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            middle_folded = (
                middle_tip.y > middle_pip.y 
            )
            knuckle_sandwich = (
                middle_tip.y > middle_pip.y and
                ring_tip.y > ring_pip.y and
                (index_tip.y - index_pip.y) > 0.02                
            )

            # Check knuckle sandwich first to prevent overlap error
            if knuckle_sandwich:
                print("you ready for a punch")
                
            elif middle_folded:
                  # Check for Point Up Index tip is significantly higher (more negative Y) than the joint
                if index_tip.y < index_pip.y and (index_pip.y - index_tip.y) > 0.03:
                    print(" Point up ->  increasing volume", i)
                    i+=1
        
                  # Check for Point Down Index tip is lower (more positive Y) than the joint
                elif index_tip.y > index_pip.y:
                    print("Pointing down -> decreasing volume", i)
                    i-=1

        cv2.imshow("Show Video", cv2.flip(frame, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'): break

cam.release()
cv2.destroyAllWindows()