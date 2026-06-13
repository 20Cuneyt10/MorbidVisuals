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

cam = cv2.VideoCapture(1) 
cam.set(3, 1280)#setting width
cam.set(4, 720)#setting height

with HandLandmarker.create_from_options(options) as landmarker:
    while cam.isOpened(): #Making sure the camera is open
        success, frame = cam.read()
        if not success: break

        # Pass frame to landmarker (converts to RGB internally with mp.Image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                for lm in hand:
                    h, w, _ = frame.shape # Getting the exact values of the frame
                    # Map normalized coordinates (0.0 - 1.0) to pixels on screen
                    #cx, cy = int(lm.x * 1280), int(lm.y * 720) was using this but it makes y axis a bit funny and it gets worser the further your hand is learned that this could be caused my a number of things but probably its bcus this part is hardcoded and assumes a linear increase in the real life size 
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        cv2.imshow("Show Video", cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cam.release()
cv2.destroyAllWindows()