import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For webcam input:
all_cam_ids = [0, 2, 3, 4]
all_caps = [cv2.VideoCapture(c) for c in all_cam_ids]

with mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    finish = False
    while not finish:
        all_kps = []
        all_imgs = []
        for cam_id, cap in zip(all_cam_ids, all_caps):
            if not cap.isOpened():
                continue

            ret, img = cap.read()
            if not ret:
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img)

            if results.pose_landmarks is not None:
                kps = [(l.x, l.y, l.z, l.visibility) for l in results.pose_landmarks.landmark]
                kps = np.array(kps)
                assert kps.shape == (33, 4), str(kps.shape)
            else:
                kps = None
            all_kps.append(kps)
            
            # # Draw the pose annotation on the image.
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
            all_imgs.append(img)
            
        show_img = cv2.vconcat([
            cv2.hconcat(all_imgs[:2]),
            cv2.hconcat(all_imgs[2:]),
        ])
        cv2.imshow(f"Cameras", show_img)
        if cv2.waitKey(1) == ord('q'):
            finish = True
        

    for cap in all_caps:
        cap.release()
