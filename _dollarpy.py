import os

import cv2

import mediapipe as mp
import dollarpy
from dollarpy import Recognizer, Template, Point

import numpy as np


def CreateTemplateFromVideo(videoPath, label):
    lms = GetLandMarksFromVideo(videoPath)
    return Template(label, lms)


def GetLandMarksFromVideo(videoPath):
    mp_drawing = mp.solutions.drawing_utils

    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(videoPath)

    landmarks = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Recolor image to RGZB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            if results.pose_landmarks == None:
                continue

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            lm = results.pose_landmarks.landmark
            # landmarks.append(Point(lm[11].x, lm[11].y))
            # landmarks.append(Point(lm[12].x, lm[12].y))
            # # landmarks.append(Point(lm[13].x, lm[13].y))
            # # landmarks.append(Point(lm[14].x, lm[14].y))
            # landmarks.append(Point(lm[24].x, lm[24].y))
            # landmarks.append(Point(lm[23].x, lm[23].y))
            # landmarks.append(Point(lm[25].x, lm[25].y))
            # landmarks.append(Point(lm[26].x, lm[26].y))
            # landmarks.append(Point(lm[28].x, lm[28].y))
            # landmarks.append(Point(lm[27].x, lm[27].y))
            #
            index = 0
            for landmark in results.pose_landmarks.landmark:
                if index == 25 or index == 26 :
                    landmarks.append(Point(landmark.x, landmark.y))

                index+=1
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()

        cv2.destroyAllWindows()

    return landmarks


#
# print (GetLandMarksFromVideo("VIDEOS/juggle/Juggle_1.mp4"))









directories = [
    "walk",
    "juggle",
    "sit"
]

templates = []
for directory in directories:
        for filename in os.listdir("VIDEOS/" + directory):
            path = os.path.join("VIDEOS/", directory, filename)
            print(path)
            # print (GetLandMarksFromVideo(path))
            print(directory)
            templates.append(CreateTemplateFromVideo(path, directory))


recognizer = Recognizer(templates)


print("testing for sit")

result = recognizer.recognize(GetLandMarksFromVideo(r"C://Users//GAMING//PycharmProjects//paper-Project1//sit_2.mp4"))
print(result)
print("testing for walk")
result = recognizer.recognize(GetLandMarksFromVideo(r"C:\\Users\\GAMING\\PycharmProjects\\paper-Project1\\Walk_3.mp4"))
print(result)
print("testing for juggle")
result = recognizer.recognize(GetLandMarksFromVideo(r"C:\\Users\\GAMING\\PycharmProjects\\paper-Project1\\Juggle_3.mp4"))
print(result)