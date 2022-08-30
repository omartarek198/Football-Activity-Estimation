import os

import cv2

import mediapipe as mp
import dollarpy
from dollarpy import Recognizer, Template, Point

import numpy as np


def CreateTemplateFromVideo(videoPath, label,target_lms):
    lms = GetLandMarksFromVideo(videoPath,target_lms)
    return Template(label, lms)


def GetLandMarksFromVideo(videoPath,target_lms):
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
            # landmarks.append(Point(lm[13].x, lm[13].y))
            # landmarks.append(Point(lm[14].x, lm[14].y))
            # landmarks.append(Point(lm[24].x, lm[24].y))
            # landmarks.append(Point(lm[23].x, lm[23].y))
            # landmarks.append(Point(lm[25].x, lm[25].y))
            # landmarks.append(Point(lm[26].x, lm[26].y))
            # landmarks.append(Point(lm[27].x, lm[27].y))
            # landmarks.append(Point(lm[28].x, lm[28].y))

            index = 0

            for target in target_lms:
                landmarks.append(Point(lm[target].x,lm[target].y))


    return landmarks


#
# print (GetLandMarksFromVideo("VIDEOS/juggle/Juggle_1.mp4"))









directories = [
    "shoot"
]


videosPaths =os.listdir("TINYVIDEOS/" + "shoot")
accuracies = []

for idx, accuracy in enumerate(accuracies):
    print(idx, " = ", accuracy)

for k in range(11,28):
    results = []
    sum = 0
    print ("Extracting " ,k ,"/28")
    for i in range(0, len(videosPaths)):
        print(i, "/", len(videosPaths))
        testPath = videosPaths[i]
        testPath = os.path.join("TINYVIDEOS/", "shoot `1", testPath)
        templates = []

        for directory in directories:
            for idx, filename in enumerate(videosPaths):
                if idx == i:
                    continue
                path = os.path.join("TINYVIDEOS/", directory, filename)
                # print (GetLandMarksFromVideo(path))
                templates.append(CreateTemplateFromVideo(path, directory, [k]))

        recognizer = Recognizer(templates)

        print("testing for shoot")

        result = recognizer.recognize(GetLandMarksFromVideo(testPath,[k]))
        print(result)
        results.append(result)

    for result in results:
        print(result)
        sum+=result[1]
    accuracies.append(sum/len(results))

for idx, accuracy in enumerate(accuracies):
    print(idx, " = ", accuracy)



