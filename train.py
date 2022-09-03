import os

import cv2

import mediapipe as mp
import dollarpy
from dollarpy import Recognizer, Template, Point

import numpy as np


def CreateTemplateFromVideo(videoPath, label,target_lms):
    lms = GetLandMarksFromVideo(videoPath,target_lms)
    return Template(label, lms)

def GetLandMarksFromFrames(frames,target_lms,activites_number):
    mp_drawing = mp.solutions.drawing_utils

    mp_pose = mp.solutions.pose
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (0, 0, 0)

    # Line thickness of 2 px
    thickness = 2
    landmarks = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for image in frames:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            if results.pose_landmarks == None:
                continue

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            lm = results.pose_landmarks.landmark

            for target in target_lms:
                landmarks.append(Point(lm[target].x, lm[target].y))
                image = cv2.putText(image, 'shots: ' + str(activites_number[0]), (20,20), font,
                                    fontScale, color, thickness, cv2.LINE_AA)
                image = cv2.putText(image, 'juggles: ' + str(activites_number[1]), (20, 50), font,
                                    fontScale, color, thickness, cv2.LINE_AA)
                cv2.imshow('Mediapipe Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    return landmarks



def GetFramesOfVideo(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    print ("FRAMES COUNT OF TEST :" ,frames)
    return frames


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

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


    return landmarks


#
# print (GetLandMarksFromVideo("VIDEOS/juggle/Juggle_1.mp4"))


def getactivityduration(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    seconds = int(duration%60)
    print(seconds)
    return seconds
    cap.release()








directories = [
    "shoot","juggle"
]

results = []

i=0
videosPaths =os.listdir("TINYVIDEOS/" + "shoot")
print(i, "/", len(videosPaths))
testPath = videosPaths[i]
testPath = os.path.join("TINYVIDEOS/", "shoot", testPath)
templates = []
targetLandmarks = [25,26]

for directory in directories:
    for idx, filename in enumerate(videosPaths):
        if idx == i:
            continue
        path = os.path.join("TINYVIDEOS/", directory, filename)
        print (directory, " " , filename)
        # print (GetLandMarksFromVideo(path))
        getactivityduration(path)
        templates.append(CreateTemplateFromVideo(path, directory, targetLandmarks))

recognizer = Recognizer(templates)

print("testing for shoot")


duration = getactivityduration("test_videos/shoot.mp4")
framesOfVideo = GetFramesOfVideo("test_videos/shoot.mp4")

print (len(framesOfVideo))
fps = int (len(framesOfVideo)/duration)

print ("fps are " , fps)




# result = recognizer.recognize(GetLandMarksFromVideo(testPath, range(10,33)))
# print(result)
# results.append(result)


#testing

step = fps * 1
start = 0
shoot =0
juggle = 0
while start < len(framesOfVideo)-step:
    end = start + step
    result = recognizer.recognize(GetLandMarksFromFrames(framesOfVideo[start:end], targetLandmarks,[shoot,juggle]))
    print(result)
    if (result[0] == 'shoot'): shoot+=1
    if (result[0] == 'juggle'): juggle+=2
    start += step


print ("shoots taken :" ,shoot)
print ("juggles done :" ,juggle)





