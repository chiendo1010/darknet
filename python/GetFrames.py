from ctypes import *
import math
import random
import cv2
import glob
import time




VideosPath = '/mnt/DATA1/Data/VehiclesDetection/LayData_Lan_1/'
DesPath = '/mnt/DATA1/Data/VehiclesDetection/Frame/'

if __name__ == "__main__":
    ListPaths = [f for f in glob.glob(VideosPath+"/*.MOV")]
    for CurrentPath in ListPaths:
        CurrentFileName = CurrentPath.split('/')[-1]

        cap = cv2.VideoCapture(CurrentPath)
        
        TimeStartReadCurrentFile = time.time()
        
        CurrentFrameCount = 0
        SaveFilesCount = 0

        print("CurrentFileName: " + CurrentFileName)

        while True:
            (grabbed, frame) = cap.read()
            if not grabbed:
                print("Chien debug break break")
                print("Chien debug break break")
                print("Chien debug break break")
                break
            
            if(CurrentFrameCount % 60 == 0):
                cv2.imwrite(DesPath +CurrentFileName +"_"+ str(SaveFilesCount).zfill(7) + ".jpg", frame)
                SaveFilesCount += 1
                print(str(CurrentFileName) + " -- SaveFilesCount: " + str(SaveFilesCount))

            CurrentFrameCount += 1

