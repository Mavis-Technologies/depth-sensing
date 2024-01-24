#!/usr/bin/env python3
import argparse
import queue
import sys
import threading
import numpy as np
import cv2
import pygame
from gtts import gTTS
import os
import roypy  # Adjust this import based on your camera SDK
from sample_camera_info import print_camera_info  # Adjust this based on your setup
from roypy_sample_utils import CameraOpener, add_camera_opener_options  # Adjust this based on your setup
from roypy_platform_utils import PlatformHelper  # Adjust this based on your setup
import time
import uuid

# Initialize Pygame for playing sounds
pygame.mixer.init()

# Text-to-Speech Function
def speak(text):
    # Generate a unique filename for each text-to-speech instance
    filename = f'temp_{uuid.uuid4()}.mp3'
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    # Wait for the playback to finish before deleting the file
    while pygame.mixer.music.get_busy():
        time.sleep(1)  # Check every 100ms

    os.remove(filename)

class MyListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.frame = 0
        self.done = False
        self.undistortImage = False
        self.lock = threading.Lock()
        self.once = False
        self.queue = q
        self.last_direction_time = 0


    def onNewData(self, data):
        p = data.npoints()
        self.queue.put(p)

    # Function to Process Depth Data and Give Directions
    def process_depth_data_and_give_directions(self, depth_image):
        
        height, width = depth_image.shape
        left_section = depth_image[:, :width // 3]
        center_section = depth_image[:, width // 3:2 * width // 3]
        right_section = depth_image[:, 2 * width // 3:]

        # Threshold for considering an obstacle 'close'
        obstacle_threshold = 0.7

        left_min = np.min(left_section[left_section > 0]) if np.any(left_section > 0) else float('inf')
        center_min = np.min(center_section[center_section > 0]) if np.any(center_section > 0) else float('inf')
        right_min = np.min(right_section[right_section > 0]) if np.any(right_section > 0) else float('inf')

        # Determine direction based on the farthest path
        if all(val <= obstacle_threshold for val in [left_min, center_min, right_min]):
            direction = 'stop'
        elif center_min >= max(left_min, right_min):
            direction = 'forward'
        elif left_min > right_min:
            direction = 'left'
        else:  # right_min is the largest
            direction = 'right'

        print(direction)

        # Speak or play a sound based on direction
        if direction == 'stop':
            speak(f"{direction}")
        else:
            speak(f"{direction}")

        

    def paint(self, data):
        # mutex to lock out changes to the distortion while drawing
        self.lock.acquire()

        depth = data[:, :, 2]

        current_time = time.time()
        if current_time - self.last_direction_time < 1.5:
            pass
        else:
            direction_thread = threading.Thread(target=self.process_depth_data_and_give_directions, args=(depth,))
            direction_thread.start()
        
            self.last_direction_time = current_time
        
        gray = data[:, :, 3]
        confidence = data[:, :, 4]
        # color = data[:, :, 0]

        zImage = np.zeros(depth.shape, np.float32)
        grayImage = np.zeros(depth.shape, np.float32)
        # colorImage = np.zeros(depth.shape, np.float32)

        # iterate over matrix, set zImage values to z values of data
        # also set grayImage adjusted gray values
        xVal = 0
        yVal = 0
        for x in zImage:        
            for y in x:
                if confidence[xVal][yVal]> 0:
                  zImage[xVal,yVal] = self.adjustZValue(depth[xVal][yVal])
                  grayImage[xVal,yVal] = self.adjustGrayValue(gray[xVal][yVal])
                yVal=yVal+1
            yVal = 0
            xVal = xVal+1

        zImage8 = np.uint8(zImage)
        grayImage8 = np.uint8(grayImage)
        # colorImage8 = np.uint8(colorImage)

        # apply undistortion
        if self.undistortImage:
            zImage8 = cv2.undistort(zImage8,self.cameraMatrix,self.distortionCoefficients)
            grayImage8 = cv2.undistort(grayImage8,self.cameraMatrix,self.distortionCoefficients)


        # Divide the depth frame into 5 vertical columns and calculate danger values
        column_width = depth.shape[1] // 5
        danger_values = []
        for i in range(5):
            # Extract each column
            column = depth[:, i * column_width: (i + 1) * column_width]

            # Find the minimum depth in the column (closest point)
            min_depth = np.min(column[np.nonzero(column)])

            # Map this depth to a danger_value from 1 to 100 (100 being closest)
            danger_value = 100 - (min_depth / 2.5 * 100)
            danger_values.append(danger_value)

            # Draw vertical lines to display each column
            cv2.line(zImage8, (i * column_width, 0), (i * column_width, zImage8.shape[0]), (0, 0, 255), 2)

            # Optionally display the danger value on each column
            cv2.putText(zImage8, str(round(danger_value)), (i * column_width + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


        # process_depth_data(zImage8)
        # print(zImage8)

        cv2.imshow('Depth', zImage8)

        self.lock.release()
        self.done = True

    def setLensParameters(self, lensParameters):
        # Construct the camera matrix
        # (fx   0    cx)
        # (0    fy   cy)
        # (0    0    1 )
        self.cameraMatrix = np.zeros((3,3),np.float32)
        self.cameraMatrix[0,0] = lensParameters['fx']
        self.cameraMatrix[0,2] = lensParameters['cx']
        self.cameraMatrix[1,1] = lensParameters['fy']
        self.cameraMatrix[1,2] = lensParameters['cy']
        self.cameraMatrix[2,2] = 1

        # Construct the distortion coefficients
        # k1 k2 p1 p2 k3
        self.distortionCoefficients = np.zeros((1,5),np.float32)
        self.distortionCoefficients[0,0] = lensParameters['k1']
        self.distortionCoefficients[0,1] = lensParameters['k2']
        self.distortionCoefficients[0,2] = lensParameters['p1']
        self.distortionCoefficients[0,3] = lensParameters['p2']
        self.distortionCoefficients[0,4] = lensParameters['k3']

    def toggleUndistort(self):
        self.lock.acquire()
        self.undistortImage = not self.undistortImage
        self.lock.release()

    # Map the depth values from the camera to 0..255
    def adjustZValue(self,zValue):
        clampedDist = min(2.5,zValue)
        newZValue = clampedDist / 2.5 * 255
        return newZValue
    
        # Map the gray values from the camera to 0..255
    def adjustGrayValue(self,grayValue):
        clampedVal = min(600,grayValue)
        newGrayValue = clampedVal / 600 * 255
        return newGrayValue

def main():
    platformhelper = PlatformHelper()
    parser = argparse.ArgumentParser(usage=__doc__)
    add_camera_opener_options(parser)
    options = parser.parse_args()

    opener = CameraOpener(options)
    try:
        cam = opener.open_camera()
    except:
        print("Could not open Camera Interface")
        sys.exit(1)

    try:
        # retrieve the interface that is available for recordings
        replay = cam.asReplay()
        print ("Using a recording")
        print ("Framecount : ", replay.frameCount())
        print ("File version : ", replay.getFileVersion())
    except SystemError:
        print ("Using a live camera")

    q = queue.Queue()
    l = MyListener(q)
    cam.registerDataListener(l)
    cam.startCapture()

    lensP = cam.getLensParameters()
    l.setLensParameters(lensP)

    while True:
        try:
            if len(q.queue) == 0:
                item = q.get(True, 1)
            else:
                for i in range(0, len(q.queue)):
                    item = q.get(True, 1)
        except queue.Empty:
            break
        else:
            l.paint(item)
            currentKey = cv2.waitKey(1)
            if currentKey == ord('d'):
                l.toggleUndistort()
            if currentKey == 27:
                break

    cam.stopCapture()
    print("Done")

if __name__ == "__main__":
    main()
