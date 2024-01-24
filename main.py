#!/usr/bin/python3

# Copyright (C) 2019 Infineon Technologies & pmdtechnologies ag
#
# THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
# KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
# PARTICULAR PURPOSE.

"""This sample shows how to use openCV on the depthdata we get back from either a camera or an rrf file.
The Camera's lens parameters are optionally used to remove the lens distortion and then the image is displayed using openCV windows.
Press 'd' on the keyboard to toggle the distortion while a window is selected. Press esc to exit.
"""

import argparse
try:
    from roypypack import roypy  # package installation
except ImportError:
    import roypy  # local installation
import queue
import sys
import threading
from sample_camera_info import print_camera_info
from roypy_sample_utils import CameraOpener, add_camera_opener_options
from roypy_platform_utils import PlatformHelper

import numpy as np
import cv2
import threading
import pygame

# Initialize Pygame Mixer
pygame.mixer.init()
sound = pygame.mixer.Sound('center_sound.mp3')  # Replace with your sound file

def play_alert_sound(volume):
    print(volume)
    pygame.mixer.Sound.set_volume(sound, volume)
    if not pygame.mixer.get_busy():
        sound.play()

def calculate_volume(avg_intensity):
    # Assuming intensity ranges from 0 (close) to 255 (far)
    # Invert the intensity to get the volume (closer objects should have higher volume)
    return (255 - avg_intensity) / 255

def process_depth_data(depth_image):
    avg_intensity = np.mean(depth_image)
    volume = calculate_volume(avg_intensity)

    if volume > 0.1:  # Set a threshold to avoid beeping for very far objects
        alert_thread = threading.Thread(target=play_alert_sound, args=(volume,))
        alert_thread.start()

class MyListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.frame = 0
        self.done = False
        self.undistortImage = False
        self.lock = threading.Lock()
        self.once = False
        self.queue = q

    def onNewData(self, data):
        p = data.npoints()
        self.queue.put(p)
        

    def paint (self, data):
        """Called in the main thread, with data containing one of the items that was added to the
        queue in onNewData.
        """
        # mutex to lock out changes to the distortion while drawing
        self.lock.acquire()
        
        depth = data[:, :, 2]

        
        max_distance = 2.0  # Example maximum distance for full volume


        # Divide the depth frame into 5 vertical columns and calculate danger values
        column_width = depth.shape[1] // 5
        print(depth.shape)
        danger_values = []
        for i in range(5):
            # Extract each column
            column = depth[:, i * column_width: (i + 1) * column_width]

            # Find the minimum depth in the column (closest point)
            min_depth = np.min(column[np.nonzero(column)])

            # Map this depth to a danger_value from 1 to 100 (100 being closest)
            danger_value = 100 - (min_depth / 2.5 * 100)
            danger_values.append(danger_value)

        # Display or handle the danger values as needed
        print("Danger Values: ", danger_values)
        
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


        depth = data[:, :, 2]
        zImage8 = np.uint8(zImage)

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

        # Display or handle the danger values as needed
        print("Danger Values: ", danger_values)

        process_depth_data(zImage8)
        print(zImage8)

        cv2.imshow('Depth', zImage8)
        # cv2.imshow('Gray',grayImage8)
        # cv2.imshow('color',data[:,:,4])

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
    

def main ():

    # Set the available arguments
    platformhelper = PlatformHelper()
    parser = argparse.ArgumentParser (usage = __doc__)
    add_camera_opener_options (parser)
    options = parser.parse_args()
   
    opener = CameraOpener (options)

    try:
        cam = opener.open_camera ()
    except:
        print("could not open Camera Interface")
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

    process_event_queue (q, l)

    cam.stopCapture()
    print("Done")

def process_event_queue (q, painter):

    while True:
        try:
            
            # try to retrieve an item from the queue.
            # this will block until an item can be retrieved
            # or the timeout of 1 second is hit
            if len(q.queue) == 0:
                item = q.get(True, 1)
            else:
                for i in range (0, len (q.queue)):
                    item = q.get(True, 1)
        except queue.Empty:
            # this will be thrown when the timeout is hit
            break
        else:
            painter.paint(item)
            # waitKey is required to use imshow, we wait for 1 millisecond
            currentKey = cv2.waitKey(1)
            if currentKey == ord('d'):
                painter.toggleUndistort()
            # close if escape key pressed
            if currentKey == 27: 
                break

if (__name__ == "__main__"):
    main()
