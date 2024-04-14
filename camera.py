# create a class that will handle the camera feed

import pygame
import cv2
import numpy as np
import sys

class Camera:
        
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)
        
    def get_frame(self):
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        return frame
    
    def release(self):
        self.cap.release()
        
    def __del__(self):
        self.release()
    
    