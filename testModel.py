from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image, ImageOps
import os

from DL3 import DLLayer, DLModel
import pygame
from pygame import surfarray
import random
import sys
from camera import Camera
from Button import Button
import numpy as np
import time
import copy

size_of_image = 46875

X = []
Y = []
directory = 'database'
# loop over the images in the "database" directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        # opening the image
        img = Image.open(f)
        # converting the image to numpy array
        img = np.array(img)
        # normalizing the image
        img = img / 255.0
        # flatten the image
        img = img.flatten()
        # adding the image to the list
        X.append(img)
        # adding the label to the list
        Y.append(filename.split('^')[0].split('image')[1].split("#")[0])
        break

X_temp = copy.deepcopy(X)



# create the model and load the weights
model = DLModel()
model.add(DLLayer ("first layer", 64,(size_of_image,),"sigmoid","He", 0.01) )
model.add(DLLayer ("first layer", 128,(64,),"sigmoid","He", 0.01) )
model.add(DLLayer ("output player", 6,(128,),"softmax","He", 0.05))
model.compile("categorical_cross_entropy")
model.load_weights("SaveDir/cards")


# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 1000, 563
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("testing model")

# camera
cam = Camera(WIDTH, HEIGHT)
rect = pygame.Rect(WIDTH/2 - 200, HEIGHT/2 - 200, 400, 400)
frame = cam.get_frame()

# text input

# But more customization possible: Pass your own font object
font = pygame.font.Font(None,30)

# grayscale function
def grayscale(img):
    arr = pygame.surfarray.array3d(img)
    #luminosity filter
    avgs = [[(r*0.298 + g*0.587 + b*0.114) for (r,g,b) in col] for col in arr]
    arr = np.array([[[avg,avg,avg] for avg in col] for col in avgs])
    return pygame.surfarray.make_surface(arr)

# predict the current frame
def normalize_image_and_send_to_model():
    global X
    global X_temp
    img = frame.subsurface(rect)
    img = grayscale(pygame.transform.scale(img, (125,125)))
    # convert the pygame surface to np array
    img = pygame.surfarray.array3d(img)
    
    img = np.array(img)
    # normalizing the image
    img = img / 255.0
    # flatten the image
    img = img.flatten()
    X_temp.append(img)
    X_temp = np.array(X_temp).T
    prediction = model.predict_percent(X_temp).T
    X_temp = copy.deepcopy(X)
    index_max = np.argmax(prediction[1])
    percent = round(prediction[1][index_max],3) * 100
    return f"Predicted: {index_max} with {percent}% confidence"

    return f""

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

def render_text(text, x, y):
        text_surface = font.render(text, True, RED)
        win.blit(text_surface, (x, y))

predicted_text = ""

prev = time.time()

x, y = 0, 0
run = True
# Main loop
while run:
    # Event loop
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            run = False
    
    
    # Draw
    win.fill(WHITE)
    frame = cam.get_frame()
    frame = pygame.transform.scale(frame, (WIDTH, HEIGHT))
    
    curr = time.time()
    if (curr - prev) > 0.2:
        prev = curr
        predicted_text = normalize_image_and_send_to_model()
    
    win.blit(frame, (x, y))
    pygame.draw.rect(win, (255, 0, 0), rect, 1)
    render_text(predicted_text,0,0)
    
    
    pygame.display.update()
    
pygame.quit()
sys.exit()

