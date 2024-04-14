# create a basic pygame program that opens a window

import pygame
from pygame_textinput import *
import random
import sys
from camera import Camera
from Button import Button
import numpy as np
# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 1000, 563
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("data gathering")

# camera
cam = Camera(WIDTH, HEIGHT)
rect = pygame.Rect(WIDTH/2 - 200, HEIGHT/2 - 200, 400, 400)
frame = cam.get_frame()

# text input
textinput = TextInputVisualizer()

# But more customization possible: Pass your own font object
font = pygame.font.Font(None,30)
# Create own manager with custom input validator
manager = TextInputManager(validator = lambda input: len(input) <= 10)




# grayscale function
def grayscale(img):
    arr = pygame.surfarray.array3d(img)
    #luminosity filter
    avgs = [[(r*0.298 + g*0.587 + b*0.114) for (r,g,b) in col] for col in arr]
    arr = np.array([[[avg,avg,avg] for avg in col] for col in avgs])
    return pygame.surfarray.make_surface(arr)

def take_picture():
    img_to_save = frame.subsurface(rect)
    img_to_save = grayscale(pygame.transform.scale(img_to_save, (125,125)))
    # a random number so that the image is not overwritten(most likely)
    random_id = random.randint(0,10000)
    
    pygame.image.save(img_to_save, f'database/image{textinput.value}^{random_id}.jpg')


take_picture_button = Button('save img',200,40,(WIDTH//2-100,20),5,take_picture)


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
x, y = 0, 0
run = True
# Main loop
while run:
    # Event loop
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            run = False
    
    #update text input
    textinput.update(events)   

    # Draw
    win.fill(WHITE)
    frame = cam.get_frame()
    
    
    frame = pygame.transform.scale(frame, (WIDTH, HEIGHT))
    win.blit(frame, (x, y))
    pygame.draw.rect(win, (255, 0, 0), rect, 1)
    take_picture_button.draw(win)
    
    win.blit(textinput.surface, (10, 10))
    
    pygame.display.update()
    
pygame.quit()
sys.exit()


