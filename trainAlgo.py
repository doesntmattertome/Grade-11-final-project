from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import time

from DL3 import DLLayer, DLModel

# load the dataset

# at first I'll keep them as list an then I'll convert them to numpy arrays
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
        
# converting the lists to numpy arrays
X = np.array(X)
Y = np.array(Y)

size_of_image = X.shape[1]


print(X.shape)
print(Y.shape)

digits = 6
examples = Y.shape[0]
Y = Y.reshape(1, examples)
Y_new = np.eye(digits)[Y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)

Y_new = Y_new.T
print(X.shape)
print(Y_new.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_new, test_size=0.2, random_state=42)


X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.T
Y_test = Y_test.T

print(Y_train.shape)
print(X_train.shape)



np.random.seed(1)
model = DLModel()
model.add(DLLayer ("first layer", 64,(size_of_image,),"sigmoid","Xaviar", 0.007, "adaptive") )
model.add(DLLayer ("second layer", 128,(64,),"sigmoid","Xaviar", 0.007, "adaptive") )
model.add(DLLayer ("output player", 6,(128,),"softmax","Xaviar", 0.01))
model.compile("categorical_cross_entropy")
model.load_weights("SaveDir/cards")

#costs = model.train(X_train,Y_train,4500)
#model.save_weights("SaveDir/cards")

"""
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(1))
plt.show()
"""

print('Deep train accuracy')
model.confusion_matrix(X_train, Y_train)
print('Deep test accuracy')
model.confusion_matrix(X_test, Y_test)
