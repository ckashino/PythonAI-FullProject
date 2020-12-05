##########################################
##      MUST INSTALL MODULES FIRST      ##
##########################################

import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from PIL import Image, ImageFilter, ImageOps


SIZE = (256, 256)
CHOICE = {0: "grapeset", 1: "potatoset"}
INPUT = int(input("enter plant type"))
PATH = None  # need to implement this in the server

def toarray(path):  # defines a function that turns the photo (at location given) to a 3D array
    image = Image.open(path)  # opens photo
    image = image.convert('RGB')  # converts to rgb, removes alpha layer from jpgs
    imageres = image.resize((SIZE), Image.ANTIALIAS)  # resizes photo to size given in size variable
    blurred = imageres.filter(ImageFilter.GaussianBlur(radius=1.4))  # Blurs photo so model is based more on colours
    array = np.array(blurred)  # makes an array from the blurred, resized value.
    #  print(array.shape)
    # array = array.flatten()
    # array = array.reshape(1000000,)
    return array
def tomirrored(path):
    image = Image.open(path)  # opens photo
    image = image.convert('RGB')  # converts to rgb, removes alpha layer from jpgs
    imageres = image.resize((SIZE), Image.ANTIALIAS)  # resizes photo to size given in size variable
    # blurred = imageres.filter(ImageFilter.GaussianBlur(radius=1.4))  # Blurs photo so model is based more on colours
    rotated = ImageOps.flip(imageres)
    mirrored = ImageOps.mirror(rotated)
    array = np.array(mirrored)  # makes an array from the blurred, resized value.
    return array


model = models.load_model(CHOICE[INPUT] + "model.h5")  # loads the correct model based on input

test = np.empty((1, 256, 256, 3))
array = tomirrored(PATH)
test[0] = (array / 255.0)

print(model.predict(test))
result = np.argmax(model.predict(test))  # gets result from the model predictions (max value in predict arrray)

#  below are if statements for determining the plant health using dictionaries

if INPUT == 0:
    health = {0: "healthy", 1: "Black measles", 2: "Black Rot", 3: "Downy Mildew", 4:"Leaf Blight", 5: "Powdery Mildew"}
    print(health[result])
