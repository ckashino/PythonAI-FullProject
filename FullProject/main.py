##########################################
##      MUST INSTALL MODULES FIRST      ##
##########################################
##################################################################
##      TO USE REPLACE PATH = NONE WITH DIRECTORY TO PHOTO      ##
##################################################################


import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from PIL import Image, ImageFilter, ImageOps


SIZE = (256, 256)
CHOICE = {0: "grapeset", 1: "potatoset", 2: "tomatoset", 3: "ribbonplantset", 4: "nerveplantset", 5: "bellpepperset", 6: "fernset", 7: "basilset", 8: "orangeset", 9: "spinichset"}
INPUT = int(input("enter plant type "))
PATH = None  # need to implement this in the server

model = models.load_model(CHOICE[INPUT] + "model.h5")  # loads the correct model based on input

def toarray(path):  # defines a function that turns the photo (at location given) to a 3D array
    image = Image.open(path)  # opens photo
    image = image.convert('RGB')  # converts to rgb, removes alpha layer from jpgs
    imageres = image.resize((SIZE), Image.ANTIALIAS)  # resizes photo to size given in size variable
    # blurred = imageres.filter(ImageFilter.GaussianBlur(radius=1.4))  # Blurs photo so model is based more on colours
    array = np.array(imageres)  # makes an array from the blurred, resized value.
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
def toresults(photopath):
    test = np.empty((1, 256, 256, 3))
    array = toarray(photopath)
    test[0] = (array / 255.0)
    print(model.predict(test))
    result = np.argmax(model.predict(test))  # gets result from the model predictions (max value in predict arrray)
    return result
def toresultsalt(photopath):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(photopath)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    result = np.argmax(model.predict(data))
    return result

#  below are if statements for determining the plant health using dictionaries

if INPUT == 0:
    HealthValue = toresults(PATH)
    health = {0: "healthy", 1: "Black measles", 2: "Black Rot", 3: "Downy Mildew", 4: "Leaf Blight", 5: "Powdery Mildew"}
    print(health[HealthValue])

if INPUT == 1:
    HealthValue = toresults(PATH)
    health = {0: "healthy", 1: "Blight"}
    print(health[HealthValue])

if INPUT == 2:
    HealthValue = toresultsalt(PATH)
    health = {0: "healthy", 1: "Early Blight", 2: "Late Blight", 3: "Yellow Leaf Curl", 4: "Bacterial Spot"}
    print(health[HealthValue])

if INPUT == 3:
    HealthValue = toresultsalt(PATH)
    health = {0: "healthy", 1: "Brown Tips", 2: "Red Spider Mites", 3: "Scale Insects"}
    print(health[HealthValue])

if INPUT == 4:
    HealthValue = toresultsalt(PATH)
    health = {0: "healthy", 1: "Drooping Leaves", 2: "Shriveling Leaves", 3: "yellowing"}
    print(health[HealthValue])

if INPUT == 5:
    HealthValue = toresultsalt(PATH)
    health = {0: "healthy", 1: "Bacterial Spot", 2: "Cercospora", 3: "Leaf miner", 4: "Powdery Milder"}
    print(health[HealthValue])

if INPUT == 6:
    HealthValue = toresultsalt(PATH)
    health = {0: "healthy", 1: "Leaf tip burn"}
    print(health[HealthValue])

if INPUT == 7:
    HealthValue = toresultsalt(PATH)
    health = {0: "healthy", 1: "Cercospora Leaf Spot", 2: "Downy Mildew", 3: "Fusarium Wilt"}
    print(health[HealthValue])

if INPUT == 8:
    HealthValue = toresultsalt(PATH)
    health = {0: "healthy", 1: "Citrus Canker", 2: "Citrus Scab", 3: "Greasy Spot", 4: "Melanose"}
    print(health[HealthValue])

if INPUT == 9:
    HealthValue = toresultsalt(PATH)
    health = {0: "healthy", 1: "Anthracnose", 2: "Cladosporium", 3: "Downleaf"}
    print(health[HealthValue])
