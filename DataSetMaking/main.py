from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
import numpy as np
import os
import matplotlib.pyplot as plt

folderpath = "grapeset"

size = (256, 256)  # a variable to change the size of photo resize

def toarray(path):  # defines a function that turns the photo (at location given) to a 3D array
    image = Image.open(path)  # opens photo
    image = image.convert('RGB')  # converts to rgb, removes alpha layer from jpgs
    imageres = image.resize((size), Image.ANTIALIAS)  # resizes photo to size given in size variable
    #blurred = imageres.filter(ImageFilter.GaussianBlur(radius=1.4))  # Blurs photo so model is based more on colours
    array = np.array(imageres)  # makes an array from the blurred, resized value.
    #  print(array.shape)
    # array = array.flatten()
    # array = array.reshape(1000000,)
    return array
def tomirrored(path):
    image = Image.open(path)  # opens photo
    image = image.convert('RGB')  # converts to rgb, removes alpha layer from jpgs
    imageres = image.resize((size), Image.ANTIALIAS)  # resizes photo to size given in size variable
    # blurred = imageres.filter(ImageFilter.GaussianBlur(radius=1.4))  # Blurs photo so model is based more on colours
    rotated = ImageOps.flip(imageres)
    imgplot = plt.imshow(rotated)
    plt.show()
    array = np.array(rotated)  # makes an array from the blurred, resized value.
    return array


# tomirrored("photos\\0 (1).jpeg")


photocount = 0
for name in os.listdir(folderpath):
    photocount += 1

print(photocount)
i = 0
combined = np.empty((photocount, 256, 256, 3), dtype='int32')

yvals = []

for name in os.listdir(folderpath):  # a loop that goes through each photo in the folder
    array = toarray(folderpath + "\\" + str(name))  # turn the photo into an array by using file name
    # print(array)
    combined[i] = array
    # print(combined[i])
    # plt.imshow(combined[i], interpolation='nearest')
    # plt.show()
    i += 1
    if (i % 100) == 0:
        print(i)
    list(name)  # convert the name of the a list
    y = int(name[0])  # Take the first character of the name, which is the label
    yvals.append(y)  # Add the label value to the yvals array.

print(combined.shape)


np.save(folderpath + "dataset.npy", combined)
np.save(folderpath + "labels.npy", yvals)

