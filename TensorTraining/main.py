import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter

datasetname = "grapeset"

SEED = 54649843513

tf.random.set_seed(SEED)

X = np.load(datasetname + "dataset.npy")
y = np.load(datasetname + "labels.npy")
X = X / 255.0

# print(y[105])
# plt.imshow(X[105], interpolation='nearest')
# plt.show()

model = models.Sequential()

# model.add(layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(256, 256, 3)))
# model.add(layers.MaxPooling2D())
# model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
# model.add(layers.MaxPooling2D())
# model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
# model.add(layers.MaxPooling2D())
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(6))


model.add(layers.Conv2D(32, 3, padding='same', activation='relu', strides=(1, 1), input_shape=(256, 256, 3)))
model.add(layers.Conv2D(32, 3, padding='same', activation='relu', strides=(2, 2)))
model.add(layers.Conv2D(64, 3, padding='same', activation='relu', strides=(1, 1)))
model.add(layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2)))
model.add(layers.Conv2D(128, 3, padding='same', activation='relu', strides=(1, 1)))
model.add(layers.Conv2D(128, 3, padding='same', activation='relu', strides=(2, 2)))
model.add(layers.Conv2D(256, 3, padding='same', activation='relu', strides=(1, 1)))
model.add(layers.Conv2D(256, 3, padding='same', activation='relu', strides=(2, 2)))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(X, y, epochs=40)

model.summary()

size = (256, 256)

def toarray(path):
    image = Image.open(path)
    image = image.convert('RGB')
    imageres = image.resize((size), Image.ANTIALIAS)
    # blurred = imageres.filter(ImageFilter.GaussianBlur(radius=1.4))
    array = np.array(imageres)
    print(array.shape)
    #array = array.flatten()
    #array = array.reshape(1000000,)
    return array

test = np.empty((1, 256, 256, 3))

array = toarray('grape.jpg')
test[0] = array / 255.0

# plt.imshow(test[0], interpolation='nearest')
# plt.show()

print(model.predict(test))
result = np.argmax(model.predict(test))
print(result)

array = toarray('leafblight.jpg')
test[0] = array / 255.0

# plt.imshow(test[0], interpolation='nearest')
# plt.show()

print(model.predict(test))
result = np.argmax(model.predict(test))
print(result)

# health = {0: "Healthy", 1: "Early Blight", 2: "Late Blight", 3: "Leaf Mold", 4: "Bacterial Spot", 5: "Mosaic Leafspot", 6: "Septoria leafspot", 7: "Two Spotted Spider Mite", 8: "Yellow Leaf Curl Virus"}
#
# print(health[result])

model.save(datasetname + "model.h5")


