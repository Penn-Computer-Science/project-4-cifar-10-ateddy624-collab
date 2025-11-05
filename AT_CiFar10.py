#imports
import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from tensorflow.keras.datasets import cifar10
#print out versions of libraries
#print("Tensorflow: ", tf.__version__ , "seaborn: " , sns.__version__ , "numpy: " , np.__version__ , "pandas: " , pd.__version__)

#load data set
#mnist = tf.keras.datasets.mnist
#create the training and testing data frames
#(training images, training labels), (testing images, testing labels)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#show the number of examples in each labeled set
#sns.countplot(x=y_train)
#show the plot
#plt.show()

#check to make sure there are NO values that are not a number (NaN)

print("Any NaN Training: ", np.isnan(x_train).any())
print("Any NaN Testing: ", np.isnan(x_test).any())

#tell the model what shape to expect
#(width, height, color chanels)
input_shape = (32, 32, 3)  #28x28 pixels, 1 color chanel (grayscale)

#reshape the training and testing data
x_train = x_train.astype('float32')/255.0

#convert our labels to be one-hot, rather than sparse
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

#show an example image from MNIST
#ex=random.randint(0,59999)
#plt.imshow(x_train[ex][:,:,0])
#plt.show()
#print(y_train[ex])

batch_size = 128
num_classes = 10
epochs = 6

#building the model...finally...
model=tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Dropout(0.75),
        #tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu', input_shape=input_shape),
        #tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
        
    ]
)

model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))


#plot out training and validation accuracy and loss
fig, ax = plt.subplots(2, 1)

ax[0].plot(history.history['loss'],  color = 'b', label="Training Loss")
ax[0].plot(history.history['val_loss'],  color = 'r', label="Validation Loss")
legend = ax[0].legend(loc='best', shadow=True)
ax[0].set_title("Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")

ax[1].plot(history.history['acc'],  color = 'b', label="Training Accuracy")
ax[1].plot(history.history['val_acc'],  color = 'r', label="Validation Accuracy")
legend = ax[1].legend(loc='best', shadow=True)
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")

plt.tight_layout()
plt.show()