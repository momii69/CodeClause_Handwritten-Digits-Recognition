#!/usr/bin/env python
# coding: utf-8

# In[20]:


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, datasets, models
from tensorflow.keras.models import Sequential


# # Prepare Dataset
# 

# In[21]:


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

print("TRAIN IMAGES: ", train_images.shape)
print("TEST IMAGES: ", test_images.shape)


# # Create Model

# In[22]:


num_classes = 10
img_height = 28
img_width = 28

model = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='sigmoid')
])


# # Compile Model

# In[23]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[24]:


model.summary()


# # Train Model

# In[25]:


epochs = 10
history = model.fit(
  train_images, 
  train_labels,
  epochs = epochs
)


# # Visualize Training Results

# In[27]:


acc = history.history['accuracy']
loss=history.history['loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, loss, label='Loss')
plt.legend(loc='lower right')
plt.title('Training Accuracy and Loss')


# # Test Image

# In[28]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming train_images and model are already defined
image = (train_images[1]).reshape(1, 28, 28, 1)
model_pred = model.predict(image, verbose=0)
predicted_class = np.argmax(model_pred, axis=1)

plt.imshow(image.reshape(28, 28), cmap='gray')
print('Prediction of model: {}'.format(predicted_class[0]))


# In[29]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming train_images and model are already defined
image = (train_images[2]).reshape(1, 28, 28, 1)
model_pred = model.predict(image, verbose=0)
predicted_class = np.argmax(model_pred, axis=1)

plt.imshow(image.reshape(28, 28), cmap='gray')
print('Prediction of model: {}'.format(predicted_class[0]))


# # Test Multiple Image

# In[30]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming test_images and model are already defined
images = test_images[1:5]
images = images.reshape(images.shape[0], 28, 28)
print("Test images array shape: {}".format(images.shape))

for i, test_image in enumerate(images, start=1):
    org_image = test_image
    test_image = test_image.reshape(1, 28, 28, 1)
    prediction = model.predict(test_image, verbose=0)
    predicted_digit = np.argmax(prediction, axis=1)

    print("Predicted digit: {}".format(predicted_digit[0]))
    plt.subplot(220 + i)
    plt.axis('off')
    plt.title("Predicted digit: {}".format(predicted_digit[0]))
    plt.imshow(org_image, cmap='gray')

plt.show()


# # Save Model

# In[31]:


model.save("tf-cnn-model.h5")


# # Load Model

# In[32]:


loaded_model = models.load_model("tf-cnn-model.h5")


# In[33]:


import matplotlib.pyplot as plt

# Assuming loaded_model and train_images are already defined
image = train_images[2].reshape(1, 28, 28, 1)
model_pred = loaded_model.predict(image)
predicted_class = model_pred.argmax(axis=-1)

plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title('Prediction of model: {}'.format(predicted_class[0]))
plt.axis('off')
plt.show()


# In[ ]:




