import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (_, _) = cifar10.load_data()

# Keep only car images
car_images = x_train[y_train.flatten() == 1]

# Load gun images from file
gun_images = []
for i in range(1, 5):  
    img = load_img(f'gun_image_{i}.jpg', target_size=(32, 32))
    img_array = img_to_array(img)
    gun_images.append(img_array)

# Create labels for car and gun images
car_labels = np.ones(len(car_images))
gun_labels = np.zeros(len(gun_images))

# Concatenate car and gun images and labels
x_train = np.concatenate((car_images, gun_images), axis=0)
y_train = np.concatenate((car_labels, gun_labels), axis=0)

# Normalize pixel values to range [0, 1]
x_train = x_train.astype('float32') / 255.0

# Convert labels to categorical format
y_train = to_categorical(y_train)

# Define and compile the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)

# Save the model
model.save('car_gun_classifier.h5')