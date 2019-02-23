import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=180,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     zoom_range=0.2)


datagen = ImageDataGenerator(**data_gen_args)

img_width, img_height = 256, 256

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory

train_samples = 100;

train_gen = datagen.flow_from_directory('data/img', target_size = (img_width, img_height), batch_size=1,
                          #save_to_dir='preview/trash', save_prefix='generated', save_format='jpeg'
                          )
test_gen = datagen.flow_from_directory('validation/img', target_size = (img_width, img_height), batch_size=1,
                          #save_to_dir='preview/not', save_prefix='generated', save_format='jpeg'
                          )


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])