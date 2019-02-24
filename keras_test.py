import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from keras import backend as K

from model_def import create_model

data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=180,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     zoom_range=0.2)


datagen = ImageDataGenerator(**data_gen_args)

img_width, img_height = 256, 256

batch_size = 25

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory

train_samples = 100;

train_gen = datagen.flow_from_directory('data/binary', target_size = (img_width, img_height), batch_size=1,
                          #save_to_dir='preview/trash', save_prefix='generated', save_format='jpeg',
                          class_mode='binary'
                          )
test_gen = datagen.flow_from_directory('validation/binary', target_size = (img_width, img_height), batch_size=1,
                          #save_to_dir='preview/not', save_prefix='generated', save_format='jpeg',
                          class_mode='binary'
                          )


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = create_model(input_shape)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(
        train_gen,
        steps_per_epoch=1200 // batch_size,
        epochs=100,
        validation_data=test_gen,
        validation_steps=150 // batch_size)
model.save_weights('third_try.h5')