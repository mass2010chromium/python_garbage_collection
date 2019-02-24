import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from keras import backend as K
from keras import optimizers
from keras.applications import VGG16

from model_def import create_model

data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=180,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     zoom_range=0.2)


datagen = ImageDataGenerator(**data_gen_args)

seed = 1
np.random.seed(seed)

img_width, img_height = 512, 512

batch_size = 80

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory

#train_samples = 100;

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

sgd = optimizers.SGD(lr=0.00025, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
model.fit_generator(
        train_gen,
        #steps_per_epoch=240 // batch_size,
        steps_per_epoch = 240,
        epochs=50,
        validation_data=test_gen,
        validation_steps = 80,
        #validation_steps=80 // batch_size
        )
model.save_weights('11.h5')