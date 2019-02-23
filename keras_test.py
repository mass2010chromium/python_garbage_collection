import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=180,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     zoom_range=0.2)


datagen = ImageDataGenerator(**data_gen_args)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow_from_directory('data/', target_size = (256, 256), batch_size=1,
                          save_to_dir='preview', save_prefix='generated', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely