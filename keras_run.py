from model_def import load_model
from keras import backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from PIL import Image


img_width, img_height = 256, 256

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = load_model(input_shape, "11.h5")

img = load_img('bottle_new.jpg').resize((img_width, img_height), Image.ANTIALIAS)  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 256, 256)
x = x.reshape((1,) + x.shape)

img_gen = ImageDataGenerator().flow(x)

for x in img_gen:
    print(model.predict(x))
    break


train_gen = ImageDataGenerator().flow_from_directory('data/binary', target_size = (img_width, img_height), batch_size=1,
                          #save_to_dir='preview/trash', save_prefix='generated', save_format='jpeg',
                          class_mode='binary'
                          )

inc = 0
correct = 0
for i in train_gen:
    prediction = model.predict(np.array(i[0]))
    print(str(prediction) + ", " + str(i[1]))
    if (prediction[0][0] >= 0.5 and i[1]):
        correct = correct + 1
    elif i[1] == 0:
        correct = correct + 1
    inc = inc + 1
    if (inc >= 300):
        break

print("Correct: " + str(correct))