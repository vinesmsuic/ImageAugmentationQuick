from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import glob
import random
import numpy as np
import cv2

#https://stackoverflow.com/questions/43382045/keras-realtime-augmentation-adding-noise-and-contrast
def add_noise(img):
    #Add random noise to an image
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img
# usage: preprocessing_function=add_noise

#https://github.com/yu4u/cutout-random-erasing/blob/master/cifar10_resnet.py
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser
# usage: preprocessing_function = get_random_eraser(v_l=0, v_h=255)

def random_exposure_and_saturation(img):
    # Random exposure and saturation (0.9 ~ 1.1 scale)
    rand_s = random.uniform(0.9, 1.1)
    rand_v = random.uniform(0.9, 1.1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    tmp = np.ones_like(img[:, :, 1]) * 255
    img[:, :, 1] = np.where(img[:, :, 1] * rand_s > 255, tmp, img[:, :, 1] * rand_s)
    img[:, :, 2] = np.where(img[:, :, 2] * rand_v > 255, tmp, img[:, :, 2] * rand_v)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # Centering helps normalization image (-1 ~ 1 value)
    return img / 127.5 - 1
# usage: preprocessing_function = random_exposure_and_saturation

def genImg(datagen, image_path, gen_count, save_to_folder):
    img = load_img(image_path)  
    x = img_to_array(img)  # creating a Numpy array with shape (3, width, height)
    x = x.reshape((1,) + x.shape)  # converting to a Numpy array with shape (1, 3, width, height)

    image_info = image_path.split(".")
    if(len(image_info)!= 2):
        print("Error: make sure your image_path has a proper naming.")
        return

    i_prefix = image_info[0]
    i_format = image_info[1]

    i = 0
    for batch in datagen.flow(x, save_to_dir=save_to_folder, save_prefix=i_prefix, save_format=i_format):
        i += 1
        if i > gen_count:
            break
#usage: genImg(datagen, image_path='Z.jpg', gen_count=10, save_to_folder='sample_data')

def genImgInFolder(input_files, gen_count, save_folder):
    loopfiles = input_files
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    print("====== Starting augmentation =======")    
    for filepath in glob.iglob(loopfiles):
        genImg(datagen, image_path=filepath, gen_count=gen_count, save_to_folder=save_folder)
        print(filepath, "has finished augmentation")  
    print("====== Finished augmentation =======")

#usage: genImgInFolder(input_files='*.jpg', gen_count=10, save_folder='maxFinal')


#============================================================================================
# Customize your datagen here
#https://keras.io/api/preprocessing/image/
datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.5,1.0],
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant',
        preprocessing_function = get_random_eraser(v_l=0, v_h=255))
#============================================================================================


#============================================================================================
genImgInFolder(input_files='*.jpg', gen_count=5, save_folder='generated_img')
#============================================================================================