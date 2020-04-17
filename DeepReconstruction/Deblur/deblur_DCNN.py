import numpy as np
import tensorflow as tf
from MainPage.ImageDistortion.add_blur import add_gaussian_blur
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import random

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# loading the CIFAR10 dataset using keras
(image_train, label_train), (image_test, label_test) = tf.keras.datasets.cifar10.load_data()
image_train = image_train / 255

image_validate = image_train[-int(image_train.shape[0] / 5):]
image_train = image_train[:-int(image_train.shape[0] / 5)]
image_test = image_test / 255


def in_mask_cifar(inp):
    x = inp + np.zeros(inp.shape)
    num_cores = multiprocessing.cpu_count()
    x = Parallel(n_jobs=num_cores)(delayed(blur_and_index)(inp[i], 7, i) for i in range(inp.shape[0]))
    # for i in range(inp.shape[0]):
    #     x[i] = add_horizontal_blur(inp[i], 5)
    #     print(i)
    # print(np.array(x).shape)
    return np.array(x)


def blur_and_index(image, kernel_size, index):
    print(index)
    return add_gaussian_blur(image, kernel_size, 5.0)


image_train_masked = in_mask_cifar(image_train)
image_test_masked = in_mask_cifar(image_test)
image_validate_masked = in_mask_cifar(image_validate)


# the U-Net model
def unet(pretrained_weights=None, input_size=(28, 28, 1)):
    # add in the layers in order
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv2)
    drop2 = Dropout(0.25)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = Conv2D(128, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = Dropout(0.25)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(256, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(256, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    up5 = Conv2D(128, 2, activation='sigmoid', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop4))
    merge5 = concatenate([drop3, up5], axis=3)
    conv5 = Conv2D(128, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(128, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(64, 2, activation='sigmoid', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([drop2, up6], axis=3)
    conv6 = Conv2D(64, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(64, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(32, 2, activation='sigmoid', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(32, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(32, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv7)

    conv7 = Conv2D(16, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv7)

    conv8 = Conv2D(input_size[2], 1, activation='sigmoid')(conv7)

    model = Model(inputs, conv8)

    model.compile(optimizer=Adam(lr=5e-4), loss='mse')
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# training for CIFAR dataset

EPOCHS = 10             # the number of epochs
BATCH_SIZE = 64        # the batch size
random.seed(666)

unet_cifar = unet(None, (32, 32, 3))
file_path = "cifar-{epoch:02d}-{val_loss:.3f}.hdf5"

try:
    unet_cifar.load_weights(file_path, by_name=True)
except:
    pass

checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = unet_cifar.fit(image_train_masked, image_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                         validation_data=(image_validate_masked, image_validate), callbacks=callbacks_list)

unet_cifar.save("cifar_unet_mse.h5")


# get CIFAR images (original, noisy, reconstructed)
img = image_test[7]
cimg = image_test_masked[7]
pred_img = unet_cifar.predict(image_test_masked[0:10])[7]

print("\n\n\nOriginal Image")
print(img)
print("\n\n\nDistorted Image")
print(cimg)
print("\n\n\nPredicted Image")
print(pred_img)


fig, axs = plt.subplots(3, 3)
axs[0, 0].imshow(np.clip(img, 0, 1))
axs[0, 1].imshow(np.clip(cimg, 0, 1))
axs[0, 2].imshow(np.clip(pred_img, -1, 1))

img = image_test[8]
cimg = image_test_masked[8]
pred_img = unet_cifar.predict(image_test_masked[0:12])[8]

axs[1, 0].imshow(np.clip(img, 0, 1))
axs[1, 1].imshow(np.clip(cimg, 0, 1))
axs[1, 2].imshow(np.clip(pred_img, 0, 1))

img = image_test[10]
cimg = image_test_masked[10]
pred_img = unet_cifar.predict(image_test_masked[0:12])[10]

axs[2, 0].imshow(np.clip(img, 0, 1))
axs[2, 1].imshow(np.clip(cimg, 0, 1))
axs[2, 2].imshow(np.clip(pred_img, 0, 1))

plt.show()
