import math

import numpy as np
import tensorflow as tf
import keras.backend as kb
from MainPage.ImageDistortion.add_patterns import add_random_patterns
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
# from tensorflow.keras.callbacks import *
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random
import gc


def keras_mse_l1_loss(y_actual, y_predicted):
    loss = kb.mean(kb.sum(kb.square(y_actual - y_predicted))) + kb.mean(kb.sum(kb.abs(y_predicted))) * 0.1
    return loss


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

    model.compile(optimizer=Adam(lr=5e-4), loss=keras_mse_l1_loss)
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def unet_2(pretrained_weights=None, input_size=(28, 28, 1)):
    # add in the layers in order
    inputs = Input(input_size)
    conv1 = Dropout(0.8)(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs))
    conv1 = Dropout(0.5)(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Dropout(0.5)(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1))
    conv2 = Dropout(0.5)(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Dropout(0.5)(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2))
    conv3 = Dropout(0.5)(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Dropout(0.5)(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3))
    conv4 = Dropout(0.5)(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4))

    up5 = Conv2DTranspose(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv4))
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Dropout(0.5)(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5))
    conv5 = Dropout(0.5)(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5))

    up6 = Conv2DTranspose(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Dropout(0.5)(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6))
    conv6 = Dropout(0.5)(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6))

    up7 = Conv2DTranspose(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Dropout(0.5)(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7))
    conv7 = Dropout(0.5)(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7))

    # conv7 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    conv8 = Conv2D(input_size[2], 1, activation='relu')(conv7)

    model = Model(inputs, conv8)

    model.compile(optimizer=Adam(lr=1e-3), loss='mse')
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def integrate_changes(distorted_image, recovered_image, threshold):
    proposed_image = np.copy(distorted_image)
    for i in range(recovered_image.shape[0]):
        for j in range(recovered_image.shape[1]):
            if np.linalg.norm(recovered_image[i, j] - proposed_image[i, j]) < threshold:
                proposed_image[i, j] = distorted_image[i, j]
            else:
                proposed_image[i, j] = recovered_image[i, j]
            # if np.linalg.norm(recovered_image - proposed_image) < threshold:
            #     proposed_image[i, j] = distorted_image[i, j]
            # else:
            #     proposed_image[i, j] = recovered_image[i, j]
    return proposed_image


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

# image_train = np.concatenate(datagen.flow(image_train, None, batch_size=64), axis=0)
# print(image_train)

image_validate = image_train[-int(image_train.shape[0] / 5):]
image_train = image_train[:-int(image_train.shape[0] / 5)]
image_test = image_test / 255

print(image_validate.shape[0])
print(image_train.shape[0])
print(image_test.shape[0])


def in_mask_cifar(inp):
    x = inp + np.zeros(inp.shape)
    for i in range(inp.shape[0]):
        x[i] = add_random_patterns(inp[i], 0.2, 2, 2, 2)
    return np.array(x)


SEED = 666
image_test_masked = in_mask_cifar(image_test)
unet_cifar = unet(None, (32, 32, 3))

for i in range(30):
    print("======================================== Trial ", i + 1, " ========================================")

    train_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=45,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range=0.2,
        zoom_range=0.4,
        horizontal_flip=True
    )

    validate_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=45,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range=0.2,
        zoom_range=0.4,
        horizontal_flip=True
    )

    train_datagen.fit(image_train)
    validate_datagen.fit(image_validate)

    image_train_augment = train_datagen.flow(image_train, None, batch_size=40000)
    image_validate_augment = train_datagen.flow(image_validate, None, batch_size=10000)

    image_train = image_train_augment[0]
    image_validate = image_validate_augment[0]

    image_train_masked = in_mask_cifar(image_train)
    image_validate_masked = in_mask_cifar(image_validate)

    EPOCHS = 4  # the number of epochs
    BATCH_SIZE = 64  # the batch size
    random.seed(666)
    # file_path = "cifar-{epoch:02d}-{val_loss:.3f}.hdf5"
    #
    # try:
    #     unet_cifar.load_weights(file_path, by_name=True)
    # except:
    #     pass
    #
    # checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # callbacks_list = [checkpoint]

    unet_cifar.fit(image_train_masked, image_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                   validation_data=(image_validate_masked, image_validate))
    gc.collect()

    if i % 5 == 4:
        # get CIFAR images (original, noisy, reconstructed)
        img = image_test[19]
        cimg = image_test_masked[19]
        pred_img = unet_cifar.predict(image_test_masked[0:20])[19]
        integrated_img = integrate_changes(cimg, pred_img, 0.6)

        print("\n\n\nOriginal Image")
        print(img)
        print("\n\n\nDistorted Image")
        print(cimg)
        print("\n\n\nPredicted Image")
        print(pred_img)

        fig, axs = plt.subplots(3, 4)
        axs[0, 0].imshow(np.clip(img, 0, 1))
        axs[0, 1].imshow(np.clip(cimg, 0, 1))
        axs[0, 2].imshow(np.clip(pred_img, -1, 1))
        axs[0, 3].imshow(np.clip(integrated_img, -1, 1))

        img = image_test[16]
        cimg = image_test_masked[16]
        pred_img = unet_cifar.predict(image_test_masked[0:20])[16]
        integrated_img = integrate_changes(cimg, pred_img, 0.6)

        axs[1, 0].imshow(np.clip(img, 0, 1))
        axs[1, 1].imshow(np.clip(cimg, 0, 1))
        axs[1, 2].imshow(np.clip(pred_img, 0, 1))
        axs[1, 3].imshow(np.clip(integrated_img, -1, 1))

        img = image_test[22]
        cimg = image_test_masked[22]
        pred_img = unet_cifar.predict(image_test_masked[0:25])[22]
        integrated_img = integrate_changes(cimg, pred_img, 0.6)

        axs[2, 0].imshow(np.clip(img, 0, 1))
        axs[2, 1].imshow(np.clip(cimg, 0, 1))
        axs[2, 2].imshow(np.clip(pred_img, 0, 1))
        axs[2, 3].imshow(np.clip(integrated_img, -1, 1))

        plt.show()

        unet_cifar.save("cifar_unet_temp_" + str(i) + ".h5")


unet_cifar.save("cifar_unet.h5")



