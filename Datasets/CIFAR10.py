import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(image_train, label_train), (image_test, label_test) = tf.keras.datasets.cifar10.load_data()
image_train = image_train / 255

image_validate = image_train[-int(image_train.shape[0] /5):]
image_train = image_train[:-int(image_train.shape[0] /5)]
image_test = image_test / 255

x_train = image_train
n_rows = x_train.shape[0]
n_cols = x_train.shape[1]
mean = 0
stddev = 0.01
noise = np.random.normal(mean, stddev, image_train.shape)

x_train_noisy = x_train + noise


# fig, axs = plt.subplots(1, 3, dpi=90)

# for i in range(3):
#   axs[i].imshow(x_train[i], cmap="gray")

figure, axes = plt.subplots(*(4, 6), figsize=(18, 12))
figure.set_size_inches(18, 12)
random_images = np.random.randint(0, len(x_train_noisy), (4, 6)).flatten()
images = np.reshape(x_train_noisy[random_images], (-1, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
labels = label_train[random_images]

for picax, idx, img, label in zip(axes.flat, random_images, images, labels):
    picax.set_title("Image %d : %d" % (idx, label))
    picax.imshow(img)
    picax.set_frame_on(False)
    picax.set_axis_off()

plt.tight_layout()
plt.show()

