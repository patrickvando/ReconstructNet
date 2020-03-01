import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255.

n_rows = x_train.shape[0]
n_cols = x_train.shape[1]
mean = 0.5
stddev = 0.3
noise = np.random.normal(mean, stddev, (n_rows, n_cols))

x_train_noisy = x_train + noise

# fig, axs = plt.subplots(1, 3, dpi=90)

# for i in range(3):
#   axs[i].imshow(x_train[i], cmap="gray")

figure, axes = plt.subplots(*(4, 6), figsize=(18, 12))
figure.set_size_inches(18, 12)
random_images = np.random.randint(0, len(x_train_noisy), (4, 6)).flatten()
images = np.reshape(x_train_noisy[random_images], (-1, 28, 28))
labels = y_train[random_images]

for picax, idx, img, label in zip(axes.flat, random_images, images, labels):
    picax.set_title("Image %d : %d" % (idx, label))
    picax.imshow(img, cmap=plt.cm.gray)
    picax.set_frame_on(False)
    picax.set_axis_off()

plt.tight_layout()