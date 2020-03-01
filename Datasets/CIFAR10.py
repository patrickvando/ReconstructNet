import tensorflow as tf

(image_train, label_train),(image_test, label_test) = tf.keras.datasets.cifar10.load_data()
image_train = image_train / 255

image_validate = image_train[-int(image_train.shape[0] /5):]
image_train = image_train[:-int(image_train.shape[0] /5)]
image_test = image_test / 255

