import cv2 as cv
import numpy as np


def add_gaussian_noise(original_image, mean, stddev):
	"""
	Add gaussian noise to the image

	:param	original_image	: the original image (array form)
	:param	mean  			: mean of gaussian noise 	(-50 to 50)
	:param	stddev			: standard deviation		(1 to 30)

	:returns the image with Gaussian noise added
	"""
	noise = original_image
	cv.randn(noise, mean, stddev)
	noisy_image = original_image + noise
	return noisy_image


def add_salt_and_pepper_noise(original_image, noise_ratio):
	"""
	Add salt and pepper noise to the image

	:param	original_image	: the original image (array form)
	:param	noise_ratio		: the ratio of pixels with noise 	(0 to 0.8)

	:returns the image with salt and pepper noise added
	"""
	num_rows, num_columns, num_channels = np.shape(original_image)
	ret = np.copy(original_image)
	num_salt = np.ceil(noise_ratio * num_rows * num_columns)
	coords = [np.random.randint(0, i - 1, int(num_salt)) for i in np.shape(
		original_image)]
	ret[coords] = (0, 0, 0)
	return ret


# img = cv.imread("./beach.jpg")
# noisy_image = add_gaussian_noise(img, 50, 15)

# plt.subplot(121),plt.imshow(img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(noisy_image),plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()