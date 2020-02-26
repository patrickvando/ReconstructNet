import math
import numpy as np
import scipy.stats as st


def convolve(image, kernel):
	"""
	Convolution function used for the different kernels
	:param image  :  the original image to convolve with the kernel
	:param kernel :  the kernel matrix to convolve with the original image

	:returns the blurry image
	"""
	kernel = np.flip(np.flip(kernel, 1), 0)

	# width, height, and color channels of the image
	width, height, channels = image.shape
	# width and height of the kernel
	k_width, k_height = kernel.shape

	k_width_left = math.floor(k_width /2)
	k_width_right = math.ceil(k_width /2)
	k_height_top = math.floor(k_height /2)
	k_height_bottom = math.ceil(k_height /2)

	ret = np.zeros(image.shape)

	image_pad = np.zeros((width + k_width + 1 , height + k_height + 1, channels))

	image_pad[k_width_left:-k_width_right - 1, k_height_top:-k_height_bottom - 1]\
		= image

	for x in range(width):
		for y in range(height):
			for c in range(channels):
				ret[x, y, c] = np.multiply(kernel, image_pad[x: x + k_width, y: y + k_height, c]).sum()
	return ret / ret.max()


def add_box_blur(original_image, kernel_size):
	"""
	Add blur using a box kernel
	Box kernel: a matrix where the entries are all the same and their sum is 1

	:param original_image  	:  the original image to convolve with the kernel
	:param kernel_size 		:  the size (1D) of the square box kernel (odd
	number, from 3 to 25)

	:returns the blurry image, which is the original image convolved with
	the kernel
	"""
	kernel = np.ones((kernel_size,kernel_size))/math.pow(kernel_size, 2)
	blurry_image = convolve(original_image, kernel)
	return blurry_image


def add_horizontal_blur(original_image, kernel_size):
	"""
	Add horizontal motion blur
	Horizontal kernel: a matrix where the entries are all 0 except for the
	center row with equal values and their sum is 1

	:param original_image  	:  the original image to convolve with the kernel
	:param kernel_size 		:  the size (1D) of the square kernel	(odd
	number, from 3 to 25)

	:returns the blurry image, which is the original image convolved with
	the horizontal blur kernel
	"""
	kernel = np.zeros((kernel_size, kernel_size))
	kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
	kernel /= kernel_size
	blurry_image = convolve(original_image, kernel)
	return blurry_image


def add_vertical_blur(original_image, kernel_size):
	"""
	Add vertical motion blur
	Vertical kernel: a matrix where the entries are all 0 except for the
	center column with equal values and their sum is 1

	:param original_image  	:  the original image to convolve with the kernel
	:param kernel_size 		:  the size (1D) of the square kernel	(odd
	number, from 3 to 25)

	:returns the blurry image, which is the original image convolved with
	the vertical blur kernel
	"""
	kernel = np.zeros((kernel_size, kernel_size))
	kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
	kernel /= kernel_size
	blurry_image = convolve(original_image, kernel)
	return blurry_image


def add_gaussian_blur(original_image, kernel_size, stddev):
	"""
	Add Gaussian blur
	Gaussian kernel: a matrix whose sum of all entries are 1, and represents a
	Gaussian 2D curve

	:param original_image  	:  the original image to convolve with the kernel
	:param kernel_size 		:  the size (1D) of the square kernel	(odd
	number, from 3 to 25)
	:param sttdev			:  the standard deviation of the gaussian
	function (float value, 1.0 to 8.0)

	:returns the blurry image, which is the original image convolved with
	the Gaussian blur kernel
	"""
	x = np.linspace(-stddev, stddev, kernel_size+1)
	kern1d = np.diff(st.norm.cdf(x))
	kern2d = np.outer(kern1d, kern1d)
	kernel = kern2d/kern2d.sum()
	blurry_image = convolve(original_image, kernel)
	return blurry_image
