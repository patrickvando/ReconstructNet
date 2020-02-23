


def convolve(image, kernel):
	kernel = np.flip(np.flip(kernel, 1), 0)

	width, height, channels = image.shape     # width, height, and color channels of the image
	k_width, k_height = kernel.shape          # width and height of the kernel

	k_width_left = math.floor(k_width /2)
	k_width_right = math.ceil(k_width /2)
	k_height_top = math.floor(k_height /2)
	k_height_bottom = math.ceil(k_height /2)

	ret = np.zeros(image.shape)

	image_pad = np.zeros \
		((width + k_width + 1 , height + k_height + 1, channels))
	# print(np.shape(image_pad))
	# print(np.shape(image))
	image_pad[k_width_left : -k_width_right - 1, k_height_top : -k_height_bottom - 1] = image
	# cv2_imshow(image_pad)

	for x in range(width):
		for y in range(height):
			for c in range(channels):
				ret[x, y, c] = np.multiply(kernel, image_pad[x: x + k_width, y: y + k_height, c]).sum()
	return ret /ret.max()



