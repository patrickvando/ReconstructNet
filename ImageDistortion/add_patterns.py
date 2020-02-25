import os
from random import randint, seed
import numpy as np
import cv2


class MaskGenerator:

	def __init__(self, height, width, channels=3, rand_seed=None,
				 file_path=None):
		"""
		Convenience functions for generating masks to be used for inpainting training

		Arguments:
			height {int} -- Mask height
			width {width} -- Mask width

		Keyword Arguments:
			channels {int} -- Channels to output (default: {3})
			rand_seed {[type]} -- Random seed (default: {None})
			file_path {[type]} -- Load masks from file_path.
				If None, generate masks with OpenCV (default: {None})
		"""

		self.height = height
		self.width = width
		self.channels = channels
		self.file_path = file_path

		# If file_path supplied, load the list of masks within the directory
		self.mask_files = []
		if self.file_path:
			filenames = [f for f in os.listdir(self.file_path)]
			self.mask_files = [f for f in filenames if any(
				filetype in f.lower() for filetype in
				['.jpeg', '.png', '.jpg'])]
			print(">> Found {} masks in {}".format(len(self.mask_files),
												   self.file_path))

		# Seed for reproducibility
		if rand_seed:
			seed(rand_seed)

	def generate_mask(self, max_size_ratio, max_lines, max_circles,
				 max_ellipses):
		"""
		Generates a random irregular mask with lines, circles and
		ellipses
		"""
		img = np.zeros((self.height, self.width, self.channels), np.uint8)

		# Set size scale
		size = int((self.width + self.height) / 2 * max_size_ratio)
		if size < 1: size = 1

		# Draw random lines
		if max_lines >= 1:
			for _ in range(randint(1, max_lines)):
				x1, x2 = randint(1, self.width), randint(1, self.width)
				y1, y2 = randint(1, self.height), randint(1, self.height)
				thickness = randint(1, size)
				cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

		# Draw random circles
		if max_circles >= 1:
			for _ in range(randint(1, max_circles)):
				x1, y1 = randint(1, self.width), randint(1, self.height)
				radius = randint(1, size)
				cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

		# Draw random ellipses
		if max_ellipses >= 1:
			for _ in range(randint(1, max_ellipses)):
				x1, y1 = randint(1, self.width), randint(1, self.height)
				s1, s2 = randint(1, self.width), randint(1, self.height)
				a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
				thickness = randint(1, size)
				cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1),
							thickness)

		return 1 - img

	def load_mask(self, rotation=True, dilation=True, cropping=True):
		"""Loads a mask from disk, and optionally augments it"""

		# Read image
		mask = cv2.imread(os.path.join(self.file_path,
									   np.random.choice(self.mask_files, 1,
														replace=False)[0]))

		# Random rotation
		if rotation:
			rand = np.random.randint(-180, 180)
			M = cv2.getRotationMatrix2D((mask.shape[1] / 2, mask.shape[0] / 2),
										rand, 1.5)
			mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

		# Random dilation
		if dilation:
			rand = np.random.randint(1, 3)
			kernel = np.ones((rand, rand), np.uint8)
			mask = cv2.erode(mask, kernel, iterations=1)

		# Random cropping
		if cropping:
			x = np.random.randint(0, mask.shape[1] - self.width)
			y = np.random.randint(0, mask.shape[0] - self.height)
			mask = mask[y:y + self.height, x:x + self.width]

		return (mask > 1).astype(np.uint8)


def add_random_patterns(
		original_image, max_size_ratio, max_lines, max_circles, max_ellipses):
	"""
	Add random patterns to an image

	:param original_image	:	the original image to ask the mask

	:param max_size_ratio	:	the max size ratio of a pattern compared to
								the 1D size of the image (height + width) / 2
								range: 0.01 to 0.30

	:param max_lines		:	the maximum number of line patterns
								range: int 0 to 10

	:param max_circles		:	the maximum number of circle patterns
								range: int 0 to 10

	:param max_ellipses		:	the maximum number of ellipses patterns
								range: int 0 to 10

	:returns: the image with superimposed patterns
	"""
	width, height, channels = np.shape(original_image)
	mask_generator = MaskGenerator(width, height, channels)
	mask = mask_generator.\
		generate_mask(max_size_ratio, max_lines, max_circles, max_ellipses)
	new_image = np.copy(original_image)
	new_image[mask[:, :, 0] == 0] = 1
	return new_image

