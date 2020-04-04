import numpy as np
import tensorflow as tf
import pypher
import cv2 as cv
import pywt


def termination(diff_u, thresh_u):
	if tf.norm(diff_u) > thresh_u:
		return False
	else:
		return True


def wave_db1_phi(X, trans):
	level = 2
	if ~trans:
		[Y, s] = pywt.wavedec2(X, level, 'db1')
	else:
		Y = pywt.waverec2(X, 'db1')
	return Y


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
	"""
	2D gaussian mask - should give the same result as MATLAB's
	fspecial('gaussian',[shape],[sigma])
	"""
	m, n = [(ss - 1.) / 2. for ss in shape]
	y, x = np.ogrid[-m:m + 1, -n:n + 1]
	h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
	h[h < np.finfo(h.dtype).eps * h.max()] = 0
	sumh = h.sum()
	if sumh != 0:
		h /= sumh
	return h


def inpainting_regularization(
		original_image, rho=1, lambda_=0.01,
		thresh_x=5e-2, alpha=1, multiplier=2):
	B = original_image
	((width, height, channels)) = np.shape(B)

	# x = np.reshape(B, [], 1)
	x = B
	gamma_new = np.zeros(((width, height, channels)))
	tau_new = np.zeros(((width, height, channels)))
	x_new = x

	diff_x = np.ones(((width, height, channels)))

	count = 0

	for i in range(channels):
		gamma_new[:, :, i] = np.reshape(
			wave_db1_phi(x[:, :, i], 0), (width, height))

	psf = matlab_style_gauss2D((5, 5), 1)
	eigs = pypher.psf2otf(psf, (width, height))
	c_eigs = np.conj(eigs)

	eigs_AtA = np.multiply(c_eigs, eigs)
	RhoEye = rho * np.ones(width, height)
	LHS = eigs_AtA + RhoEye

	RHS1 = np.zeros(((width, height, channels)))

	for i in range(channels):
		RHS1[:, :, i] = np.multiply(eigs, np.fft.fft2(B[:, :, i]))

	rnorm = tf.norm(gamma_new)
	Wx = np.zeros(((width, height, channels)))

	while termination(diff_x, thresh_x) is False and count < 1000:
		count = count + 1
		x = x_new
		tau = tau_new
		gamma = gamma_new

		for i in range(channels):
			RHS2_ = np.reshape(
				wave_db1_phi(gamma[:, :, i] - tau[:, :, i] / rho, 1),
				(width, height))
			RHS2 = np.fft.fft2(RHS2_)
			RHS = RHS1[:, :, i] + RHS2
			x_new[:, :, i] = np.fft.ifft2(np.divide(RHS, LHS))
			x_new[:, :, i] = np.real(x_new[:, :, i])

			Wx[:, :, i] = np.reshape(
				wave_db1_phi(x_new[:, :, i], 0), (width, height))
			gamma_new[:, :, i] = \
				np.max(
					np.abs(Wx[:, :, i] + tau[:, :, i] / rho) - lambda_ / rho,
					0) * np.sign(Wx[:, :, i] + tau[:, :, i] / rho)

			tau_new[:, :, i] = tau[:, :, i] + rho * (
						Wx[:, :, i] - gamma_new[:, :, i])

			rnorm_old = rnorm
			rnorm = tf.norm(Wx - gamma_new)

			if rnorm > alpha * rnorm_old:
				rho = rho * multiplier

	return x_new
