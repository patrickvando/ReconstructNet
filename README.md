# ReconstructNet

CWRU EECS 395 Senior Project.

Written by Patrick Do, Jacob Kang, Thomas Lerner, and Kyle Pham.

Currently live at www.reconstructnet.com

<p align="middle">
  <img src="/media/images/readme/title_a.jpg" width="450" />
  <img src="/media/images/readme/title_b.jpg" width="450" /> 
</p>

## Project Background

ReconstructNet is an online web application where users can upload an image of their choosing, and then apply various image reconstruction techniques to that image. There are currently three main features: (1) a de-blurring feature, (2) a de-noising feature, and (3) an inpainting feature. Additionally, the user can also add noise, blurring, and superimposed patterns to their input image in order to test the accuracy and quality of the reconstruction features. Also included are features for sharpening an image and increasing contrast. After processing the image, users can download the reconstructed image for their personal use.?


Images are prone to be corrupted by the acquisition channel or by artificial editing. Image noise is a random visual distortion that affects the color or brightness of different pixels in an image. We call the process of removing image noise denoising. Image blur is a reduction in the detail and contrast in a region of a photo. We call the process of removing image blur deblurring. If an image has missing regions or has some undesirable added pattern, such as superimposed text or lines and scratches, a user may wish to fill in the missing or altered regions. We call this process inpainting. Image reconstruction is an important task, as it is useful to be able to remove various distortions from an image.?


Powerful paid desktop applications like Photoshop can offer effective implementations of de-noising, de-blurring, and inpainting. However, some people may not have access to or may not want to use paid desktop applications. In this case, a web application that offers image reconstruction techniques is desirable.

## Technologies Used

Front-end: HTML, CSS, JavaScript, Bootstrap, Jquery

Web Application: Django

Web Hosting: PythonAnywhere

Application Logic: OpenCV, NumPy, Pillow, Tensorflow, Keras, Scikit-learn

Dataset for Neural Network training: CIFAR-10


The backend of ReconstructNet was written in Django, a Python Model-View-Template backend framework. PythonAnywhere, a cloud-based web hosting service, was used to host the Django application online. Bootstrap 4.4, a CSS/HTML/Javascript framework, was used to handle the design and formatting of the frontend. The image processing algorithms utilize the computer vision library OpenCV, the scientific computing library Numpy, and the python imaging library Pillow. For constructing, training, and evaluating neural network models, the Python packages Tensorflow, Keras, and Scikit-learn were used. The dataset CIFAR-10, which contains 60,000 images of size 32x32, was used for neural network training.


## Delivered Features

There are four main components of ReconstructNet: Application Frontend, Application Architecture, Traditional Algorithms, and Neural Network Reconstruction Models.

### Application Frontend

The user can begin editing the image using the sidebar menu. The sidebar menu was created using Bootstrap?s accordion collapse component. Expanding the accordion reveals the image editing tools currently available in our build. The user can use sliders to adjust the strength of image editing algorithms (adding noise, blurring, ect.) to their chosen image. The number corresponding to the slider value updates dynamically when the slider is moved. A tooltip is displayed when hovering over this number. For specific neural networks, the user can use a select component to adjust the ?strength? of the neural network. The ReconstructNet ?logo?? on the top left and the source code button in the bottom right link to the github repository. There are buttons for resetting, uploading, and downloading images. All components were styled using only vanilla Bootstrap and custom CSS.

![UI Screencap](/media/images/readme/figureB.png "UI with Remove Noise tab open, while loading up a new image.")

### Application Architecture

The web application runs image processing algorithms on the backend in response to requests issued by the frontend. Whenever a user clicks a button on the frontend, an AJAX GET fetch request is issued to the backend. The name of the algorithm and any associated values are passed as parameters in the URL of the GET request. The web application creates an entry in the database for each user that contains two images, one being the original image, and the other being the current image.


When a traditional algorithm is called, the web application directly applies the specified algorithm to the current image. When a neural network is called, the web application loads a .h5 file containing the pre-trained neural network, breaks the current image into pieces, applies the neural network to each piece, then puts the pieces back together. A traditional algorithm or a neural network can be applied multiple times to the current image. After the traditional algorithm or neural network has finished running, the current image is sent to the frontend and displayed in the body of the web page.


When the user resets the image, the database entry corresponding to the original image is copied onto the database entry corresponding to the current image. When the user uploads an image, that image replaces the original and current image in the database. When the user downloads the image, the web application sends the current image from the server as a file attachment.

### Algorithms

We have implemented several algorithms to allow the user to distort an existing image. Results from these distortion algorithms can be found in the Appendix.

Adding noise: There are two different types of noise implemented. For Salt-and-Pepper noise, random black and white pixels are added to the image. For Gaussian noise, the RGB values of every pixel are added to or subtracted to according to a Gaussian distribution. The users can adjust the ratio of pixels affected by the Salt-and-Pepper noise, and the standard deviation of the Gaussian distribution for the Gaussian noise.

Adding blur: There are four different types of blurring in ReconstructNet, all of them implemented using the convolution technique, taking a square kernel and convolving it with the image matrix. For the box blur, pixel colors are averaged in a neighborhood surrounding each pixel. For Gaussian blur, the kernel matrix is formed using a Gaussian distribution and the sum of entries is still 1, creating a less significant blurring effect than the box blur. For motion blur (two types: horizontal and vertical), the pixel values along a line segment representing the direction of motion are averaged. For all of these blur types, the user is allowed to adjust the size of the kernel.

Adding superimposed patterns: In order to do image inpainting reconstruction, we allow the users to add random superimposed patterns onto the input image. Masks with patterns such as lines, circles, and ellipses of different sizes are randomly created in the image. These masks will be added onto the image, creating white patterns simulating scratches on the image. The user is allowed to adjust the maximum size/thickness of these patterns.??

Increasing contrast:? To increase the contrast, we collect all the RGB values, take a specified lower and upper percentile, and then apply a linear stretching function to stretch the values between those percentiles to cover the full color range. The user can modify this percentile.

Sharpening an image: The Unsharp Masking algorithm is used to sharpen the edges in an image. A Gaussian blurred version of the image is compared to the original image, and where the difference exceeds some threshold, the contrast of the image is increased. The implemented algorithm allows the user to modify the percentile of the contrast, the radius of the Gaussian blur, and the threshold, but for the sake of simplicity the user can only modify the contrast percentile from the frontend.?

### Neural Network Reconstruction Models

For our Deep Convolutional Neural Networks for image reconstruction (denoising, deblurring, and inpainting), we used a simplified version of the popular convolutional neural network structure U-net for image processing to train end-to-end models that take inputs of size 32x32x3 and outputs of the same size from the CIFAR-10 dataset. We incorporated regularization terms in the loss function for the neural networks in addition to the mean square loss function to preserve strong edge features. We used 40,000 32x32 images from the CIFAR-10 dataset for training, 10,000 for validation, and 10,000 for testing. To improve the results of the models, data augmentation, where images are randomly transformed (flipped, rotated, moved, zoomed in, or sheared), was used to generate more training data for the neural networks. After the neural networks were pre-trained and evaluated for accuracy, we saved these neural network models into .h5 files so that we could use the models directly for ReconstructNet without having to train them again each time the user chooses an image reconstruction option.


In order to use the neural networks for images of different sizes for ReconstructNet, we implemented a tiling algorithm where high-resolution images are separated into overlapping blocks of size 32x32. Each time the user chooses an image reconstruction option, we use this method to separate the image into these smaller blocks. After applying a neural network to each block, we then reconstruct the image by merging the blocks together and averaging pixel values in the overlapping regions, resulting in an equivalently high-resolution image.

### Sample Results
<p align="middle">
<p>Original Image</p>
  <img src="/media/images/readme/fig1a.jpg" width="450" />
</p>

| Default aligned | Right aligned  |
|-----------------|---------------:|
| First body part | fourth cell    |
| Second line     | baz            |
| Third line      | bar            |
|-----------------+----------------|
| Second body     |                |
| 2nd line        |                |
|-----------------+----------------|
| Third body      | Foo            |


<p align="middle">
<div>
  <img src="/media/images/readme/fig1a.jpg" width="450" />
  description 1
 </div>
<div>
  <img src="/media/images/readme/fig2b.jpg" width="450" /> 
  description 2
</div>
</p>

<p align="middle">
  <img src="/media/images/readme/fig1a.jpg" width="450" />
  <img src="/media/images/readme/fig2b.jpg" width="450" /> 
</p>



