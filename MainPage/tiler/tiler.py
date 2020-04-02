import cv2
import numpy as np
import math

class Tiler:
    def tile(image, tile_height, tile_width):
        """
        Breaks an image into a grid of smaller images (tiles)
        If the image does not perfectly split into tiles,
        each tile is padded with black space
        :param image: the original image
        :param tile_width: width of each tile
        :param tile_height: height of each tile
        :return: a 2d list of tiles
        """
        height, width, channels = image.shape
        padded_width = int(math.ceil(width / tile_width) * tile_width)
        padded_height = int(math.ceil(height / tile_height) * tile_height)
        padded = np.zeros((padded_height, padded_width, channels), dtype='uint8')
        padded[:height, :width] = image
        tiles = padded.reshape(-1, padded_width // tile_width, tile_width, channels).swapaxes(0, 1)
        tiles = tiles.reshape(-1, padded_height // tile_height, tile_height, tile_width, channels).swapaxes(0, 1)
        return tiles

    def stitch(tile_matrix):
        """
        stitches a 2d list of identical tiles into a composite image.
        If an image is split apart into tiles, then stitched together
        the result may not be equivalent to the original image,
        since the result may include padded space. 
        In this case it is necessary to call the Tiler.crop function.

        :param tile_matrix: 2d list of tiles  
        :return: the single image composed from the tiles
        """
        m_height, m_width, tile_height, tile_width, channels = tile_matrix.shape
        composite = tile_matrix.swapaxes(0, 1).reshape(-1, tile_height * m_height, tile_width, channels)
        composite = composite.swapaxes(0, 1).reshape(m_height * tile_height, m_width * tile_width, channels)
        return composite
    
    def crop(image, height, width):
        image = np.delete(image, np.s_[height::], 0)
        image = np.delete(image, np.s_[width::], 1)
        return image

    def show_tiles(tile_matrix):
        """
        For debugging
        Shows all of the tiled images.

        :param tile_matrix: matrix of images
        """
        m_height, m_width, tile_height, tile_width, channels = tile_matrix.shape
        space_buf_x = 100 
        space_buf_y = 0
        for row in range(m_height):
            for col in range(m_width):
                name = "tile " + str(row) + "," + str(col)                
                Tiler.show_image(tile_matrix[row][col], name, (tile_width + space_buf_x) * col, (tile_height + space_buf_y) * row, False)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def show_image(img, name = "img", x = 0, y = 0, single_img = True):
        """
        For debugging
        Shows a single image.

        :param image_path: path to the original image
        :param tile_width: width of each tile
        :param tile_height: height of each tile
        """
        cv2.namedWindow(name)
        cv2.imshow(name, img)
        cv2.moveWindow(name, x, y)
        if single_img:
            cv2.waitKey()
            cv2.destroyAllWindows()

"""
#For debugging
img = cv2.imread("test.jpg")
tiles = Tiler.tile(img, 28, 28) 
composite = Tiler.stitch(tiles)
cropped_composite = Tiler.crop(composite, img.shape[0], img.shape[1])

#Tiler.show_image(img)
#Tiler.show_tiles(tiles)
#Tiler.show_image(composite)
Tiler.show_image(cropped_composite)
"""
