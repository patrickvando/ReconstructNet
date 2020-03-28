import cv2
import numpy as np
import math

class Tiler:
    def tile(image, tile_height, tile_width):
        """
        Breaks an image into tiles of a desired height and width
        If the image does not perfectly split into tiles, each tile is padded with black space
        :param image: the original image
        :param tile_width: width of each tile
        :param tile_height: height of each tile
        :return: a 2d list of the tiles (a 2x3 list of tiles, for example)
        """
        height, width, channels = image.shape
        padded_width = int(math.ceil(width / tile_width) * tile_width)
        padded_height = int(math.ceil(height / tile_height) * tile_height)
        padded = np.zeros((padded_height, padded_width, channels), dtype='uint8')
        padded[:height, :width] = image
        tiles = padded.reshape(-1, padded_width // tile_width, tile_width, channels)
        tiles = tiles.swapaxes(0, 1).reshape(-1, padded_height // tile_height, tile_height, tile_width, channels).swapaxes(0, 1)
        return tiles

    def show_tiles(image_path, tile_height, tile_width):
        """
        For debugging. Shows an example of how to access tiled images.
        Shows all of the tiled images.
        run like:
        Tiler.show_tiles("test.jpg", 200, 200)

        :param image_path: path to the original image
        :param tile_width: width of each tile
        :param tile_height: height of each tile

        """
        img = cv2.imread(image_path)
        tiles = Tiler.tile(img, tile_height, tile_width) 
        space_buf_x = 100 
        space_buf_y = 0
        for row in range(tiles.shape[0]):
            for col in range(tiles.shape[1]):
                winname = "image " + str(row) + "," + str(col)                
                cv2.namedWindow(winname)        # Create a named window
                cv2.moveWindow(winname, (tile_width + space_buf_x) * col, (tile_height + space_buf_y) * row)
                cv2.imshow(winname, tiles[row][col])
        cv2.waitKey()
        cv2.destroyAllWindows()

    def stitch(tile_matrix):
        """
        stitches a 2d list of identical tiles into a composite image, then crops that image to fit the desired height and width
        :param tile_matrix: 2d list of tiles  
        :return: the cropped image composed from the tiles
        """
        pass
    
    def crop(image, height, width):
        pass

#Tiler.show_tiles("test.jpg", 200, 200)
