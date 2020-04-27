import cv2
import numpy as np
import math

class Tiler:
    def make_padding(tile, tile_height, tile_width):
        t_h, t_w, channels = tile.shape
        padded = np.full((tile_height, tile_width, channels), 255)
        p_h, p_w, channels = padded.shape
        padded[:t_h, :t_w] = tile
        for r in range(t_h):
            target_p = padded[r][t_w - 1]
            rp = np.repeat(target_p.reshape(1, 1, 3), p_w - t_w, axis=0).reshape(1, -1, 3)
            
            padded[r, t_w:p_w] = rp

        for c in range(t_w):
            target_p = padded[t_h - 1][c]
            rp = np.repeat(target_p.reshape(1, 1, 3), p_h - t_h, axis=0).reshape(1, -1, 3)
            padded[t_h:p_h, c] = rp

        for r in range(t_h, p_h):
            target_p = padded[r][t_w - 1]
            rp = np.repeat(target_p.reshape(1, 1, 3), p_w - t_w, axis=0).reshape(1, -1, 3)
            padded[r, t_w:p_w] = rp

        return padded

    def tile_overlap(image, tile_height, tile_width, overlap):
        height, width, channels = image.shape
        tile_matrix = np.array(np.zeros(shape=(tile_height, tile_width, channels)))
        tile_matrix = tile_matrix.reshape(1, *tile_matrix.shape)
        for r in range(0, height, tile_height - overlap):
            for c in range(0, width, tile_width - overlap):
                tile = image[r:r+tile_height, c:c+tile_width]
                h, w, c = tile.shape
                if h < tile_height or w < tile_width:
                    padded = Tiler.make_padding(tile, tile_height, tile_width)
                    tile = padded
                tile = tile.reshape(1, *tile.shape)
                
                tile_matrix = np.concatenate((tile_matrix, tile))
        tile_matrix = tile_matrix[1:] 
        t_m_height = math.ceil(height / (tile_height - overlap))
        t_m_width = math.ceil(width / (tile_width - overlap))
        tile_matrix = tile_matrix.reshape(t_m_height, t_m_width, tile_height, tile_width, channels)
        return tile_matrix



    def stitch_overlap(tile_matrix, overlap):
        h, w, tile_height, tile_width, channels = tile_matrix.shape
        res = np.zeros(shape=(h * (tile_height - overlap) + overlap, w * (tile_width - overlap) + overlap, channels))
        target_height, target_width, channels = res.shape
        oh = tile_height - overlap
        ow = tile_width - overlap
        for r in range(h):
            for c in range(w):
                res[r * oh: r * oh + tile_height, c * ow: c * ow + tile_width] += tile_matrix[r, c]

        for r in range(oh, oh * h, oh):
            res[r:r + overlap, 0:target_width] = res[r:r + overlap, 0:target_width] / 2

        for c in range(ow, ow * w, ow):
            res[0:target_height, c:c + overlap] = res[0:target_height, c: c + overlap] / 2

        return res


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
        img = img.astype("uint8")
        cv2.namedWindow(name)
        cv2.imshow(name, img)
        cv2.moveWindow(name, x, y)
        if single_img:
            cv2.waitKey()
            cv2.destroyAllWindows()
"""
#For debugging
img = cv2.imread("test.jpg")
tiles = Tiler.tile(img, 29, 28) 
composite = Tiler.stitch(tiles)
cropped_composite = Tiler.crop(composite, img.shape[0], img.shape[1])

#Tiler.show_image(img)
#Tiler.show_tiles(tiles)
#Tiler.show_image(composite)
Tiler.show_image(cropped_composite)
"""
"""
img = cv2.imread("beach.jpg")
#img = np.zeros((485, 500, 3))
tiles = Tiler.tile_overlap(img, 32, 32, 10)
composite = Tiler.stitch_overlap(tiles, 10)
#composite = Tiler.crop(composite, img.shape[0], img.shape[1])
Tiler.show_image(composite)
"""
"""
testimg = np.array([[[1, 2, 3],[4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
t = np.repeat(testimg.reshape(1, *testimg.shape), 16, axis = 0)

print(t.shape)
t.reshape(8, 8, 3)
print(t.shape)
print(t)
"""
