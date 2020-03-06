import random
from PIL import Image

#radius -> between 1 and 15
#contrast level -> between 1 and 50 
#threshold -> between 1 and 255
#does not work with greyscale images (fix later)
def unsharp_mask(img_path, radius, contrast_level, threshold):
    im = Image.open(img_path)

    blurred = gaussian(im, radius)
    #unsharp_mask = subtract(im, blurred)
    unsharp_mask = subtract(blurred, im)
    unsharp_mask.show()
    high_contrast = contrast(im, contrast_level, 100 - contrast_level)

    sharpened = sharpen(im, unsharp_mask, high_contrast, threshold)
    
    #sharpened.save(img_path)

def sharpen(im, unsharp_mask, high_contrast, threshold):
    pixelMap = im.load()
    umMap = unsharp_mask.load()
    hcMap = high_contrast.load()

    img = Image.new(im.mode, im.size)
    pixelsNew = img.load()

    for i in range(im.size[0]):
        for j in range(im.size[1]):
            if luminance(umMap[i, j]) > threshold:
                pixelsNew[i, j] = hcMap[i, j]
            else:
                pixelsNew[i, j] = pixelMap[i, j]
    return img
  

def luminance(pixel):
    R, G, B = pixel
    return (R+R+B+G+G+G)/6

#should these values be clamped?
def subtract(im1, im2):
    pixelMap1 = im1.load()
    pixelMap2 = im2.load()

    img = Image.new(im1.mode, im1.size)
    pixelsNew = img.load()

    for i in range(im1.size[0]):
        for j in range(im1.size[1]):
            a = pixelMap1[i, j] 
            b = pixelMap2[i, j]
            pixelsNew[i, j] = tuple([x-y for x, y in zip(a, b)])
    return img
def gaussian(im, filter_size):
    pixelMap = im.load()
    
    img = Image.new( im.mode, im.size)
    pixelsNew = img.load()

    filt = get_filter(filter_size)

    for i in range(img.size[0]):
        for j in range(img.size[1] - len(filt)):
            w = [0, 0, 0]
            for k in range(len(filt)):
                tup = pixelMap[i, j + k]
                w[0] += filt[k]*tup[0]
                w[1] += filt[k]*tup[1]
                w[2] += filt[k]*tup[2]
            pixelsNew[i, j + len(filt) // 2] = (int(w[0]), int(w[1]), int(w[2]), 255)

    for j in range(img.size[1]):
        for i in range(img.size[0] - len(filt)):
            w = [0, 0, 0]
            for k in range(len(filt)):
                tup = pixelMap[i + k, j]
                w[0] += filt[k]*tup[0]
                w[1] += filt[k]*tup[1]
                w[2] += filt[k]*tup[2]
            pixelsNew[i + len(filt) // 2, j] = (int(w[0]), int(w[1]), int(w[2]), 255)
    return img

def get_filter(n):
    lst = []
    for k in range(n + 1):
        lst.append(combination(n, k) / 2**n)
    return lst

def combination(n, r):
    res = 1
    for k in range(1, r + 1):
        res *= (n + 1 - k) / k
    return int(res)

def contrast(im, lower_percentile, upper_percentile):
    pixelMap = im.load()

    img = Image.new( im.mode, im.size)
    pixelsNew = img.load()

    R = []
    G = []
    B = []
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            tup = pixelMap[i, j]
            R.append(tup[0])
            G.append(tup[1])
            B.append(tup[2])
    R.sort()
    G.sort()
    B.sort()
    low_R = getPercentile(R, lower_percentile)
    high_R = getPercentile(R, upper_percentile)
    low_G = getPercentile(G, lower_percentile)
    high_G = getPercentile(G, upper_percentile)
    low_B = getPercentile(B, lower_percentile)
    high_B = getPercentile(B, upper_percentile)
    for i in range(img.size[0]):
        for j in range(img.size[1]):
                R, G, B = pixelMap[i, j]
                R = flatten(R, low_R, high_R)
                G = flatten(G, low_G, high_G)
                B = flatten(B, low_B, high_B)
                pixelsNew[i,j] = (R, G, B, 255)
    return img

def flatten(val, low, high):
    r = (val - low) / (high - low + 1) * 255
    r = max(0, min(r, 255))
    return int(r)

def getPercentile(lst, n):
    return lst[int(n / 100 * (len(lst) - 1))]
    #return quickSelect(lst, int(n / 100 * len(lst)))

#bad partitioning
def quickSelect(lst, n):
    start = 0
    end = len(lst) - 1
    while True:
        pivot = lst[end]
        i = start
        for k in range(start, end):
            if lst[k] < pivot:
                lst[i], lst[k] = lst[k], lst[i]
                i += 1
        lst[i], lst[end] = lst[end], lst[i]
        if i == n:
            return pivot
 
unsharp_mask("lena_copy.png", 11, 5, 10)
