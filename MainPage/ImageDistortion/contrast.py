import random
from PIL import Image

def increase_contrast(im, contrast_level):
    print("HAPPENING")
    res = contrast(im, contrast_level, 100 - contrast_level)
    print("contrast", res.size)
    return res

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
        elif n < i:
            end = i - 1
        else:
            start = i + 1

#increase_contrast("lena_copy.png", 30)
