import os
import numpy as np
import pandas as pd
import cPickle as pickle
from natsort import natsorted

from skimage import exposure
from matplotlib import pyplot
from skimage.io import imread
from PIL import Image
from skimage.io import imshow
from skimage.filters import sobel
from skimage import feature

from sklearn.preprocessing import StandardScaler

PATH = '/Volumes/Mildred/Kaggle/chars_74k/Data/test'

maxPixel = 64
imageSize = maxPixel * maxPixel
num_features = imageSize

def plot_sample(x):
    img = x.reshape(maxPixel, maxPixel)
    imshow(img)
    pyplot.show()

def load_images(path):
    print 'reading file names ... '
    names = [d for d in os.listdir (path) if d.endswith('.Bmp')]
    names = natsorted(names)
    num_rows = len(names)
    print num_rows

    print 'making dataset ... '
    test_image = np.zeros((num_rows, num_features), dtype = float)
    file_names = []
    i = 0
    for n in names:
        print n.split('.')[0]

        image = imread(os.path.join(path, n), as_grey = True)
        #image = sobel(image)

        test_image[i, 0:num_features] = np.reshape(image, (1, num_features))
        i += 1

    return test_image

test = load_images(PATH)

print test[0]
print test.shape

np.save('test_32.npy', test)

plot_sample(test[0])
print np.amax(test[0])
print np.amin(test[0])

#print file_names
