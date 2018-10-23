import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
import scipy.misc

from .USM import USM
from .LoG import LoG2d

def show_images(images, cols = 1, titles = None, gray=False):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    my_dpi=96
    fig = plt.figure(figsize=(700/my_dpi, 500/my_dpi), dpi=my_dpi)
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        #if image.ndim == 2:
        #    plt.gray()
        if gray:
            plt.imshow(image, cmap = plt.cm.gray)
        else:
            plt.imshow(image)
        a.set_title(title)
    plt.show(block=True)

def make_plottable(img):
    #mini = torch.min(img)
    #img = img - mini
    #maxi = torch.max(img)
    #img = img / maxi
    img[img<0] = 0
    img[img>255] = 255
    print("conv", torch.min(img), torch.max(img))
    return img

def gray():
    filename = r"/Users/josemariocarranza/Dropbox/lena.png"#SHUP-170700-TECH-01.jpg"#lena.png"
    x = scipy.ndimage.imread(filename, flatten=True)
    x = scipy.misc.imresize(x, (224, 224), interp='bilinear')
    #x = scipy.ndimage.gaussian_filter(x, sigma=1)
    x = x.reshape((1, 1,x.shape[1], x.shape[0]))
    #funciona con numeros 0-255
    x = torch.FloatTensor(x)
    in_channels = 1
    kernel_size = 3
    dilation = 1
    sigma = 1.6
    lambda_ = 2
    padding = int((1*(in_channels-1)+((kernel_size-1)*(dilation-1))+kernel_size-in_channels)/2)
    l = LoG2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, fixed_coeff=True, sigma=sigma, padding=padding)
    u = USM(in_channels=in_channels, kernel_size=kernel_size, fixed_coeff=True, sigma=sigma)
    u.alpha = nn.Parameter(torch.FloatTensor([lambda_]), requires_grad=True)
    yl = l(x)
    yu = u(x)
    x = make_plottable(x)
    yl = make_plottable(yl)
    yu = make_plottable(yu)
    x = x.view(224, 224).detach().numpy()
    yl = yl.view(224, 224).detach().numpy()
    yu = yu.view(224, 224).detach().numpy()
    images = [  x.astype(dtype=np.float), yl.astype(dtype=np.float), yu.astype(dtype=np.float) ]
    show_images(images, cols = 1, titles = ["Original", "LoG", "USM"], gray=True)

def color():
    filename = r"/Users/josemariocarranza/Dropbox/lena.png"#SHUP-170700-TECH-01.jpg"
    x = scipy.ndimage.imread(filename)
    x = scipy.misc.imresize(x, (224, 224, 3), interp='bilinear')
    #x = scipy.ndimage.gaussian_filter(x, sigma=3)
    x = x.reshape((1, x.shape[2], x.shape[1], x.shape[0]))
    x = torch.FloatTensor(x / 255.0)
    #print(x)
    in_channels = 3
    kernel_size = 3
    dilation = 1
    sigma = 1.6
    lambda_ = 2
    padding = int((1*(in_channels-1)+((kernel_size-1)*(dilation-1))+kernel_size-in_channels)/2)
    l = LoG2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, fixed_coeff=True, sigma=sigma, padding=padding)
    u = USM(in_channels=in_channels, kernel_size=kernel_size, fixed_coeff=True, sigma=sigma)
    u.alpha = nn.Parameter(torch.FloatTensor([lambda_]), requires_grad=True)
    yl = l(x)
    yu = u(x)
    x = make_plottable(x)
    yl = make_plottable(yl)
    yu = make_plottable(yu)
    x = x.view(224, 224, 3).detach().numpy()
    yl = yl.view(224, 224, 3).detach().numpy()
    yu = yu.view(224, 224, 3).detach().numpy()
    images = [  x.astype(dtype=np.float), yl.astype(dtype=np.float), yu.astype(dtype=np.float) ]
    show_images(images, cols = 1, titles = ["Original", "LoG", "USM"], gray=False)

gray()
color()
