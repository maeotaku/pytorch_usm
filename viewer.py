import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import scipy
import scipy.ndimage
import scipy.misc

from USM import USM
from LoG import LoG2d

imgs = [r"/home/michaelgm/Desktop/NewViewer/lena.jpg",
        r"/home/michaelgm/Desktop/NewViewer/carnivora.jpg",
        r"/home/michaelgm/Desktop/NewViewer/girasol.jpg"]

lambdas = [2, 5, 10]

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
    #print("conv", torch.min(img), torch.max(img))
    return img

def gray():
    filename = r"/home/michaelgm/Desktop/NewViewer/lena.jpg"#SHUP-170700-TECH-01.jpg"#lena.png"
    x = scipy.ndimage.imread(filename, flatten=True)
    print("size: ",x.shape)
    x = scipy.misc.imresize(x, (224, 224), interp='bilinear')
    print("size: ",x.shape)
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
    filename = r"/home/michaelgm/Desktop/NewViewer/lena.jpg"#SHUP-170700-TECH-01.jpg"
    x = scipy.ndimage.imread(filename)
    x = scipy.misc.imresize(x, (224, 224, 3), interp='bilinear')
    #x = scipy.ndimage.gaussian_filter(x, sigma=3)
    x = x.reshape((1, x.shape[2], x.shape[1], x.shape[0]))
    x = torch.FloatTensor(x / 255.0)
    in_channels = 3
    kernel_size = 3
    dilation = 1
    sigma = 0.000000001
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


def channels(x, lambda_):
    x = x.reshape((1, 1,x.shape[1], x.shape[0]))
    #funciona con numeros 0-255
    x = torch.FloatTensor(x/255.0)
    in_channels = 1
    kernel_size = 3
    dilation = 1
    sigma = 1.6
    #lambda_ = 7
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
    
    return(yl, yu)


def color_channels(imgs, lambdas):
    images = []
    for img in imgs:
        x = scipy.ndimage.imread(img)
        x = scipy.misc.imresize(x, (224, 224, 3), interp='bilinear')

        x_r = x[:,:,0]  #Red channel
        x_g = x[:,:,1]  #Green channel
        x_b = x[:,:,2]  #Blue channel

        x = scipy.misc.imresize(x, (224, 224, 3), interp='bilinear')
        x = x.reshape((1, x.shape[2], x.shape[1], x.shape[0]))
        x = torch.FloatTensor(x/ 255.0)
        x = make_plottable(x)
        x = x.view(224, 224, 3).detach().numpy()

        img_lambdas = [x.astype(dtype=np.float)]
        for lambda_ in lambdas:
            yl_r, yu_r = channels(x_r, lambda_)
            yl_g, yu_g = channels(x_g, lambda_)
            yl_b, yu_b = channels(x_b, lambda_)

            #LoG
            #yl = np.empty((224, 224, 3))
            #yl[:,:,0] = yl_r
            #yl[:,:,1] = yl_g
            #yl[:,:,2] = yl_b

            #Unsharp Masking
            yu = np.empty((224, 224, 3))
            yu[:,:,0] = yu_r
            yu[:,:,1] = yu_g
            yu[:,:,2] = yu_b

            img_lambdas.append(yu.astype(dtype=np.float))
        images.append(img_lambdas)
    display_images(x, images, lambdas)

#Input: Original image ready to display, list with each image and the lambdas list
def display_images(original_image, images, lambdas):
    fig=plt.figure(figsize=(10, 10))
    rows = len(images)
    columns =  len(lambdas)+1  #+1 because of the original image
    img_pos = 1
    for i in range(rows):
        for j in range(columns):
            a = fig.add_subplot(rows, columns, img_pos)
            plt.axis('off')
            plt.imshow(images[i][j])
            if i == 0:
                if j==0:
                    a.set_title("Original")
                else:
                    a.set_title('λ = '+str(lambdas[j-1]))
            img_pos += 1
    plt.show()

color_channels(imgs, lambdas)

