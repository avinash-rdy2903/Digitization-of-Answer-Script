import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    dx = gaussian_filter((random_state.rand(h,w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(h,w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    
    if len(image.shape) > 2:
        c = image.shape[2]
        distored_image = [map_coordinates(image[:,:,i], indices, order=1, mode='reflect') for i in range(c)]
        distored_image = np.concatenate(distored_image, axis=1)
    else:
        distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    random_state.seed(32)
    return distored_image.reshape(image.shape)