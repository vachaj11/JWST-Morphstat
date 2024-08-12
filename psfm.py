"""methods for adjusting resolution of frames
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling import models, fitting
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.cosmology import LambdaCDM

# cosmological parameters from Planck 2018
H0 = 67.49
Om0 = 0.315
Od0 = 0.6847

# angular pixel size in radians
sc = 0.025 / 3600 / 180 * np.pi


def get_psf_std(psf):
    """get the standard deviation of the provided gaussian-like psf"""
    y, x = np.mgrid[: psf.shape[0], : psf.shape[1]]
    gaus = models.Gaussian2D(x_mean=int(x.max() / 2), y_mean=int(y.max() / 2))
    fit = fitting.LevMarLSQFitter()
    result = fit(gaus, x, y, psf)
    x_std = result.x_stddev.value
    y_std = result.y_stddev.value
    return (x_std, y_std)


def get_pixel_size(z):
    """from galaxy information get the size of one pixel the frames (in Mpc)"""
    cosm = LambdaCDM(H0, Om0, Od0)
    dist = cosm.angular_diameter_distance(z)
    return dist.to("lyr").value * sc


def convolve_std(data, std):
    """convolve given data with 2D gaussian kernel of given stddev"""
    gaus = Gaussian2DKernel(x_stddev=std, y_stddev=std)
    return convolve(data, gaus)
