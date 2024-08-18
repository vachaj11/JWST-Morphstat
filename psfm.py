"""Methods for adjusting resolution of frames and related functionalities.

Holds methods that serve the purpose of adjusting the resolution of frames
which in turns entails calculating the current resolution by fitting gaussian
to the psf, translating the pixel size to actual physical size at the 
galaxie's position and convolving the frame with an appropriately sized 
gaussian to adjust the resolution.

Attributes:
    
    H0 (float): Standard Hubble constant from Planck 2018.
    Om0 (float): Standard non-relativistic matter density from Planck 2018.
    Od0 (float): Standard dark energy density from Planck 2018.
    cosm (astropy.cosmology.LambdaCDM): Astropy cosmology instance used for
        cosmological calculations.
    sc (float): Angular size of one frame pixel in radians.
    calculated_stds (dict): Pre-calculated values of stddevs of 2D-Gaussian
        fits of PSFs.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.cosmology import LambdaCDM
from astropy.modeling import fitting, models

# cosmological parameters from Planck 2018
H0 = 67.49
Om0 = 0.315
Od0 = 0.6847

# astropy cosmology
cosm = LambdaCDM(H0, Om0, Od0)

# angular pixel size in radians
sc = 0.025 / 3600 / 180 * np.pi

# pre-calculated stds for the filters
calculated_stds = {
    "F090W": (0.5094135710972533, 0.48798589051061103),
    "F115W": (0.6245571765839361, 0.6391595055066647),
    "F150W": (0.8278315754120771, 0.8064850690050779),
    "F182M": (0.9870815704870342, 1.0087911840098556),
    "F200W": (1.0523929865217865, 1.0738129236725904),
    "F210M": (1.1236368428664782, 1.1446619651300811),
    "F277W": (1.4567996051637313, 1.495440661949474),
    "F300M": (1.5976007778114267, 1.6412691987766348),
    "F335M": (1.7943536817369525, 1.8432931719785013),
    "F356W": (1.9295525921391032, 1.8790711890874585),
    "F410M": (2.1784779101855714, 2.2342502692497477),
    "F430M": (2.288480160296812, 2.3457943371497545),
    "F444W": (2.3134618014245603, 2.370797381028655),
    "F460M": (2.4733636857760466, 2.533261797657502),
    "F480M": (2.5710296272085498, 2.632500314265589),
}


def get_psf_std(name, psf=None, show_fit=False):
    """Get the standard deviation of the provided gaussian-like psf.

    For a provided gaussian-like PSF, obtains stddev of its Gaussian fit
    by either directly undertaking the fitting or by looking up the value
    in already calculated data of :attr:`calculated_stds`.

    Args:
        name (str): Name of the filter who's stddev is to be obtained/looked
            up.
        psf (numpy.array or None): The psf to be fitted and stddev obtained
            represented by a numpy array or None is none is provided.
        show_fit (bool): Whether the results of the Gaussian fitting are to be
            visualised by plotting of the residuals.

    Returns:
        tuple: tuple containing the stddevs of the psf in x and y.
    """
    if name in calculated_stds.keys():
        stds = calculated_stds[name]
    elif psf is not None:
        warnings.warn(f"There is no calculated psf size for {name}")
        y, x = np.mgrid[: psf.shape[0], : psf.shape[1]]
        gaus = models.Gaussian2D(x_mean=int(x.max() / 2), y_mean=int(y.max() / 2))
        fit = fitting.LevMarLSQFitter()
        result = fit(gaus, x, y, psf)
        x_std = result.x_stddev.value
        y_std = result.y_stddev.value
        stds = (x_std, y_std)
        if show_fit:
            plt.imshow(result(x, y) - psf)
            plt.show()
            plt.plot(range(x.max() + 1), result(x, y)[int(y.max() / 2)])
            plt.plot(range(x.max() + 1), psf[int(y.max() / 2)])
            plt.show()
    else:
        warnings.warn(f"Could not determine psf size for {name}. Returning None.")
        stds = None
    return stds


def get_pixel_size(z):
    """From galaxy redshift get the size of one pixel of the frames
    (in lyr).

    By employing astropy's cosmology caluclation and :attr:`sc` pixel size,
    obtains the actual size of frames' pixels in light years at the galaxy's
    distance.

    Args:
        z (float): The redshift of the galaxy.

    Returns:
        float: Physical size at the inputted reshift corresponding to one
        pixel.
    """
    dist = cosm.angular_diameter_distance(z)
    return dist.to("lyr").value * sc


def convolve_std(data, std):
    """Convolve given data with 2D gaussian kernel of given stddev.

    Args:
        data (numpy.array or similar): Data to be convolved.
        std (float): Standard deviation of gaussian kernel to be used for
            the convolution.

    Returns:
        numpy.array or similar: Convolved data.
    """
    gaus = Gaussian2DKernel(x_stddev=std, y_stddev=std)
    return convolve(data, gaus)
