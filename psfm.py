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
    calculated_fwhms (dict): Pre-calculated values of widths of 2D-Gaussian
        fits of PSFs.
"""

import warnings
import os

import astropy
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.cosmology import LambdaCDM
from astropy.modeling import fitting, models
from photutils.psf import TopHatWindow, create_matching_kernel, matching

# cosmological parameters from Planck 2018
H0 = 67.49
Om0 = 0.315
Od0 = 0.6847

# astropy cosmology
cosm = LambdaCDM(H0, Om0, Od0)

# angular pixel size in radians
sc = 0.025 / 3600 / 180 * np.pi

# pre-calculated fwhms for the filters
calculated_fwhms = {
    "F090W": (1.1995772884306106, 1.149118956666665),
    "F115W": (1.470719758687787, 1.5051056155391633),
    "F150W": (1.9493943876899091, 1.8991272065113258),
    "F182M": (2.324399468263498, 2.375521701356913),
    "F200W": (2.4781960999114885, 2.5286361972775047),
    "F210M": (2.645962560917274, 2.6954729402728326),
    "F277W": (3.430500911832727, 3.521493646912973),
    "F300M": (3.7620623355473835, 3.8648936085711054),
    "F335M": (4.22538001762926, 4.340623710243656),
    "F356W": (4.543749121910588, 4.424874502103289),
    "F410M": (5.129923450562115, 5.261257319645101),
    "F430M": (5.3889589541225735, 5.52392352664033),
    "F444W": (5.447786223407964, 5.582801195553154),
    "F460M": (5.824326385917065, 5.965375660435022),
    "F480M": (6.054312102519142, 6.199064508582882),
}

g_scale = 1

def get_psf(name, scale = g_scale):
    """Tries to get psf of the frame.

    Based on the filter's name tries to obtain the point spread function
    from fits file at the specified path.

    Args:
        name (str): The name of the filter.

    Returns:
        numpy.array or None: Array holding the psf data or `None` if no
            were found.
    """
    try:
        #path = f"../psf/webbpsf_NIRCam_{name}_pixsc25mas.fits"
        dpath = "../psf/"
        files = [os.path.join(dpath, f) for f in os.listdir(dpath) if os.path.isfile(os.path.join(dpath, f)) and f[-5:] == ".fits"]
        film = [f for f in files if (name in f)]
        if film:
            psf = astropy.io.fits.open(film[0])[0].data
            if len(psf.shape) == 3:
                psf = np.sum(psf, axis = 0)
            psf /= np.nansum(psf)
            if scale != 1:
                psf = scale_psf(psf, 1/scale)
            return psf
        else:
            return None
    except:
        warnings.warn(f"Haven't found psf for filter {name}.")
        return None


def get_psf_fwhm(name, psf=None, show_fit=False):
    """Get the width of the provided gaussian-like psf.

    For a provided gaussian-like PSF, obtains fwhm of its Gaussian fit
    by either directly undertaking the fitting or by looking up the value
    in already calculated data of :attr:`calculated_fwhms`.

    Args:
        name (str): Name of the filter who's width is to be obtained/looked
            up.
        psf (numpy.array or None): The psf to be fitted and fwhm obtained
            represented by a numpy array or None is none is provided.
        show_fit (bool): Whether the results of the Gaussian fitting are to be
            visualised by plotting of the residuals.

    Returns:
        tuple: tuple containing the widths of the psf in x and y.
    """
    if name in calculated_fwhms.keys():
        fwhms = calculated_fwhms[name]
    elif psf is not None:
        warnings.warn(f"There is no calculated psf size for {name}")
        y, x = np.mgrid[: psf.shape[0], : psf.shape[1]]
        gaus = models.Gaussian2D(x_mean=int(x.max() / 2), y_mean=int(y.max() / 2))
        fit = fitting.LevMarLSQFitter()
        result = fit(gaus, x, y, psf)
        x_fwhm = result.x_stddev.value * 2 * np.sqrt(2 * np.log(2))
        y_fwhm = result.y_stddev.value * 2 * np.sqrt(2 * np.log(2))
        fwhms = (x_fwhm, y_fwhm)
        if show_fit:
            plt.imshow(result(x, y) - psf)
            plt.show()
            plt.plot(range(x.max() + 1), result(x, y)[int(y.max() / 2)])
            plt.plot(range(x.max() + 1), psf[int(y.max() / 2)])
            plt.show()
    else:
        warnings.warn(f"Could not determine psf size for {name}. Returning None.")
        fwhms = None
    return fwhms


def get_pixel_size(z):
    """From galaxy redshift get the size of one pixel of the frames
    (in kpc).

    By employing astropy's cosmology caluclation and :attr:`sc` pixel size,
    obtains the actual size of frames' pixels in kiloparsecs at the galaxy's
    distance.

    Args:
        z (float): The redshift of the galaxy.

    Returns:
        float: Physical size at the inputted reshift corresponding to one
        pixel.
    """
    dist = cosm.angular_diameter_distance(z)
    return dist.to("kpc").value * sc


def kernel_fwhm(fin_fwhm, cur_fwhm):
    """Creates gaussian kernel to go from one gaussian psf to another based
    on their widths.

    Uses the fact that convolution of two gaussian is a gaussian with width
    equal to sqrt of sum of squares of their widths of the starting gaussians,
    to calculate what width a gaussian kernel to match two gaussian psfs
    should have, and returns the created kernel.

    Args:
        fin_fwhm (float): Width of the gaussian psf to be reached by
            convolving the starting-point psf with the kernel.
        cur_fwhm (float): Width of the starting-point gaussian psf .

    Returns:
        astropy.convolution.Kernel2D: The gaussian kernel with the calculated
            appropriate fwhm.
    """
    fwhm = np.sqrt(fin_fwhm**2 - cur_fwhm**2)
    std = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gaus = Gaussian2DKernel(x_stddev=std, y_stddev=std)
    return gaus


def get_conv_kernel(psf, psf_target, scale_ratio):
    """Creates kernel needed to go (by convolution with it) from one psf
    to another.

    Uses methods from `photutils` to determine what kernel needs to be used
    to bridge two psfs. This is done with fourier transforms where
    some part of the frequency-space need to be filtered out using a windowing
    function, who's optimal shape will differ between cases. Because of this
    the process is slightly iterative, trying different options and for each
    calculating residuals from perfect transformation, and finally deciding
    which option works the best based on the smallest residuals.
    The target psf is also scaled up/down before the kernel-creation to
    account for different pixel sizes in the psf images.
    If the process fails for any reason, `None` is returned.

    Args:
        psf (numpy.array): Data array describing the starting-point psf.
        psf_target (numpy.array): The psf to be reached from `psf` by
            applying convolution with the kernel. Yet unscaled.
        scale_ratio (float): Factor by which `psf_target` should be scaled
            before calculating the kernel to account for pixel-size
            diffrences.

    Returns:
        numpy.array: The kernel going from `psf` to scaled `psf_target`.
    """
    kernels = dict()
    psf_scaled = scale_psf(psf_target, scale_ratio)
    ts = [0.35, 0.37, 0.33, 0.39, 0.31, 0.41, 0.29, 0.43, 0.27, 0.45, 0.25]
    for t in ts:
        window = TopHatWindow(t)
        kernel = create_matching_kernel(psf, psf_scaled, window=window)
        re_psf = convolve(psf, kernel)
        re_psf /= np.sum(re_psf)
        residuals = np.sum(np.abs(re_psf - psf_scaled))
        kernels[residuals] = kernel
        if residuals < 0.01:
            return kernel
    res_min = min(kernels.keys())
    if res_min < 0.25:
        return kernels[res_min]
    else:
        return None


def scale_psf(psf, scale_ratio):
    """Rescales and renormalises the passed psf according to a scale ratio.

    Using `photutils.psf.matching.resize_psf` convenience function, resizes
    the passed psf by a factor of `scale_ratio`.
    Since `resize_psf` only changes resolution, the psf has to be
    furthermore cropped or its edges padded with 0s to maintain the same shape
    of the psf data array.

    Args:
        psf (numpy.array): The psf to be rescaled.
        scale_ratio (float): The scaling factor, also the ratio of output to
            input pixel sizes.

    Returns:
        numpy.array: The rescaled psf.
    """
    shape = psf.shape
    psf_scaled = matching.resize_psf(psf, 1, scale_ratio)
    nshape = psf_scaled.shape
    if scale_ratio < 1:
        xstart = nshape[0] // 2 - shape[0] // 2
        ystart = nshape[1] // 2 - shape[1] // 2
        npsf = psf_scaled[xstart : xstart + shape[0], ystart : ystart + shape[1]]
        npsf /= np.sum(npsf)
    elif scale_ratio > 1:
        xstart = shape[0] // 2 - nshape[0] // 2
        ystart = shape[1] // 2 - nshape[1] // 2
        npsf = np.zeros(shape)
        npsf[xstart : xstart + nshape[0], ystart : ystart + nshape[1]] = psf_scaled
    else:
        npsf = psf
    npsf /= np.nansum(npsf)
    return npsf
