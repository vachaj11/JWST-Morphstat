"""classes representing each of the galaxies and frames, holding methods for
statmorph computation, testing of the results, etc
"""
import statmorph
import astropy
import matplotlib.pyplot as plt
import numpy as np

from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import (
    make_2dgaussian_kernel,
    detect_sources,
    detect_threshold,
    SegmentationImage,
)
from photutils.utils import circular_footprint
from statmorph.utils.image_diagnostics import make_figure


class galaxy:
    """class representing a single galaxy with multiple parameters, frames in
    different filters and corresponding methods
    """
    def __init__(self, name, info, filters, fitss):
        self.fitss = fitss
        self.info = info
        self.name = name
        self.filters = filters
        self.frames = self.get_frames(self.filters, self.fitss)
        self.target_flag = self.target_test(self.frames)

    def get_frames(self, filters, fitss):
        """calculates frames objects from provided fits data"""
        if len(filters) == len(fitss):
            frames = []
            for i in range(len(filters)):
                frames.append(frame(filters[i], fitss[i]))
            return frames
        else:
            print("number of frames not matching the number of filters")
            return []

    def target_test(self, frames):
        """tests whether identified targets in multiple frames are
        overlapping"""
        try:
            for i in range(len(frames) - 1):
                target1 = frames[i].target
                target2 = frames[i + 1].target
                overlap = target1 * target2
                t1_size = np.sum(target1)
                t2_size = np.sum(target2)
                ol_size = np.sum(overlap)
                if ol_size < 0.5 * min(t1_size, t2_size):
                    print(
                        f"targets in {self.name} frames {frames[i+1].name} and {frames[i].name} non-ovelapping"
                    )
                    return 3
            return 1
        except:
            return 2


class frame:
    """class holding all data relating to a single frame/photo of a galaxy at
    some wavelength needed to run statmorph, corresponding methods, etc.
    """
    def __init__(self, name, fits):
        self.name = name
        self.fits = fits
        self.data = self.fits[1].data
        self.convolved = self.convolve(self.data)
        self.objects_seg = self.segment(self.convolved)
        self.get_background(self.objects_seg, self.data)
        self.data_sub = self.bg_subtract(self.data)
        self.target, self.mask = self.isolate(self.objects_seg)
        self.psf = self.get_psf(self.name)
        self.stmo = self.get_stmo(self.data_sub, self.target, self.mask, self.psf)

    def convolve(self, data):
        dim = min(data.shape)
        fwhm = int(np.ceil(dim / 25) // 2 * 2 - 1)
        size = int(np.ceil(dim / 15) // 2 * 2 + 1)
        kernel = make_2dgaussian_kernel(fwhm, size=size)
        return convolve(data, kernel)

    def segment(self, data):
        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
        threshold = detect_threshold(data, nsigma=8.0, sigma_clip=sigma_clip)
        seg_map = detect_sources(data, threshold, 40)
        if seg_map is not None:
            return seg_map
        else:
            mean = np.mean(data)
            mp = (np.clip(data, None, mean) - mean).astype(bool).astype(int)
            return SegmentationImage(mp)

    def get_background(self, seg_map, data):
        footprint = circular_footprint(radius=25)
        self.bg_mask = seg_map.make_source_mask(footprint=footprint)
        self.background = data * (1 - self.bg_mask)
        self.bg_mean, self.bg_med, self.bg_std = sigma_clipped_stats(
            self.data, sigma=3.0, mask=self.bg_mask
        )

    def bg_subtract(self, data):
        if self.bg_std > 0.5 * self.bg_med:
            return data
        else:
            print("subtracted background!")
            return data - self.bg_med

    def isolate(self, seg_map):
        """Isolate the target in the segmentation map from the rest"""
        item = self.get_central(seg_map)
        seg_map_t = np.invert((seg_map - item).astype(bool)).astype(int)
        mask = (seg_map - seg_map_t * item).astype(bool)
        return seg_map_t, mask

    def get_central(self, seg_map, margin=0.6):
        """Get the index of the most central target in the segmentation map"""
        members = self.get_members(seg_map)
        sx, sy = seg_map.shape
        cx = int(sx / 2)
        cy = int(sy / 2)
        for i in range(1, min(cx, cy) - 2):
            mask = np.zeros((sx, sy))
            mask[cx - i : cx + i, cy - i : cy + i] = 1
            elements = self.get_members(mask * seg_map)
            for i in elements:
                if elements[i] >= members[i] * margin:
                    return i
        return max(members, key=members.get)

    def get_members(self, seg_map):
        """counts how big area is spanned by an object in a given seg map"""
        unique, counts = np.unique(seg_map, return_counts=True)
        elements = dict(zip(unique, counts))
        elements.pop(0)
        return elements

    def get_stmo(self, data, seg_map, mask, psf):
        if psf is not None:
            return statmorph.source_morphology(
                data, seg_map, gain=1, mask=mask, psf=psf
            )[0]
        else:
            return statmorph.source_morphology(data, seg_map, gain=1, mask=mask)[0]

    def get_psf(self, name):
        try:
            path = f"../psf/webbpsf_NIRCam_{name}_pixsc25mas.fits"
            psf = astropy.io.fits.open(path)[0].data
            return psf
        except:
            print(f"haven't found psf for filter {name}")
            return None

    def show_seg(self):
        fig, axs = plt.subplots(2, 4)
        axs[0, 0].imshow(np.log(self.data))
        axs[0, 0].set_title("raw data")
        axs[0, 1].imshow(np.log(self.convolved))
        axs[0, 1].set_title("convolved data")
        axs[0, 2].imshow(self.objects_seg)
        axs[0, 2].set_title("segmentation map")
        axs[0, 3].imshow(self.bg_mask)
        axs[0, 3].set_title("background mask")
        im = axs[1, 0].imshow(self.background)
        axs[1, 0].set_title("isolated background")
        cmap = im.get_cmap()
        norm = plt.Normalize(vmin=self.background.min(), vmax=self.background.max())
        val = self.bg_med
        color = cmap(norm(val))
        axs[1, 0].legend(
            [plt.Rectangle((0, 0), 1, 1, color=color)],
            [f"Median value:\n {val:.4f}"],
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="black",
            handlelength=1,
            handletextpad=0.5,
            markerfirst=False,
        )
        axs[1, 1].imshow(np.log(self.data_sub))
        axs[1, 1].set_title("subtracted data")
        axs[1, 2].imshow(self.target)
        axs[1, 2].set_title("target")
        axs[1, 3].imshow(self.mask)
        axs[1, 3].set_title("mask")
        fig.tight_layout()
        plt.show()

    def show_stmo(self):
        fig = make_figure(self.stmo)
        plt.show()
