"""classes representing each of the galaxies and frames, holding methods for
statmorph computation, testing of the results, etc
"""

import warnings
import statmorph
import astropy
import matplotlib.pyplot as plt
import numpy as np
import seg

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
        self.update_frame_masks()
        self.flag_targets()
        for f in self.frames:
            f.calc_stmo()

    def get_frames(self, filters, fitss):
        """calculates frames objects from provided fits data"""
        if len(filters) == len(fitss):
            frames = []
            for i in range(len(filters)):
                frames.append(frame(filters[i], fitss[i]))
            return frames
        else:
            warnings.warn(
                f"Number of frames not matching the number of filters for galaxy {self.name}."
            )
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
                    warnings.warn(
                        f"Targets in {self.name} frames {frames[i+1].name} and {frames[i].name} non-ovelapping."
                    )
                    return 3
                if ol_size < 0.5 * max(t1_size, t2_size):
                    warnings.warn(
                        f"Targets in {self.name} frames {frames[i+1].name} and {frames[i].name} suspiciously different."
                    )
                    return 2
            return 0
        except:
            return 1

    def update_frame_masks(self):
        """checks whether the blue frames have large enough segmentation mask
        and if not replaces their segmentation map with the average of
        the rest
        """
        f_red = []
        i_blue = []
        for f in self.frames:
            if int(f.name[1:-1]) < 200:
                i_blue.append(self.frames.index(f))
            else:
                f_red.append(f)
        if len(f_red) > 0:
            targets = [f.target for f in f_red]
            masks = [f.mask for f in f_red]
            t_sum = np.zeros(targets[0].shape)
            m_sum = np.zeros(masks[0].shape)
            for t in targets:
                t_sum = t_sum + t
            for m in masks:
                m_sum = m_sum + m
            margin = int((len(f_red) + 1) / 2)
            t_area = np.sum(t_sum >= margin)
            target_avg = (t_sum >= margin).astype(int)
            mask_avg = ((m_sum >= margin) * (1 - target_avg)).astype(bool)
            for i in i_blue:
                f_area = np.sum(self.frames[i].target)
                if f_area * 3 < t_area or f_area > t_area * 2:
                    self.frames[i].target = target_avg
                    self.frames[i].mask = mask_avg

    def flag_targets(self):
        """for each of the frames determines whether its target segmentation
        map is comparable to the others and flags outliers
        """
        targets = [f.target for f in self.frames]
        t_sum = np.zeros(targets[0].shape)
        margin = int((len(targets) + 1) / 2)
        for t in targets:
            t_sum = t_sum + t
        target_avg = (t_sum >= margin).astype(int)
        t_area = np.sum(target_avg)
        for i in range(len(self.frames)):
            target = self.frames[i].target
            overlap = np.sum(target * target_avg)
            if overlap * 3 < t_area or overlap * 4 < np.sum(target):
                self.frames[i].flag_seg = 1
            else:
                self.frames[i].flag_seg = 0


class frame:
    """class holding all data relating to a single frame/photo of a galaxy at
    some wavelength needed to run statmorph, corresponding methods, etc.
    """

    def __init__(self, name, fits):
        self.name = name
        self.fits = fits
        self.data = self.fits[1].data
        self.convolved = self.convolve(self.data)
        self.objects_seg, self.threshold = self.segment(self.convolved)
        self.get_background(self.objects_seg, self.data)
        self.data_sub = self.bg_subtract(self.data)
        self.target, self.mask = self.isolate(self.objects_seg)
        self.psf = self.get_psf(self.name)

    def calc_stmo(self):
        self.stmo = self.get_stmo(self.data_sub, self.target, self.mask, self.psf)

    def convolve(self, data):
        dim = min(data.shape)
        fwhm = int(np.ceil(dim / 25) // 2 * 2 - 1)
        size = int(np.ceil(dim / 15) // 2 * 2 + 1)
        kernel = make_2dgaussian_kernel(fwhm, size=size)
        return convolve(data, kernel)

    def segment(self, data):
        # sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
        # threshold = detect_threshold(data, nsigma=8.0, sigma_clip=sigma_clip)
        agr = int(self.name[1:-1]) < 150
        threshold = seg.find_threshold(data, seg.find_max(data), agr)
        seg_map = detect_sources(data, threshold, 40)
        if seg_map is not None:
            return seg_map, threshold
        else:
            mean = np.mean(data)
            mp = (np.clip(data, None, mean) - mean).astype(bool).astype(int)
            return SegmentationImage(mp), threshold

    def get_background(self, seg_map, data):
        footprint = circular_footprint(radius=25)
        self.bg_mask = seg_map.make_source_mask(footprint=footprint)
        self.background = data * (1 - self.bg_mask)
        self.bg_mean, self.bg_med, self.bg_std = sigma_clipped_stats(
            self.data, sigma=3.0, mask=self.bg_mask
        )

    def bg_subtract(self, data):
        if self.bg_std > 0.1 * self.bg_med:
            return data
        else:
            warnings.warn("Subtracted background!")
            return data - self.bg_med

    def enlarge_mask(self, mask, seg_map):
        fpm = circular_footprint(radius=10)
        fps = circular_footprint(radius=20)
        mask_o = SegmentationImage(mask.astype(int))
        mask_l = mask_o.make_source_mask(footprint=fpm)
        sm_o = SegmentationImage(seg_map)
        sm_l = sm_o.make_source_mask(footprint=fps)
        return (mask_l * (1 - sm_l) + mask).astype(bool)

    def isolate(self, seg_map):
        """Isolate the target in the segmentation map from the rest"""
        item = self.get_central(seg_map)
        seg_map_t = np.invert((seg_map - item).astype(bool)).astype(int)
        mask = (seg_map - seg_map_t * item).astype(bool)
        mask = self.enlarge_mask(mask, seg_map_t)
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
            warnings.warn(f"Haven't found psf for filter {name}.")
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

    def show_stmo(self, save_path=None):
        fig = make_figure(self.stmo)
        if save_path is not None:
            fig.savefig(save_path, dpi=200)
            plt.close(fig)
        else:
            plt.show()
