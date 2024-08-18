"""Classes representing each of the galaxies and frames, holding methods for
statmorph computation, testing of the results, etc.

Holds classes used for internal representation of the galaxies and frames by
the program. These classes at initiation store base-data (fits date, etc)
provided and subsequently can use them for calculations, visualisations, etc.
"""

import warnings

import astropy
import matplotlib.pyplot as plt
import numpy as np
import statmorph
from astropy.convolution import convolve
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.segmentation import (
    SegmentationImage,
    detect_sources,
    detect_threshold,
    make_2dgaussian_kernel,
)
from photutils.utils import circular_footprint
from statmorph.utils.image_diagnostics import make_figure

import psfm
import seg


class galaxy:
    """Class representing a single galaxy with multiple input parameters,
    frames in different filters and corresponding methods for testing and
    calculations.

    Based on provided initiation parameters including galaxy information,
    filters in which it is pictured, corresponding fits files, etc. can
    undertake various preparations for statmorph calculation, the calculation
    itself and testing of the results.

    Args:
        name (str): Name of the galaxy.
        info (dict): Dictionary storing various information about the galaxy
            obtained elsewhere.
        filters (list of str): List of filter names the galaxy is pictured at.
        fitss (list of astropy.io.fits.HDUList): List of fits files
            corresponding to each of the filters.
        psf_res (float or None): Target resolution (in lightyears) in which
            all the calculations are to be undertaken. If `None` no adjustions
            are made.

    Attributes:
        name (str): Name of the galaxy.
        info (dict): Dictionary storing various information about the galaxy
            obtained elsewhere.
        filters (list of str): List of filter names the galaxy is pictured at.
        fitss (list of astropy.io.fits.HDUList): List of fits files
            corresponding to each of the filters.
        frames (list of :obj:`frame`): List of internal representations of
            frames corresponding to each of the filters.
        pixel_size (float): Physical size (in lyr) corresponding to one pixel
            in the frames.
        target_flag (int): Flag noting whether targets identified in each of
            the frames are overlapping. (0 - good, 1-3 - increasingly bad)
    """

    def __init__(self, name, info, filters, fitss, psf_res=None):
        """Initiates the class, stores input parameters/data and undertakes
        all the calculations of resolution-adjustions, segmentation
        maps, statmorph, etc.
        """
        self.fitss = fitss
        self.info = info
        self.name = name
        self.filters = filters
        self.pixel_size = psfm.get_pixel_size(self.info["ZBEST"])
        self.frames = self.get_frames(self.filters, self.fitss)
        if psf_res is not None:
            for f in self.frames:
                f.adjust_resolution(psf_res / self.pixel_size)
        for f in self.frames:
            f.calc_frames()
        self.target_flag = self.target_test(self.frames)
        if psf_res is None:
            self.update_frame_masks()
        self.flag_targets()
        for f in self.frames:
            f.calc_stmo()

    def get_frames(self, filters, fitss):
        """Calculates :obj:`frames` objects from provided fits data.

        For each filter and corresponding fits data obtains internal
        representation of the frame given by the :obj:`frame` class and
        returns them in a list.

        If lenght of :arg:`filters` and :arg:`fitss` don't match, returns
        an empty list.

        Args:
            filters (list of str): List of filter names.
            fitts (list of astropy.io.fits.HDUList): List of fits files corresponding to
                each of the filters.

        Returns:
            list: List of internal representations of frames given by the
                :obj:`frame` class.
        """
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
        """Tests whether identified targets in multiple frames are
        overlapping.

        By looking at calculated target segmentation maps across multiple
        frames determines whether the targets are overlapping and if not flags
        the cases as suspicious.

        Args:
            frames (list of :obj:`frame`): List of frames who's targets are to
                be evaluated.

        Returns:
            int: Flag corresponding to the overlap (0 - good, 1 - evaluation
                not possible, 2 - targets suspiciously different, 3 -
                targets largely non-ovelapping).
        """
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
        """Checks whether the blue frames have large enough segmentation mask
        and if not replaces their segmentation map with the average of
        the rest.

        Resolves difficulty of segmentation map creation for blue (=
        wavelength < 200 microns) filters by first checking whether they are
        suspiciously different from the rest, and if yes, replacing their
        target segmentation map and mask by the average of the others (more
        precisely by overlap of more than half of the other's segmentation
        maps/masks).

        Looks at and modifies all initiated frames of the galaxy.
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
        """For each of the frames determines whether its target segmentation
        map is comparable to the others and flags outliers.

        Calculates average segmentation map and mask across all frames of the
        galaxy (by marking pixel which feature in more than half of the
        segmantation maps/masks) and then for each frame checks whether there
        is large enough agreement with the average, and if not marks the case
        as problematic by modifying the appropriate flag of the :obj:`frame`
        object.
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
    """Class holding all data relating to a single frame/photo of a galaxy at
    some wavelength/in some filter needed to run statmorph, corresponding
    methods, etc.

    On initiation stores all data needed for running statmorph and other
    calculations on the image data which is then done by methods held by
    the class in multiple steps.

    Args:
        name (str): Name of the filter the frame was taken with.
        fits (astropy.io.fits.HDUList: Fits file corresponding the to image.

    Attributes:
        name (str): Name of the filter the frame was taken with.
        fits (astropy.io.fits.HDUList): Fits file corresponding the to image.
        data (numpy.array or similar): Data array contained in the fits file.
        psf (numpy.array or None): The point spread function corresponding to
            the filter. None if no psf was able to be obtained.
        adjustment (float or None): Information whether the resolution of the
            frame has been adjusted. If not `None`, then the value corresponds
            to the final resolution in lyr.
        flag_seg (int): Flag noting whether the target segmentation map used
            by this frame is consistent with the ones for other frames of the
            galaxy.
        convolved (numpy.array or similar): Slightly convolved version of the
            data used for segmentation map creation.
        objects_seg (numpy.array or similar): Full segmentation map of the
            data identifying both target and areas to be masked with different
            yet undecided indices.
        threshold (float): Threshold used for the creation of the segmentation
            map.
        cmax (tuple): x and y position of the identified centre of the target
            in the frame.
        data_sub (numpy.array or similar): Version of the data with subtracted
            background. If no subtraction is needed, same as :attr:`data`.
        target (numpy.array): Segmentation map identifying the target and
            nothing else.
        mask (numpy.array): Segmentation map identifying the masked area and
            nothing else.
        flag_corr (int): Flag noting whether the data is likely corrupted.
            (0 - fine, 1-3 - increasingly more likely)
        stmo (statmorph.SourceMorphology): Object holding results and methods
            of the statmorph calculations for the frame.
        bg_mask (numpy.array): Segmentation map identifying the background of
            the data.
        background (numpy.array): Background isolated from the data
            (everything else replaced by 0)
        bg_mean (float): Mean value of the background.
        bg_med (float): Median value of the background.
        bg_std (float): Standard deviation of the background.
    """

    def __init__(self, name, fits):
        """Initiates the class and stores all of the initially provided data."""
        self.name = name
        self.fits = fits
        self.data = self.fits[0].data
        self.psf = self.get_psf(self.name)
        self.adjustment = None
        self.flag_seg = 0

    def calc_frames(self):
        """Undertakes preparatory calculations of statmorph, most importantly
        determination of the target and mask segmentation maps.
        """
        self.convolved = self.convolve(self.data)
        self.objects_seg, self.threshold, self.cmax = self.segment(self.convolved)
        self.get_background(self.objects_seg, self.data)
        self.data_sub = self.bg_subtract(self.data)
        self.target, self.mask = self.isolate(self.objects_seg, self.cmax)
        self.flag_corr = self.get_corr_flag(self.data)

    def calc_stmo(self):
        """Runs statmorph calculation and stores the result."""
        self.stmo = self.get_stmo(self.data_sub, self.target, self.mask, self.psf)

    def adjust_resolution(self, fin_std):
        """Adjust the resolution of the frame image and its psf based on
        the requested final resolution.

        Starts by comparing the stddev of the currect psf and one of the
        requested final psf. If the latter is larger, than calculates what
        additional convolution of the data has to be done (in terms of stddev
        of gaussian kernel to be used) and applies it to the data and psf.
        Finally stores information of the adjustment into the
        :attr:`adjustment` attribute.

        Args:
            fin_std (float): Size of the required final psf standard
                deviation.
        """
        if self.psf is not None:
            stds = psfm.get_psf_std(self.name, self.psf)
            std_s = np.sum(stds) / 2
            if fin_std >= std_s:
                conv_std = np.sqrt(fin_std**2 - std_s**2)
                self.data = psfm.convolve_std(self.data, conv_std)
                self.psf = psfm.convolve_std(self.psf, conv_std)
                self.adjustment = fin_std
            else:
                warnings.warn(
                    f"Resolution of frame {self.name} is already lower than requested."
                )
        else:
            warnings.warn(
                f"Cannot adjust resolution of frame {self.name}, as it doesn't have a psf."
            )

    def convolve(self, data):
        """Slightly convolves given data.

        Takes inputted data and convolves it with Gaussian kernel with fwhm of
        ~1/25 the frame's dimensions.

        Args:
            data (numpy.array or similar): Data to be convolved

        Returns:
            numpy.array or similar: Convolved data.
        """
        dim = min(data.shape)
        fwhm = int(np.ceil(dim / 25) // 2 * 2 - 1)
        size = int(np.ceil(dim / 15) // 2 * 2 + 1)
        kernel = make_2dgaussian_kernel(fwhm, size=size)
        return convolve(data, kernel)

    def segment(self, data):
        """Creates segmentation map based on the inputted data.

        Finds threshold optimal for segmentation map creation based on methods
        in the `seg` module and uses it to create segmentation map with
        photutils tools.

        Also depending on the wavelength of the filter and whether its
        resolution has been adjusted, uses various levels of aggresivity
        in determining the threshold.

        Args:
            data (numpy.array or similar): Data to be segmented.

        Returns:
            tuple: A tuple of:

                * *numpy.array* - Segmentation map determined. Various indices
                  denoting various objects.
                * *float* - Threshold value used for the segmentation map
                  creation.
                * *tuple* - Tuple holding the x and y positions of the
                  suspected target's center.
        """
        # legacy method
        # sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
        # threshold = detect_threshold(data, nsigma=8.0, sigma_clip=sigma_clip)
        agr = int(self.name[1:-1]) < 150 and not self.adjustment
        cmax = seg.find_max(data)
        threshold = seg.find_threshold(data, cmax, agr)
        seg_map = detect_sources(data, threshold, 40)
        if seg_map is not None:
            return seg_map, threshold, cmax
        else:
            mean = np.mean(data)
            mp = (np.clip(data, None, mean) - mean).astype(bool).astype(int)
            return SegmentationImage(mp), threshold, cmax

    def get_background(self, seg_map, data):
        """Based on segmentation map gets the background of the image and
        calculates its statistics.

        By marking everything (-some footprint) not identified by the
        segmentation map, isolates the background and then obtains some of
        its statistics, which it stores as the class' values.

        Args:
            seg_map (numpy.array or similar): Segmentation map to be used for
                the background identification.
            data (numpy.array or similar): data from which the background is
                to be isolated
        """
        footprint = circular_footprint(radius=15)
        self.bg_mask = seg_map.make_source_mask(footprint=footprint)
        self.background = data * (1 - self.bg_mask)
        self.bg_mean, self.bg_med, self.bg_std = sigma_clipped_stats(
            self.data, sigma=3.0, mask=self.bg_mask
        )

    def bg_subtract(self, data):
        """Determine whether the data is background subtracted, and if not
        subtract it.

        Based on previously calculated background statistics, determines
        whether the frame's data is background subtracted. If yes, returns the
        inputted data, if  not, returns it with the median background
        value subtracted.

        Args:
            data (numpy.array or similar): Data to be potentially subtracted.

        Returns:
            numpy.array or similar: Subtracted (or not) data.
        """
        if self.bg_std > 0.1 * self.bg_med:
            return data
        else:
            warnings.warn("Subtracted background!")
            return data - self.bg_med

    def enlarge_mask(self, mask, target, radius=10):
        """Enlarges the provided mask.

        By taking circular footprint around it, enlarges the inputted mask,
        however also makes sure the the enlargement does not interfere with
        the areas directly around the target (taken as its circular footprint
        with twice the mask-enlargement radius).

        Args:
            mask (numpy.array or similar): Mask to be enlarged.
            target (numpy.array or similar): Target to be avoided in the
                enlargements.
            radius (int): Radius of the circular footprint to be used for
                enlargement in pixels.
        """
        fpm = circular_footprint(radius=radius)
        fps = circular_footprint(radius=2 * radius)
        mask_o = SegmentationImage(mask.astype(int))
        mask_l = mask_o.make_source_mask(footprint=fpm)
        sm_o = SegmentationImage(target)
        sm_l = sm_o.make_source_mask(footprint=fps)
        return (mask_l * (1 - sm_l) + mask).astype(bool)

    def isolate(self, seg_map, pos):
        """Isolate the target and mask in the segmentation map.

        Based on provided target position identifies the target in the
        provided segmentation mask and takes everything else as a mask. It
        then enlarges the latter and returns both.

        Args:
            seg_map (numpy.array or similar): Segmentation map which should
                have different objects marked with different integers.
            pos (tuple): x and y position denoting the centre of the target.

        Returns:
            tuple: A tuple consisting of:

                * *numpy.array* - Segmentation map identifying the target.
                * *numpy.array* - Segmentation map identifying the mask.
        """
        item = self.get_central(seg_map, pos)
        seg_map_t = np.invert((seg_map - item).astype(bool)).astype(int)
        mask = (seg_map - seg_map_t * item).astype(bool)
        mask = self.enlarge_mask(mask, seg_map_t)
        return seg_map_t, mask

    def get_central(self, seg_map, pos=None, margin=0.6):
        """Get the index of the target in the segmentation map.

        Identifies the target in the segmentation map by either looking at
        the index at the provided position or, if no position is provided,
        by taking the most "central" object of the segmentation map, where
        this is determined by taking square area centered at the image's
        center and iteratively increasing its size until some object of the
        segmentation map is covered by some percentage of its area given by
        the margin parameter.

        Args:
            seg_map (numpy.array or similar): Segmentation map from which the
                target is to be identified.
            pos (tuple): x and y position of the target
            margin (float): Float between 0 and 1 setting the margin used for
                the iterative process of identifying the most central target.

        Returns:
            int: Index of the identified target in the segmentation map.
        """
        if pos is not None and seg_map.data[pos] > 0:
            return seg_map.data[pos]
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
        """Counts how big area is spanned by an object in a given seg map.

        For each object in the given segmentation map determines the size
        of its area and writes it into a dictionary that is returned.

        Args:
            seg_map (numpy.array or similar): Segmentation map which's
                elements are to be measured.

        Returns:
            dict: Dictionary holding area of each object in the segmentation
                map.
        """
        unique, counts = np.unique(seg_map, return_counts=True)
        elements = dict(zip(unique, counts))
        elements.pop(0)
        return elements

    def get_stmo(self, data, seg_map, mask, psf=None):
        """Runs statmorph calculation.

        Based on previously calculated parameters, maps, etc. runs statmorph
        calculation and returns its results.

        Args:
            data (numpy.array): Data which will be the basis of the
                calculation.
            seg_map (numpy.array): Segmentation map identifying targets (in
                this case should be only one) to be used for the calculation.
            mask (numpy.array of bool): Segmentation map identifying which
                areas are to be masked for the calculation.
            psf (numpy.array): A psf to be used for the calculation. Not
                needed but increases the accuracy of the results.

        Returns:
            statmorph.SourceMorphology: Object holding results and methods
                of the statmorph calculation.
        """
        if psf is not None:
            return statmorph.source_morphology(
                data, seg_map, gain=1, mask=mask, psf=psf
            )[0]
        else:
            return statmorph.source_morphology(data, seg_map, gain=1, mask=mask)[0]

    def get_psf(self, name):
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
            path = f"../psf/webbpsf_NIRCam_{name}_pixsc25mas.fits"
            psf = astropy.io.fits.open(path)[0].data
            return psf
        except:
            warnings.warn(f"Haven't found psf for filter {name}.")
            return None

    def get_corr_flag(self, data):
        """Gets flag noting whether the given data is corrupted.

        Based on the amount of 0s in the inputted data, guesses whether it is
        likely corrupted by over-saturation or similar, or not.

        Args:
            data (numpy.array): Data to be checked for corruption.

        Returns:
            int: Flag denoting the likelihood of corruption (0 - fine, 1 -
                suspicious, 2 - likely corrupted, 3 - almost certainly
                corrupted).
        """
        zeros = np.sum(data == 0.0)
        total = np.prod(data.shape)
        if zeros > 0.8 * total:
            return 3
        elif zeros > 0.5 * total:
            return 2
        elif zeros > 0.1 * total:
            return 1
        else:
            return 0

    def show_seg(self):
        """Visualises some pre-requisites for the statmorph calculation.

        Using matplotlib plots and shows data and segmentation maps of target
        and background which are used as a basis for statmorph calculations.
        """
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
        """Shows visualisation of statmorph output or saves it.

        Calls statmorph's own method for result visualisation and if
        file-path is provided, saves it.

        Args:
            save_path (str): Path where the visualisation is to be saved.
                `None` if it is to be shown instead.
        """
        try:
            fig = make_figure(self.stmo)
            if save_path is not None:
                fig.savefig(save_path, dpi=200)
                plt.close(fig)
            else:
                plt.show()
        except:
            warnings.warn("Couldn't get the output image due to catastrophic flag.")

    def show_bg(self):
        """Plot and show histogram of the background.

        Calculates the histogram of the background, removing all 0/masked
        values and show it using matplotlib.
        """
        b = np.copy(self.background)
        rangev = (b.min(), b.max())
        b[b == 0.0] = -1000000
        counts, bins = np.histogram(b, range=rangev)
        plt.stairs(counts, bins)
        plt.show()
