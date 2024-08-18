"""Methods for creation of segmentation map suited for the specific purposes
and characteristics of the project.

This module holds a list of methods that are used to determine a brightness
threshold to be used for segmentation map creation. 
The overall approach for this is to iteratively increase the threshold in 
each step checking:

    * What is the relative increase in the area q = area/area_old with respect
      to previous step.
    * What is the ratio q = threshold/mean_brightness
    
When the combination of these values goes beyond some border curve, the value
from last iteration is taken as the final threshold value.

The main method running this iterative process is :obj:`find_threshold` and 
the border at which the iterative process is stopped is defined by the methods
:obj:`func` and :obj:`threshold_reached`.
The remaining methods are various substeps, including methods labeled *Legacy*
which define a legacy way of finding a segmentation area _given_ some 
threshold. This has been replaced by a faster `photutils` method producing 
mostly identical results.
Separately, there is also a method :obj:`find_max` and related, which 
determine local maximum by climbing up a gradient.
"""

import json
import time
import warnings

import astropy
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import convolve
from photutils.segmentation import (
    SegmentationImage,
    detect_sources,
    detect_threshold,
    make_2dgaussian_kernel,
)
from photutils.utils import circular_footprint


def convolution(data, mult=1.0):
    """Smoothes given data by colvolving with a gaussian kernel of appropriate
    size.

    From the size of the image determines a size of gaussian kernel to smooth
    the image with (about ~1/25 of the image's size times the optional
    parameter) and returns their convolution.

    Args:
        data (numpy.array or similar): Data to be smoothed.
        mult (float): Multiplier of the size of gaussian kernel to be used for
            the smoothing. 1.0 by default, higher values, the smoother the
            image becomes.

    Returns:
        numpy.array: Smoothed out data.
    """
    dim = min(data.shape)
    fwhm = int(np.ceil(dim / 25 * mult) // 2 * 2 - 1)
    size = int(np.ceil(dim / 15 * mult) // 2 * 2 + 1)
    kernel = make_2dgaussian_kernel(fwhm, size=size)
    return convolve(data, kernel)


def highest_neighbour(data, pos):
    """Finds highest neighbour of the current position.

    Lookes at the 8 neighbours of the passed position in the data and
    returns the position of the highest one. If none is higher than the
    initial position, returns the initial.

    Args:
        data (numpy.array or similar): Data in which the positions are to be
            evaluated.
        pos (tuple): x and y values of the position who's neighbours are to
            be checked.

    Returns:
        tuple: x and y values of the neighbouring position with the highest
            value.
    """
    cropped = data[pos[0] - 1 : pos[0] + 2, pos[1] - 1 : pos[1] + 2]
    try:
        maxind = np.unravel_index(np.argmax(cropped), (3, 3))
        pos_n_x = pos[0] + maxind[0] - 1
        pos_n_y = pos[1] + maxind[1] - 1
        return (pos_n_x, pos_n_y)
    except ValueError:
        warnings.warn("Finding image peak escaped image boundaries.")
        return pos


def find_max(data, start=None):
    """By climbing up gradient from start position finds a local maximum.

    First smoothes out the data slightly by convolving by gaussian kernel and
    then iteratively climbs up the steepest gradient until it reaches a peak
    (=no neighbours with higher values).

    Args:
        data (numpy.array or similar): Data where the peak is to be looked up.
        start (tuple or None): x and y values of the position to be started
            from. If None, the most central position in the data is used
            instead.

    Returns:
        tuple: x and y values (indices in the data array) of the position of
            the maximum found.
    """
    maxsf = 0
    maxed = False
    data = convolution(data, 2)
    if start is not None:
        pos = start
    else:
        pos = (int(data.shape[0] / 2), int(data.shape[1] / 2))
    if data[pos] == 0:
        warnings.warn("found suspicious values when searching for peak")
        data = convolution(data, 10)
    while not maxed:
        pos_n = highest_neighbour(data, pos)
        maxsf = data[pos_n]
        if pos_n == pos:
            maxed = True
        else:
            pos = pos_n
    return pos


def find_edge(data, pos, thresh):
    """From some start position, finds the first position on the right of it
    where a given threshold value is reached.

    *Legacy*

    Starts at the passed position and then iteratively goes right until a
    value lower than a threshold is reached at which point returns the
    position one left of it. If the threshold is not crossed, returns the most
    right position in the data.

    Args:
        data (numpy.array or similar): Data where the edge-position is to be
            searched for.
        pos (tuple): x and y values of position from which the search is to be
            started.
        thresh (float): Value of the threshold defining the edge to be found.

    Returns:
        tuple: x and y values to the threshold-edge position found.
    """
    sy = data.shape[1]
    for i in range(pos[1], sy - 1):
        if data[pos[0], i + 1] < thresh:
            return (pos[0], i)
    return (pos[0], sy - 1)


def next_pos(data, pos, thresh, direc):
    """Gets next position along the edge of an area given by a threshold
    value.

    *Legacy*

    Starting from the passed position and direction looks clockwise at the
    neighbouring values and returns first which is above the given threshold.
    Also returns direction, which is one step counterclockwise from the
    direction having been moved in.
    If no neighbouring position is found to be above the threshold, the
    original position and direction are returned.

    Args:
        data (numpy.array or similar): Data array on which the next position
            is to be found.
        pos (tuple): x and y values of the start position as indices in the
            data array.
        thresh (float): The threshold value used to find the next position.
        direc (int): Direction in which the next position is first going to be
            searched for. 0-4 starting from right and going clockwise

    Returns:
        tuple: A tuple consisting of:

            * *tuple* - x and y values of the next position found.
            * *int* - Integer denoting the direction search for next value
              should be initiated in.
    """
    shape = data.shape
    poss = {
        0: (pos[0], pos[1] + 1),
        1: (pos[0] - 1, pos[1]),
        2: (pos[0], pos[1] - 1),
        3: (pos[0] + 1, pos[1]),
    }
    bigger = {}
    for k in poss:
        if 0 <= poss[k][0] < shape[0] and 0 <= poss[k][1] < shape[1]:
            bigger[k] = data[poss[k]] >= thresh
        else:
            bigger[k] = False
    for i in range(4):
        if bigger[(direc + i) % 4]:
            return poss[(direc + i) % 4], (direc + i - 1) % 4
    return pos, direc


def get_edge(data, pos0, thresh):
    """From some start position gets a list of edge positions of an
    area bordering a threshold.

    *Legacy*

    By iteratively asking for a next position using an algorithm equivalent to
    "follow the uninterupted wall on your right hand", determines a closed
    area (runs until the initial position is reached) bordering a given
    threshold.

    Args:
        data (numpy.array or similar): Data in which the edge given by a
            threshold is to be found.
        pos0 (tuple): x and y position defining the starting position of the
            threshold edge search. Right of it (in the data array) should be
            a value lower than the threshold, otherwise the method gets into
            an infinite loop.
        thresh (float): The threshold value to be used for the determination
            of the edge.

    Returns:
        list: List of tuples, each holding the x and y values of one position
            on the edge-loop.
    """
    edge = [pos0]
    closed = False
    direc = 0
    pos = pos0
    while not closed:
        pos, direc = next_pos(data, pos, thresh, direc)
        if pos == pos0:
            closed = True
        else:
            edge.append(pos)
    return edge


def area_from_inds(shape, inds):
    """From list of indices denoting path positions, creates a map of the
    path in an array denoting different slopes/directions of the path with
    different values.

    *Legacy*

    Starts at an array filled with zeros and then going along the passed path
    for each position of the path increases the corresponding value of the
    array by a value depending on the direction of transition of the path
    across the position. This indexing is subsequently used to easily
    identify an area enclosed by the path.

    Args:
        shape (tuple): Dimensions of the array in which the path is set in x
            and y directions.
        inds (list): List of tuples each containing the x and y values of one
            of the positions along the path.

    Returns:
        numpy.array: Array with the path visualised according to its
            direction.
    """
    blank = np.zeros(shape)
    li = len(inds)
    for i in range(li):
        val = blank[inds[i]]
        pre = inds[(i - 1) % li]
        pos = inds[(i + 1) % li]
        val += pos[0] - pre[0]
        val += (pos[1] - pre[1]) * 2
        if val:
            blank[inds[i]] = val
        else:
            blank[inds[i]] = 4
    return blank


def fill_area(edge_a):
    """Uniformly fills an area given by a map of path in an array.

    *Legacy*

    Takes an array which contains a path map denoted by non-zero-integers as
    created by the :obj:`area_from_inds` method, and fills its area using the
    values of the path and some modulo algebra by going from left to right
    across each line of the array.
    (this with the above are quite nice methods for filling an area
    enclosed by path I think, but didn't end up being used)

    Args:
        edge_a (numpy.array or similar): An array with a path denoted by
            integer values corresponding to path's direction as created by
            :obj:`area_from_inds`.

    Returns:
        numpy.array: An array with the path filled in (with 1s) and everything
            else 0.
    """
    new_a = np.zeros(edge_a.shape)
    for r in range(len(edge_a)):
        count = 0
        for c in range(len(edge_a[r])):
            count += edge_a[r, c]
            if edge_a[r, c]:
                new_a[r, c] = 1
            elif count % 4 == 2:
                new_a[r, c] = 1
    return new_a


def area_within_legacy(data, pos_o, thresh):
    """Gets an area falling within a given threshold and including the
    provided point.

    *Legacy*

    Obtains an area of values higher than a given threshold that is
    uninterupted (& all its holes are filled) and includes the initial point.
    This is done by first finding an edge of the area right of the starting
    point, then following the edge along the area's border (i.e. the
    threshold) and finally filling the found edge-path. If this area does is
    found not to include the starting point, then the process is repeated by
    looking for the next more-right crossing of the threshold as a starting
    point of the edge, or the border of the image.

    Args:
        data (numpy.array or similar): Data array who's values are to be used
            to determine the area.
        pos_o (tuple): x and y position of the starting point which is to be
            included in the segmented area.
        thresh (float): Threshold value defining the area's border.

    Returns:
        numpy.array: Array with the same dimensions as the data and with the
            segmented area denoted by 1s and background with 0s.
    """
    enclosed = False
    pos = pos_o
    while not enclosed:
        time.sleep(1)
        pos0 = find_edge(data, pos, thresh)
        edge = get_edge(data, pos0, thresh)
        edge_a = area_from_inds(data.shape, edge)
        check_row = edge_a[pos_o[0], pos_o[1] :]
        check_sum = check_row.sum()
        if check_sum % 4 == 2 or (len(edge) == 1 and edge[0] == pos_o):
            enclosed = True
        else:
            pos = (pos_o[0], pos_o[1] + np.max(np.nonzero(check_row)))
    area = fill_area(edge_a)
    return area


def area_within(data, pos_o, thresh):
    """Gets an area falling within a given threshold and including the
    provided point.

    *Legacy*

    Obtains an area of values higher than a given threshold that is
    uninterupted (& all its holes are filled) and includes the initial point.
    This is done by first creating multiple-object segmentation map at that
    threshold with photutils methods, then selecting the "object" area that
    includes the provided point and scrapping others.
    Some edge cases are: i) If the value at the provided position is lower
    than the threshold, then error is raised. ii) If the segmentation map
    cannot be created e.g. for lack of continuous area, then an empty array
    with only 1 at the provided position is created. iii) If for some reason
    the created segmentation map does not include the provided point, the
    legacy method :obj:`area_within_legacy` is used instead.

    Args:
        data (numpy.array or similar): Data array who's values are to be used
            to determine the area.
        pos_o (tuple): x and y position of the starting point which is to be
            included in the segmented area.
        thresh (float): Threshold value defining the area's border.

    Returns:
        numpy.array: Array with the same dimensions as the data and with the
            segmented area denoted by 1s and background with 0s.
    """
    if data[pos_o] < thresh:
        raise Exception("Value at a specified target position is bellow threshold.")
    seg = detect_sources(data, thresh, 1)
    if seg is None:
        bl = np.zeros(data.shape)
        bl[pos_o] = 1
        return bl
    elif not seg.data[pos_o]:
        warnings.warn("using legacy area calculation")
        return area_within_legacy(data, pos_o, thresh)
    else:
        sd = seg.data
        return np.invert((sd - sd[pos_o]).astype(bool)).astype(int)


def func(x, aggressive=False):
    """Straightforward function defining the threshold boundary curve.

    Defines the curve of area_ratio(petri_ratio) above which the increments
    are classified as large enough for the threshold brightness to be reached.
    (more in the module description)
    The specific form of the curve was determined for the specific dataset.
    There is also an option for more aggresive, lower-positioned boundary,
    which was designed for more-clumpy and bluer frames.

    Args:
        x (float): The value representing the petri-ratio at the iteration
            step, i.e. the ratio between the threshold brightness and the mean
            enclosed brightness.
        aggressive (bool): Whether a more aggressive, stricter/lower boundary
            curve should be used.

    Returns:
        float: Value corresponding to the boundary area-ratio of the boundary
            curve at the given petri-ratio.
    """
    if aggressive:
        return 1 + ((x - 0.1) * 2.5) ** 2 * 0.5
    else:
        return 1 + ((x - 0.1) * 2.5) ** 2


def threshold_reached(petr, ar, aggressive=False):
    """Based on petri-ratio and the increase in area, determines whether
    the desired threshold has been reached.

    From the petri-ratio (threshold/mean brightness) and relative increase in
    area at an iteration, determines whether the combination of values is
    sufficient for the iteration to be said to reach the desired threshold.

    Args:
        petr (float): Value of the petri-ratio.
        ar (float): Value of the relative increase in area.
        aggressive (bool): Whether a more aggressive, stricter/lower boundary
            curve should be used.

    Returns:
        bool: `True` if the boundary for threshold has been reached, `False`
            otherwise.
    """
    if petr > 0.5:
        return False
    elif petr < 0.1:
        return True
    elif ar >= func(petr, aggressive=False):
        return True
    else:
        return False


def get_thresholds(data, maxv, n=1000):
    """Creates a list of evenly (w.r.t. data) spread threshold values.

    Based on passed data and some local maximum brightness (of found local
    peak in the data, around which the iterative process will revolve)
    determines evenly spread threshold brightness values which can serve as
    a basis of the iterative process. The number of the values is 1000 by
    default and they are spaced w.r.t. the brightness distribution of the
    data, i.e. they are the values of the 10th, 17th, 24th, etc. brightest
    pixel in the data. Also there are lowest/highest brightness values w.r.t.
    the data imposed. The highest from the local maximum value passed and the
    lowest as the 33% percentile brightness in the data.

    Args:
        data (numpy.array or similar): Data, the distribution of which is
            going to be used for spacing of the brightness threshold values.
        maxv (float): Brightness value of a local maximum in the data, which
            is going to be used to impose a limit on the values of threshold.
        n (int): Number of values to be included in the list of thresholds,
            i.e. the finesse of the spacing.

    Returns:
        list: List of float values corresponding to the evenly spaced
            threshold values.
    """
    flat = data.flatten()
    sort = np.sort(flat)[::-1]
    max_ind = np.argmax(sort < maxv)
    incr = (len(sort) / 3 - max(max_ind, len(sort) / 250)) / n
    if incr < 0:
        incr = (len(sort) - max_ind - 1) / n
    thrs = []
    for i in range(n - 1):
        thrs.append(sort[max_ind + int(i * incr)])
    thrs.append(0.0)
    return thrs


def find_threshold(data, pos_o, aggressive=False):
    """By iteration determines threshold at which only the galaxy is
    included in the segmentation map.

    Using an iterative method as described in the docstring of this module,
    determines a threshold at which a small increase in the threshold
    brightness yields a large increase in the area of the target in the
    segmentation map, at the same time assuming that enough of the galaxy is
    already enclosed within the threshold brightness (this is to avoid clumps
    being identified as external objects, and is done by simultaneously also
    looking at the petri-ratio).
    The method works by first getting a suitable list of threshold values and
    then going through them and at each step looking at the outcome
    segmentation map and checking the above-described parameters until their
    sufficient combination is found.

    Args:
        data (numpy.array or similar): Data array serving as the basis for the
            whole process, i.e. giving the brightness distributions, basis for
            the segmentation maps creation, etc.
        pos_o (tuple): x and y positions of the centre (or rather value with
            highest brightness value) of the galaxy. Used to check that
            a correct object in the segmentation maps is considered.
        aggressive (bool): Option to use a more aggressive selection of
            threshold brightness value. Tuned to bluer-frames, however at them
            even with this set to true, the approach can still turn out
            unreliable, e.g. taking clumps as external objects to the galaxy.

    Returns:
        float: The threshold brightness value found to be optimal for creating
        the segmentation map of the galaxy.
    """
    thresholds = get_thresholds(data, data[pos_o])
    ar_o = np.prod(data.shape)
    th_o = 1
    pr_o = 1
    for th in thresholds:
        area = area_within(data, pos_o, th)
        petri = th * area.sum() / (area * data).sum()
        arear = (area.sum()) / ar_o
        if threshold_reached(min(pr_o, petri), arear, aggressive):
            return th_o
        else:
            ar_o = area.sum()
            th_o = th
            pr_o = petri
    return th_o
