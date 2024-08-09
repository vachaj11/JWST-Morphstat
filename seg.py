"""methods for creation of better segmentation map
"""

import warnings
import json
import astropy
import time
import matplotlib.pyplot as plt
import numpy as np

from astropy.convolution import convolve
from photutils.segmentation import (
    make_2dgaussian_kernel,
    detect_sources,
    detect_threshold,
    SegmentationImage,
)
from photutils.utils import circular_footprint


def convolution(data, mult=1.0):
    """convolves the data with a gaussian kernel (its size determined by the
    size of the frame)
    """
    dim = min(data.shape)
    fwhm = int(np.ceil(dim / 25 * mult) // 2 * 2 - 1)
    size = int(np.ceil(dim / 15 * mult) // 2 * 2 + 1)
    kernel = make_2dgaussian_kernel(fwhm, size=size)
    return convolve(data, kernel)


def highest_neighbour(data, pos):
    """finds highest neighbour of the current position"""
    cropped = data[pos[0] - 1 : pos[0] + 2, pos[1] - 1 : pos[1] + 2]
    try:
        maxind = np.unravel_index(np.argmax(cropped), (3, 3))
        pos_n_x = pos[0] + maxind[0] - 1
        pos_n_y = pos[1] + maxind[1] - 1
        return (pos_n_x, pos_n_y)
    except ValueError:
        print("finding image peak escaped image boundaries")
        return pos


def find_max(data, start=None):
    """by climbing up gradient from start position finds a local maximum"""
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


def get_thresholds(data, maxv, n=1000):
    """creates list of evenly (w.r.t. data) spread threshold values"""
    flat = data.flatten()
    sort = np.sort(flat)[::-1]
    max_ind = np.argmax(sort < maxv)
    incr = (len(sort) / 3 - max(max_ind, len(sort) / 250)) / n
    thrs = []
    for i in range(n - 1):
        thrs.append(sort[max_ind + int(i * incr)])
    thrs.append(0.0)
    return thrs


def find_edge(data, pos, thresh):
    """from some start position, finds the first position on the right of it
    where a given threshold value is reached
    """
    sy = data.shape[1]
    for i in range(pos[1], sy - 1):
        if data[pos[0], i + 1] < thresh:
            return (pos[0], i)
    return (pos[0], sy - 1)


def next_pos(data, pos, thresh, direc):
    """gets next position along the edge of an area given by a threshold value
    (and direction to be continued in in the next step)
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
            bigger[k] = data[poss[k]] > thresh
        else:
            bigger[k] = False
    for i in range(4):
        if bigger[(direc + i) % 4]:
            return poss[(direc + i) % 4], (direc + i - 1) % 4
    return pos, direc


def get_edge(data, pos0, thresh):
    """from some start position, data and threshold calculates an edge
    postions of an area that is within that threshold and includes the
    start point
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
    """from list of indices denoting path, creates an area of the path
    denoting different slopes/directions of the path by different values
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
    """fills an area given by an area array with edge-path uniformly"""
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
    """gets area falling within a given threshold and including the
    provided point
    """
    enclosed = False
    pos = pos_o
    while not enclosed:
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
    """gets area falling within a given threshold and including the
    provided point; uses photutils so much faster than the above
    """
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


def func(x, aggresive=False):
    """simple function defining the threshold boundary"""
    if aggresive:
        return 1 + ((x - 0.1) * 2.5) ** 2 * 0.5
    else:
        return 1 + ((x - 0.1) * 2.5) ** 2


def threshold_reached(petr, ar, aggresive=False):
    """based on petri ratio and increase in area determines whether threshold
    has been reached
    """
    if petr > 0.5:
        return False
    elif petr < 0.1:
        return True
    elif ar >= func(petr, aggresive=False):
        return True
    else:
        return False


def find_threshold(data, pos_o, aggresive=False):
    """by iteration determines threshold at which only the galaxy is included"""
    thresholds = get_thresholds(data, data[pos_o])
    ar_o = np.prod(data.shape)
    th_o = 1
    pr_o = 1
    for th in thresholds:
        area = area_within(data, pos_o, th)
        petri = th * area.sum() / (area * data).sum()
        arear = (area.sum()) / ar_o
        if threshold_reached(min(pr_o, petri), arear):
            return th_o
        else:
            ar_o = area.sum()
            th_o = th
            pr_o = petri
    return th_o
