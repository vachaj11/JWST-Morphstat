"""methods for creation of better segmentation map
"""

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

def convolution(data, mult = 1.0):
    dim = min(data.shape)
    fwhm = int(np.ceil(dim / 25*mult) // 2 * 2 - 1)
    size = int(np.ceil(dim / 15*mult) // 2 * 2 + 1)
    kernel = make_2dgaussian_kernel(fwhm, size=size)
    return convolve(data, kernel)
    
def highest_neighbour(data, pos):
    cropped = data[pos[0]-1:pos[0]+2,pos[1]-1:pos[1]+2]
    maxind = np.unravel_index(np.argmax(cropped),(3,3))
    pos_n_x = pos[0]+maxind[0]-1
    pos_n_y = pos[1]+maxind[1]-1
    return (pos_n_x,pos_n_y)
    
def find_max(data, start = None):
    maxsf = 0
    maxed = False
    if start is not None:
        pos = start
    else:
        pos = (int(data.shape[0]/2), int(data.shape[1]/2))
    while not maxed:
        pos_n = highest_neighbour(data, pos)
        maxsf = data[pos_n]
        if pos_n == pos:
            maxed = True
        else:
            pos = pos_n
    return pos
    
def get_tresholds(maxv, n=100):
    lnrange = np.linspace(0,np.log(maxv),n)
    rang = np.exp(lnrange)-1
    return rang
    
def find_edge(data, pos, thresh):
    sy = data.shape[1]
    for i in range(pos[1],sy-1):
        if data[pos[0],i+1] < thresh:
            return (pos[0],i)
    return (pos[0],sy-1)

def next_pos(data, pos, thresh, direc):
    poss = {
        0: (pos[0],pos[1]+1),
        1: (pos[0]-1,pos[1]),
        2: (pos[0],pos[1]-1),
        3: (pos[0]+1,pos[1]),
    }
    bigger = {}
    for k in poss:
        try:
            bigger[k] = data[poss[k]] > thresh
        except:
            bigger[k] = False
    for i in range(4):
        if bigger[(direc+i)%4]:
            return poss[(direc+i)%4], (direc+i-1)%4
    return pos, direc

def get_edge(data, pos0, thresh):
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
    blank = np.zeros(shape)
    for i in inds:
        blank[i] = 1
    return blank

def fill_area(edge_a):
    new_a = edge_a
    for r in range(len(edge_a)):
        inds = np.nonzero(edge_a[r])[0]
        for i in range(int(len(inds)/2)):
            new_a[r,inds[i*2]:inds[i*2+1]] = 1
    return new_a

def area_within(data, pos_o, thresh):
    enclosed = False
    pos = pos_o
    while not enclosed:
        pos0 = find_edge(data, pos, thresh)
        edge = get_edge(data, pos0, thresh)
        edge_a = area_from_inds(data.shape, edge)
        check_row = edge_a[pos_o[0],pos_o[1]:]
        check_sum = check_row.sum()
        if check_sum % 2 != 0:
            enclosed = True
        else:
            pos = [pos_o[0],pos_o[1]+np.max(np.nonzero(check_row))]
    area = fill_area(edge_a)
    return area
