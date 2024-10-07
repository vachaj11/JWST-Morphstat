"""Fairly unrelated module with methods to find a line which best separates two
sets of galaxies in some combination of two their parameters.

The main module here is :obj:`max_sep`.
"""

import numpy as np
import resu

ro = lambda s, an: np.tan(np.arctan(s) + an)


def get_vals(gal, valx, valy):
    vsx = []
    vsy = []
    galo = []
    for g in gal:
        vx = resu.get_filter_or_avg(g, valx, filt="avg")
        vy = resu.get_filter_or_avg(g, valy, filt="avg")
        if vx is not None and vy is not None:
            vsx.append(vx)
            vsy.append(vy)
            galo.append(g)
    return vsx, vsy, galo

def best_parameters(gal1, gal2, parameters):
    parameters = list(parameters)
    vals = {}
    for i1 in range(len(parameters)):
        for i2 in range(i1, len(parameters)):
            p, s, d = max_sep(gal1, gal2, parameters[i1], parameters[i2])
            vals[(parameters[i1],parameters[i2])] = [p, s, d]
    return vals
            

def max_sep(gal1, gal2, valx, valy, param=None):
    """For two sets of galaxies and two parameters, finds a line which best
    separates the two sets of galaxies in the parameter space of the two
    parameters.
    If `param` is passed, it also stores the distance to the final line in the
    information (under the key `param`) of all galaxies in the two sets.
    """
    val1x, val1y, gal1f = get_vals(gal1, valx, valy)
    val2x, val2y, gal2f = get_vals(gal2, valx, valy)
    v1m = (np.mean(val1x), np.mean(val1y))
    v2m = (np.mean(val2x), np.mean(val2y))
    v0 = np.array(((v1m[0] + v2m[0]) / 2, (v1m[1] + v2m[1]) / 2))
    slope = (v1m[1] - v2m[1]) / (v1m[0] - v2m[0])
    s0 = -1 / slope
    vo, so, mdif = max_dif((val1x, val1y), (val2x, val2y), v0, s0, same = valx==valy)
    if param is not None:
        for i in range(len(gal1f)):
            gal1f[i]["frames"][0][param] = ldis(val1x[i], val1y[i], vo, so)
        for i in range(len(gal2f)):
            gal2f[i]["frames"][0][param] = ldis(val2x[i], val2y[i], vo, so)
    return vo, so, mdif

def get_above_line(gal, valx, valy, point, slope):
    vx, vy, _ = get_vals(gal, valx, valy)
    vx = np.array(vx)
    vy = np.array(vy)
    dist = ldis(vx, vy, point, slope)
    ratio = (dist > 0).sum() / len(dist)
    return ratio

def max_dif(v1, v2, v0, s0, same = False):
    xvals = np.concatenate((v1[0],v2[0]))
    yvals = np.concatenate((v1[1],v2[1]))
    rang = np.array([xvals.max()-xvals.min(),yvals.max()-yvals.min()])/np.pi
    v = v0
    s = s0
    dd = 0.001
    while dd < 4:
        if same:
            a = [
                (-dd, 0),
                (0, 0),
                (dd, 0),
            ]
        else:
            a = [
                (-dd, -dd),
                (-dd, 0),
                (-dd, dd),
                (0, -dd),
                (0, 0),
                (0, dd),
                (dd, -dd),
                (dd, 0),
                (dd, dd),
            ]
        vals = dict()
        for i in a:
            vals[i] = evalu(v1, v2, v + i[0]*rang, ro(s, i[1]))
        mval = max(vals, key=vals.get)
        if mval != (0, 0):
            v = v + mval[0]*rang
            s = ro(s, mval[1])
            dd = 0.001
        else:
            dd *= 1.1
    return v, s, evalu(v1, v2, v,s)


def evalu(v1, v2, v, s):
    v1x = np.array(v1[0])
    v1y = np.array(v1[1])
    v2x = np.array(v2[0])
    v2y = np.array(v2[1])
    dis1 = ldis(v1x, v1y, v, s)
    dis1 = (dis1 > 0).sum() / len(dis1)
    dis2 = ldis(v2x, v2y, v, s)
    dis2 = (dis2 > 0).sum() / len(dis2)
    return np.abs(dis1 - dis2)


def ldis(vx, vy, v, s):
    return ((vx - v[0]) - 1 / s * (vy - v[1])) / np.sqrt(1 + 1 / s**2)
