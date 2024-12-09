"""Fairly unrelated module with methods to find a line which best separates two
sets of galaxies in some combination of two their parameters.

The main module here is :obj:`max_sep`.
"""

import time
from multiprocessing import Manager, Process
from threading import Thread

import numpy as np

import resu

ro = lambda s, an, r: np.tan(np.arctan(s / r) + an) * r
mo = lambda v, s, d: v + np.array([1 / s, -1]) / np.sqrt(1 + 1 / s**2) * d


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


def best_parameters_l(gal1, gal2, parameters):
    parameters = list(parameters)
    vals = {}
    start = time.time()
    for i1 in range(len(parameters)):
        for i2 in range(i1, len(parameters)):
            starti = time.time()
            p, s, (d, _) = max_sep(gal1, gal2, parameters[i1], parameters[i2])
            vals[(parameters[i1], parameters[i2])] = [p, s, d]

            print(
                f"Finished {parameters[i1]} X {parameters[i2]} with {d} in {time.time()-starti} s."
            )
    print(f"Finished running in {time.time()-start} s.")
    return vals


def best_parameters_t(gal1, gal2, pars):
    pars = list(pars)
    vals = {}
    ts = {}
    start = time.time()
    for i1 in range(len(pars)):
        for i2 in range(i1, len(pars)):
            vals[(pars[i1], pars[i2])] = None
            ts[(pars[i1], pars[i2])] = Thread(
                target=threadp, args=(vals, gal1, gal2, pars[i1], pars[i2])
            )
            ts[(pars[i1], pars[i2])].start()
    for k in ts:
        ts[k].join()
    print(f"Finished running in {time.time()-start} s.")
    return vals


def best_parameters(gal1, gal2, pars):
    pars = list(pars)
    manag = Manager()
    vals = manag.dict()
    ts = {}
    start = time.time()
    for i1 in range(len(pars)):
        for i2 in range(i1, len(pars)):
            vals[(pars[i1], pars[i2])] = None
            ts[(pars[i1], pars[i2])] = Process(
                target=threadp, args=(vals, gal1, gal2, pars[i1], pars[i2])
            )
            ts[(pars[i1], pars[i2])].start()
    for k in ts:
        ts[k].join()
    print(f"Finished running in {time.time()-start} s.")
    return vals


def threadp(vals, gal1, gal2, p1, p2):
    start = time.time()
    p, s, (d, _) = max_sep(gal1, gal2, p1, p2)
    vals[(p1, p2)] = [p, s, d]
    print(f"Finished {p1} X {p2} with {d} in {time.time()-start} s.")


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
    vm = np.array([v1m[0] + v2m[0], v1m[1] + v2m[1]]) / 2
    # v0 = np.array(((v1m[0] + v2m[0]) / 2, (v1m[1] + v2m[1]) / 2))
    # slope = (v1m[1] - v2m[1]) / (v1m[0] - v2m[0])
    # s0 = -1 / slope
    vo, so, mdif = max_dif_c((val1x, val1y), (val2x, val2y), vm)
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


def max_dif_c(v1, v2, vm):
    if v1[0] == v1[1]:
        vo, so, mdif = max_dif_s(v1, v2)
    else:
        voh, soh, mdifh = max_dif_s(v1, v2, 1)
        vov, sov, mdifv = max_dif_s(v1, v2, 0)
        vop, sop, mdifp = max_dif_2(v1, v2)
        options = [(vop, sop, mdifp), (voh, 10**-10, mdifh), (vov, 10**10, mdifv)]
        voi, soi, mdifi = max(options, key=lambda x: x[2][0])
        vom = voi + (np.array([1, soi]) * (vm - voi)).sum() * np.array([1, soi]) / (
            1 + soi**2
        )
        vo, so, mdif = max_dif(v1, v2, vom, soi)
        if mdifi > mdif:
            print("Just checking...:")
            print(f"{vo} -> {voi} , {so} -> {soi} , {mdif[0]} -> {mdifi[0]}")
            vo, so, mdif = voi, soi, mdifi
    return vo, so, mdif


def max_dif_s(v1, v2, ind=0):
    v1i = (v1[ind], v1[ind])
    v2i = (v2[ind], v2[ind])
    vs = np.concatenate((v1i[0], v2i[0]))
    vals = []
    for i in range(len(vs)):
        v = np.array([vs[i], vs[i]])
        s = 2
        vals.append([v, s, evalu(v1i, v2i, v, s)])
    m = max(vals, key=lambda k: k[2][0])
    return m[0], m[1], m[2]


def max_dif(v1, v2, v0, s0):
    xvals = np.concatenate((v1[0], v2[0]))
    yvals = np.concatenate((v1[1], v2[1]))
    rat = (yvals.max() - yvals.min()) / (xvals.max() - xvals.min())
    v = v0
    s = s0
    r = 100
    dist = np.abs(ldis(np.array(v1[0] + v2[0]), np.array(v1[1] + v2[1]), v0, s0))
    dd = np.min(dist[np.nonzero(dist)]) / (r + 1)
    dm = dist.min()
    while dd < dm:
        a = [(a * dd, b * dd) for a in range(-r, r + 1) for b in range(-r, r + 1)]
        vals = dict()
        for i in a:
            vals[i] = evalu(v1, v2, mo(v, s, i[0]), ro(s, i[1] / dm * np.pi, rat))[0]
        mval = [k for k in vals if max(vals.values()) == vals[k]]
        if (0, 0) not in mval:
            ind = mval[0]
            v = mo(v, s, ind[0])
            s = ro(s, ind[1] / dm * np.pi, rat)
            dist = np.abs(ldis(np.array(v1[0] + v2[0]), np.array(v1[1] + v2[1]), v, s))
            dd = np.min(dist[np.nonzero(dist)]) / (r + 1)
            dm = dist.min()
        else:
            dd *= r / 5

    return v, s, evalu(v1, v2, v, s)


def max_dif_2(v1, v2):
    xvals = np.concatenate((v1[0], v2[0]))
    yvals = np.concatenate((v1[1], v2[1]))
    rat = (yvals.max() - yvals.min()) / (xvals.max() - xvals.min())
    vals = np.column_stack((xvals, yvals))
    vers = [[1, 50], [0.5, 100], [0.1, 500], [0.02, 2000], [0.004, 5000]]
    le = len(xvals)
    ind = {i: [0, 1] for i in range(le)}
    for it in vers:
        sort = sorted(ind, key=lambda x: ind[x][0])
        slopes = [np.tan((0.5 + i) * np.pi / it[1]) * rat for i in range(it[1])]
        for i in sort[: int(it[0] * le)]:
            v = vals[i]
            for s in slopes:
                e = evalu(v1, v2, v, s)
                if e[0] > ind[i][0]:
                    ind[i] = [e[0], s]
    m = max(ind, key=lambda x: ind[x][0])
    v = vals[m]
    s = ind[m][1]
    return v, s, evalu(v1, v2, v, s)


def evalu(v1, v2, v, s):
    v1x = np.array(v1[0])
    v1y = np.array(v1[1])
    v2x = np.array(v2[0])
    v2y = np.array(v2[1])
    dis1 = ldis(v1x, v1y, v, s)
    dis1 = (dis1 > 0).sum() / len(dis1)
    dis2 = ldis(v2x, v2y, v, s)
    dis2 = (dis2 <= 0).sum() / len(dis2)
    return np.abs(dis1 + dis2 - 1), (dis1, dis2)


def ldis(vx, vy, v, s):
    sign = -s / np.abs(s)
    return ((vx - v[0]) - 1 / s * (vy - v[1])) / np.sqrt(1 + 1 / s**2) * sign
