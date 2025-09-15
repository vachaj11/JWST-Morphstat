"""Fairly unrelated module with methods to find a line which best separates two
sets of galaxies in some combination of two their parameters.

The main module here is :obj:`max_sep`.
"""

import time
from multiprocessing import Manager, Process
from threading import Thread

import numpy as np

import resu

ro = lambda s, an, r: np.tan(np.arctan(s / r) + an) * r + 10**-10
mo = lambda v, s, d: v + np.array([1 / s, -1]) / np.sqrt(1 + 1 / s**2) * d

def error_poly(ites, er, xlims, ylims, xax = True, mdif = None):
    if mdif is None:
        m = max(ites, key= lambda x: x[2])[2]
    else:
        m = mdif
    close = []
    for i in ites:
        if abs(i[2]-m) < er:
            close.append(i)
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    for i in close:
        xmins.append(i[0][1]+(xlims[0]-i[0][0])*i[1])
        xmaxs.append(i[0][1]+(xlims[1]-i[0][0])*i[1])
        ymins.append(i[0][0]+(ylims[0]-i[0][1])/i[1])
        ymaxs.append(i[0][0]+(ylims[1]-i[0][1])/i[1])
    if xax:
        return [(min(ymins),ylims[0]),(max(ymins),ylims[0]),(max(ymaxs), ylims[1]),(min(ymaxs),ylims[1])]
    else:
        return [(xlims[0],min(xmins)),(xlims[0],max(xmins)),(xlims[1],max(xmaxs)),(xlims[1],min(xmaxs))]
        
def error_poly_2(ites, er, xlims, ylims, rt = True, mdif = None):
    n = 1000
    xe = (xlims[1]-xlims[0])/20
    ye = (ylims[1]-ylims[0])/20
    intr = lambda p1, s1, p2, s2: ((p1[1]-p2[1]+s2*p2[0]-s1*p1[0])/(s2-s1),(p1[1]*s2-p2[1]*s1+s1*s2*(p2[0]-p1[0]))/(s2-s1))
    if mdif is None:
        m = max(ites, key= lambda x: x[2])[2]
    else:
        m = mdif
    close = []
    for i in ites:
        if abs(i[2]-m) < er:
            close.append(i)
    print(f"No. of error lines: {len(close)}")
    if rt:
        points0 = []
        points1 = []
        p0 = (xlims[1]+xe,ylims[1]+ye)
        p1 = (xlims[0]-xe,ylims[0]-ye)
        slopes = [np.tan(i/n*np.pi/2) for i in range(0,n)]
        for s in slopes:
            inters0 = []
            inters1 = []
            for b in close:
                inters0.append(intr(p0,s,b[0],b[1]))
                inters1.append(intr(p1,s,b[0],b[1]))
            points0.append(max(inters0, key = lambda x: x[0]))
            points1.append(min(inters1, key = lambda x: x[0]))
        if points0[-1][1]>ylims[0] and points1[0][0]<xlims[1]:
            points0.append((xlims[1],ylims[0]))
        if points0[0][0]>xlims[0] and points1[-1][1]<ylims[1]:
            points1.append((xlims[0],ylims[1]))
        return points0+points1
    else:
        points0 = []
        points1 = []
        p0 = (xlims[1]+xe,ylims[0]-ye)
        p1 = (xlims[0]-xe,ylims[1]+ye)
        slopes = [-np.tan(i/n*np.pi/2) for i in range(0,n)]
        for s in slopes:
            inters0 = []
            inters1 = []
            for b in close:
                inters0.append(intr(p0,s,b[0],b[1]))
                inters1.append(intr(p1,s,b[0],b[1]))
            points0.append(max(inters0, key = lambda x: x[0]))
            points1.append(min(inters1, key = lambda x: x[0]))
        if points0[-1][1]<ylims[1] and points1[0][0]<xlims[1]:
            points0.append((xlims[1],ylims[1]))
        if points0[0][0]>xlims[0] and points1[-1][1]>ylims[0]:
            points1.append((xlims[0],ylims[0]))
        return points0+points1
            
            
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


def best_parameters_l(gal1, gal2, parameters, e = None):
    parameters = list(parameters)
    vals = {}
    start = time.time()
    for i1 in range(len(parameters)):
        for i2 in range(i1, len(parameters)):
            starti = time.time()
            p, s, (d, _) = max_sep(gal1, gal2, parameters[i1], parameters[i2], e = e)
            vals[(parameters[i1], parameters[i2])] = [p, s, d]

            print(
                f"Finished {parameters[i1]} X {parameters[i2]} with {d:.3f} in {time.time()-starti:.2f} s."
            )
    print(f"Finished running in {time.time()-start:.2f} s.")
    return vals


def best_parameters_t(gal1, gal2, pars, e = None):
    pars = list(pars)
    vals = {}
    ts = {}
    start = time.time()
    for i1 in range(len(pars)):
        for i2 in range(i1, len(pars)):
            vals[(pars[i1], pars[i2])] = None
            ts[(pars[i1], pars[i2])] = Thread(
                target=threadp, args=(vals, gal1, gal2, pars[i1], pars[i2]), kwargs = {"e":e}
            )
            ts[(pars[i1], pars[i2])].start()
    for k in ts:
        ts[k].join()
    print(f"Finished running in {time.time()-start:.2f} s.")
    return vals


def best_parameters(gal1, gal2, pars, e = None):
    pars = list(pars)
    manag = Manager()
    vals = manag.dict()
    ts = {}
    start = time.time()
    for i1 in range(len(pars)):
        for i2 in range(i1, len(pars)):
            vals[(pars[i1], pars[i2])] = None
            ts[(pars[i1], pars[i2])] = Process(
                target=threadp, args=(vals, gal1, gal2, pars[i1], pars[i2]), kwargs = {"e":e}
            )
            ts[(pars[i1], pars[i2])].start()
    for k in ts:
        ts[k].join()
    print(f"Finished running in {time.time()-start:.2f} s.")
    return vals


def threadp(vals, gal1, gal2, p1, p2, e = None):
    start = time.time()
    p, s, (d, _, _) = max_sep(gal1, gal2, p1, p2, e = e)
    vals[(p1, p2)] = [p, s, d]
    print(f"Finished {p1} X {p2} with {d:.3f} in {time.time()-start:.2f} s.")


def max_sep(gal1, gal2, valx, valy, param=None, e = None, rfull = None):
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
    vo, so, mdif = max_dif_c((val1x, val1y), (val2x, val2y), vm, e = e,rfull=rfull)
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
    mdif = evalu([vx, vy], [vx, vy], point, slope, e = None)
    return mdif


def max_dif_c(v1, v2, vm, e = None, rfull = None):
    if v1[0] == v1[1]:
        vo, so, mdif = max_dif_s(v1, v2, e = e)
    else:
        voh, soh, mdifh = max_dif_s(v1, v2, 1, e = e, rfull = rfull)
        vov, sov, mdifv = max_dif_s(v1, v2, 0, e = e, rfull = rfull)
        vop, sop, mdifp = max_dif_2(v1, v2, e = e, rfull = rfull)
        options = [(vop, sop, mdifp), (voh, 10**-10, mdifh), (vov, 10**10, mdifv)]
        voi, soi, mdifi = max(options, key=lambda x: x[2][0])
        vom = voi + (np.array([1, soi]) * (vm - voi)).sum() * np.array([1, soi]) / (
            1 + soi**2
        )
        vo, so, mdif = max_dif(v1, v2, vom, soi, e = e, rfull = rfull)
        if mdifi > mdif:
            print("Just checking...:")
            print(f"{vo} -> {voi} , {so} -> {soi} , {mdif[0]} -> {mdifi[0]}")
            vo, so, mdif = voi, soi, mdifi
    return vo, so, mdif


def max_dif_s(v1, v2, ind=0, e = None, rfull = None):
    v1i = (v1[ind], v1[ind])
    v2i = (v2[ind], v2[ind])
    vs = np.concatenate((v1i[0], v2i[0]))
    vals = []
    for i in range(len(vs)):
        v = np.array([vs[i], vs[i]])
        s = 2
        ev = evalu(v1i, v2i, v, s, e = e)
        vals.append([v, s, ev])
        if rfull is not None:
            rfull.append([v,s,ev[0]])
    m = max(vals, key=lambda k: k[2][0])
    return (i for i in m)


def max_dif(v1, v2, v0, s0, e = None, rfull = None):
    xvals = np.concatenate((v1[0], v2[0]))
    yvals = np.concatenate((v1[1], v2[1]))
    rat = (yvals.max() - yvals.min()) / (xvals.max() - xvals.min())
    v = v0
    s = s0
    r = 55
    dist = np.abs(ldis(np.array(v1[0] + v2[0]), np.array(v1[1] + v2[1]), v0, s0))
    dd = np.min(dist[np.nonzero(dist)]) / (r + 1)
    dm = dist.min()
    while dd < dm:
        a = [(a * dd, b * dd) for a in range(-r, r + 1) for b in range(-r, r + 1)]
        vals = dict()
        for i in a:
            vn = mo(v, s, i[0])
            sn = ro(s, i[1] / dm * np.pi, rat)
            vals[i] = evalu(v1, v2, vn, sn, e=e)[0]
            if rfull is not None:
                rfull.append([vn, sn, vals[i]])
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
    return v, s, evalu(v1, v2, v, s, e = e)


def max_dif_2(v1, v2, e = None, rfull = None):
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
                er = evalu(v1, v2, v, s, e = e)
                if er[0] > ind[i][0]:
                    ind[i] = [er[0], s]
                if rfull is not None:
                    rfull.append([v, s, er[0]])
    m = max(ind, key=lambda x: ind[x][0])
    v = vals[m]
    s = ind[m][1]
    return v, s, evalu(v1, v2, v, s, e = e)


def evalu(v1, v2, v, s, e = None):
    v1x = np.array(v1[0])
    v1y = np.array(v1[1])
    v2x = np.array(v2[0])
    v2y = np.array(v2[1])
    dis1 = ldis(v1x, v1y, v, s)
    n1 = (dis1 > 0).sum()
    dis1 = n1 / len(dis1)
    dis2 = ldis(v2x, v2y, v, s)
    n2 = (dis2 <= 0).sum()
    dis2 = n2 / len(dis2)
    n = (n1 + (len(v2x)-n2))
    ntot = len(v1x)+len(v2x)
    tfra = len(v1x)/(len(v1x)+len(v2x))
    tfra = min(tfra, 1-tfra)
    lfra = n/ntot
    lfra = min(lfra, 1-lfra)
    if e is None or tfra - e < lfra < tfra + e:
        return np.abs(dis1 + dis2 - 1), (dis1, dis2), (lfra, tfra)
    else:
        return 0, (dis1, dis2), (lfra, tfra)

def ldis(vx, vy, v, s):
    sign = -s / np.abs(s)
    return ((vx - v[0]) - 1 / s * (vy - v[1])) / np.sqrt(1 + 1 / s**2) * sign
