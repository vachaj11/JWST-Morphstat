"""various methods for filtering and manipulating the results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import run
import psfm


def get_galaxy_entry(galaxies_full, gal_name, fil_names=None):
    """gets a galaxy of certain name and certain filters from a list"""
    for g in galaxies_full:
        if g["name"] == gal_name:
            ind = []
            for f in g["filters"]:
                if fil_names is not None and f not in fil_names:
                    ind.append(g["filters"].index(f))
            if len(ind) < len(g["filters"]):
                ind.reverse()
                for i in ind:
                    for nam in {"filters, files, fileInfo, frames"}:
                        if nam in g.keys():
                            g[name].pop(i)
                return g
            else:
                return None
    return None


def get_bad_frames(galaxy):
    """gets indexes of bad frames in a galaxy"""
    ind = []
    for f in galaxy["frames"]:
        if (
            f["flag"] > 1
            or f["_flag_seg"] > 0
            or f["flag_sersic"] > 2
            or not f["_psf_used"]
        ):
            ind.append(galaxy["frames"].index(f))
    ind.reverse()
    return ind


def frames_pruning(galaxy):
    """removes bad frames from a galaxy"""
    ind = get_bad_frames(galaxy)
    for i in ind:
        galaxy["frames"].pop(i)
        galaxy["filters"].pop(i)
    return galaxy


def galaxy_pruning(galaxies):
    """removes problematic galaxies and frames from an output list of
    galaxies
    """
    gal_out = []
    for g in galaxies:
        if g["misc"]["target_flag"] >= 0:  # filter disabled
            gp = frames_pruning(g)
            if len(gp["filters"]) > 0:
                gal_out.append(gp)
    return gal_out


def get_bad_cases(galaxies_full, galaxies_out):
    """creates input list of problematic galaxies frames from output
    list and input list of galaxies
    """
    gal_bad = []
    for g in galaxies_out:
        ind = get_bad_frames(g)
        filters = []
        for i in ind:
            filters.append(g["filters"][i])
        if filters:
            gal_b = get_galaxy_entry(galaxies_full, g["name"], filters)
            if gal_b is not None:
                gal_bad.append(gal_b)
    return gal_bad


def json_bad_cases(path_full, path_out, path_new):
    """from old input+output creates new input json with problematic cases"""
    gal_full = run.fetch_json(path_full)["galaxies"]
    gal_out = run.fetch_json(path_out)["galaxies"]
    gal_new = get_bad_cases(gal_full, gal_out)
    run.save_as_json({"galaxies": gal_new}, path_new)


def get_subset(galaxies_out, galaxies_in):
    """get subset of '_out' galaxies that matches set of '_in' galaxies"""
    galaxies_sub = []
    for gi in galaxies_in:
        go = next((g for g in galaxies_out if g["name"] == gi["name"]), None)
        if go is not None:
            galaxies_sub.append(go)
    return galaxies_sub


def get_complement(galaxies_out, galaxies_in):
    """get complement of '_out' galaxies that is not in the '_in' galaxies"""
    galaxies_com = []
    for go in galaxies_out:
        gi = next((g for g in galaxies_in if g["name"] == go["name"]), None)
        if gi is None:
            galaxies_com.append(go)
    return galaxies_com


def json_subset(path_out, path_in, path_new):
    """from input file specifing subset of output file's galaxies create a
    subset output json
    """
    gal_out = run.fetch_json(path_out)["galaxies"]
    gal_in = run.fetch_json(path_in)["galaxies"]
    gal_new = get_subset(gal_out, gal_in)
    run.save_as_json({"galaxies", gal_new}, path_new)


def get_separate_in_value(galaxies, value):
    """separate galaxies based on requested value"""
    values = {}
    for g in galaxies:
        if value in g["info"].keys():
            val = g["info"][value]
        elif value in g["misc"].keys():
            val = g["misc"][value]
        else:
            val = None
        if val in values.keys():
            values[val].append(g)
        else:
            values[val] = [g]
    return values


def get_filter_or_avg(galaxy, value, filt):
    """depending on provided parameter/name of filter either return value for
    the galaxy in a given filter or averaged across filters
    works for both input and output galaxy list formats
    also if filt is of the form rfwXXX, takes XXX to be rest frame wavelength
    and uses the closest-matching filter
    """
    if filt == "avg":
        val = 0
        for i in range(len(galaxy["filters"])):
            for k in galaxy.keys():
                f = galaxy[k]
                if type(f) == list and len(f) == len(galaxy["filters"]):
                    if type(f[i]) == dict and value in f[i].keys():
                        val += float(f[i][value])
        if val:
            val = val / len(galaxy["filters"])
            return val
        else:
            return None
    if filt[:3] == "rfw":
        rfw = float(filt[3:])
        filters = [int(f[1:-1]) for f in galaxy["filters"]]
        diff = [abs(rfw - f) / f for f in filters]
        ind = diff.index(min(diff))
        filt = galaxy["filters"][ind]
    if filt in galaxy["filters"]:
        i = galaxy["filters"].index(filt)
        for k in galaxy.keys():
            if (
                len(galaxy[k]) == len(galaxy["filters"])
                and type(galaxy[k]) == list
                and type(galaxy[k][i]) == dict
            ):
                if value in galaxy[k][i].keys():
                    return float(galaxy[k][i][value])
        return None
    else:
        return None


def get_most_filters(galaxies, no):
    """for the given set of galaxies, finds the first n most common filters"""
    filt_names = []
    for g in galaxies:
        filt_names += g["filters"]
    unique, counts = np.unique(filt_names, return_counts=True)
    no_w_filt = zip(unique, counts)
    filt_cut = [l[0] for l in sorted(no_w_filt, key=lambda i: i[1], reverse=True)][:no]
    return filt_cut


def get_outliers(galaxies, value, sig=3, filt="avg"):
    """for provided galaxies find outliers of n-sigma in the given value"""
    data = []
    ind = []
    for g in galaxies:
        val = get_filter_or_avg(g, value, filt)
        if val:
            data.append(val)
            ind.append(galaxies.index(g))
    mean = np.mean(data)
    std = np.std(data)
    gal_bad = []
    for i in range(len(data)):
        if data[i] - mean < -sig * std or data[i] - mean > sig * std:
            gal_bad.append(galaxies[ind[i]])
    return gal_bad


def print_closest(pos, data, fig=None):
    """prints the name of the galaxy closest to the position"""
    if not (pos[0] and pos[1]):
        return None
    if fig is not None:
        ax = fig.axes[0]
        wy = abs(ax.get_xlim()[0] - ax.get_xlim()[1])
        wx = abs(ax.get_ylim()[0] - ax.get_ylim()[1])
    else:
        wx = 1
        wy = 1
    if data:
        closest = data[0][2]
        distance = abs(data[0][0] - pos[0]) * wx + abs(data[0][1] - pos[1]) * wy
    else:
        print("No data to show")
        return None
    for d in data:
        coor = [d[0] - pos[0], d[1] - pos[1]]
        dist = np.sqrt((coor[0] * wx) ** 2 + (coor[1] * wy) ** 2)
        if dist < distance:
            closest = d[2]
            distance = dist
    print(closest)


def get_galaxy(name):
    """find galaxy of a given name and calculate stmo for it returning full
    results object
    """
    filj = run.fetch_json("dictionary_full.json")["galaxies"]
    gal_entry = get_galaxy_entry(filj, name)
    return run.calculate_stmo(gal_entry)


def get_optim_rfw(galaxies, return_filtered = False):
    """determines the optimum rest frame wavelength to be used for the
    provided set of galaxies
    """
    vals = np.linspace(25, 270, num=400)
    z = [g["info"]["ZBEST"] for g in galaxies]
    filters = [[int(f[1:-1]) for f in g["filters"]] for g in galaxies]
    v_best = 0
    diff_best = len(galaxies)
    gals_best = []
    for v in vals:
        diff = 0
        gals = []
        for i in range(len(galaxies)):
            v_r = v * (1 + z[i])
            diffs = [abs(v_r - v_i) / v_i for v_i in filters[i]]
            if len(diffs) > 0:
                ind = diffs.index(min(diffs))
                diff += diffs[ind]
                gal = dict()
                gi = galaxies[i]
                for k in galaxies[i]:
                    if type(gi[k]) == list and len(gi[k]) == len(diffs):
                        gal[k] = [gi[k][ind]]
                    else:
                        gal[k] = gi[k]
                gals.append(gal)
                    
        if diff < diff_best:
            v_best = v
            diff_best = diff
            gals_best = gals
    if return_filtered:
        return v_best, gals_best
    else:
        return v_best
        
def get_maximal_std_distance(galaxies):
    """Determines the maximal std (in lyr) of psf present in the given set
    of galaxies.
    """
    max_std = 0.
    for g in galaxies:
        px_size = psfm.get_pixel_size(g["info"]["ZBEST"])
        for f in g["filters"]:
            psf_stds_px = psfm.get_psf_std(f)
            if psf_stds_px is not None:
                psf_std_px = np.sum(psf_stds_px)/2
                psf_std = px_size*psf_std_px
                max_std = max(psf_std, max_std)
    return max_std
