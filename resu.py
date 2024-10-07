"""Various methods for filtering and manipulating the result/input 
dictionaries.

This modules holds methods with logic for both preparing the input lists of 
dictionaries (each representing a galaxy) for statmorph computation as well as
working with the results. Both in terms of basic set operations, filtering
using some specific parameters, finding optimal rest frame-wavelenghts and
psf resolution, etc. 
There are also a few mothods for data extraction from the dictionaries used
by the visualisation methods in :obj:`vis`.
"""

import copy
import json

import cv2
import numpy as np

import psfm
import run


def get_galaxy_entry(galaxies_full, gal_name, fil_names=None):
    """Gets a galaxy of certain name and certain filters from a list.

    From a list of galaxies/dictionaries extracts a galaxy of a given name
    and returns it, leaving also only some subset of its filters if specified.

    Args:
        galaxies_full (list): List of dictionaries each representing one
            input/output galaxy.
        gal_name (str): Name of the galaxy to be extracted.
        fil_names (list): List of str each name of a filter to be included in
            the extracted dictionary.

    Returns:
        dict: A dictionary with the extracted galaxy data. `None` if no
            galaxy of the specified name was found in the passed list or
            it didn't have any specified filters.
    """
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


def get_bad_frames(galaxy, strength="normal"):
    """Gets indexes of bad frames in a galaxy.

    Based on various flags and parameters decides which frames in the passed
    galaxy are "bad" and returns a list of their indices.
    There are 4 different settings of the "bad" boundary: "light", "normal",
    "strict" and "sersic".

    Args:
        galaxy (dict): Dictionary representing the galaxy who's frames are to
            be checked.
        strength (str): Setting of the boundary strength. By default "normal".

    Returns:
        list: List of indices of the bad frames in the galaxy.
    """
    ind = []
    if "frames" not in galaxy.keys():
        return ind
    for f in galaxy["frames"]:
        if strength == "normal" and (
            f["flag"] > 1
            or f["flag_sersic"] > 2
            or f["_flag_seg"] > 0
            or f["_flag_corr"] > 2
            or not f["_psf_used"]
        ):
            ind.append(galaxy["frames"].index(f))
        elif strength == "strict" and (
            f["flag"] > 0
            or f["flag_sersic"] > 1
            or f["_flag_seg"] > 0
            or f["_flag_corr"] > 1
            or not f["_psf_used"]
        ):
            ind.append(galaxy["frames"].index(f))
        elif strength == "sersic" and (
            f["flag"] > 2
            or f["flag_sersic"] > 0
            or f["_flag_corr"] > 2
            or not f["_psf_used"]
        ):
            ind.append(galaxy["frames"].index(f))
        elif strength == "light" and (
            f["flag"] > 2 or f["_flag_corr"] > 2 or not f["_psf_used"]
        ):
            ind.append(galaxy["frames"].index(f))
    ind.reverse()
    return ind


def frames_pruning(galaxy, strength="normal"):
    """Removes bad frames from a galaxy.

    Based on various flags and characteristics of results decides which
    frames in a galaxy are "bad" and removes them.

    Args:
        galaxy (dict): Dictionary representing the galaxy who's "bad" frames
            are to be removed.
        strength (str): Denotes the strength used in deciding which frames
            are bad. Passed to :obj:`get_bad_frames`.

    Returns:
        dict: Dictionary representing the galaxy with the "bad" frames
            removed.
    """
    ind = get_bad_frames(galaxy, strength)
    gal = dict()
    for k in galaxy.keys():
        if type(galaxy[k]) == list and len(galaxy[k]) == len(galaxy["filters"]):
            gal[k] = []
            for i in range(len(galaxy[k])):
                if i not in ind:
                    gal[k].append(galaxy[k][i])
        else:
            gal[k] = galaxy[k]
    return gal


def galaxy_pruning(galaxies, strength="normal"):
    """Removes problematic galaxies and frames from an output list of
    galaxies.

    Based on various flags and characteristics of results decides which
    galaxies and frames in them are "bad" and removes them.

    Args:
        galaxies (list): List of dictionaries representing the galaxies who's
            "bad" frames are to be removed.
        strength (str): Denotes the strength used in deciding which frames
            are bad. Passed to :obj:`frames_pruning` and then
            :obj:`get_bad_frames`.

    Returns:
        list: List of dictionaries representing the galaxies with the "bad"
            frames removed.
    """
    gal_out = []
    gal_good = []
    for g in galaxies:
        if "_flag_rfw" not in g["info"].keys() or g["info"]["_flag_rfw"] < 1:
            gal_good.append(g)
    for g in gal_good:
        gp = frames_pruning(g, strength)
        if len(gp["filters"]) > 0:
            gal_out.append(gp)
    return gal_out


def get_bad_cases(galaxies_full, galaxies_out):
    """Creates a list of problematic galaxies and frames from
    output/results list and its superset input list of galaxies.

    Gets what results were unreliable in calculated frames' data and creates
    an input list of galaxies with only those problematic cases.
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
    """From old input+output creates new input json with problematic cases.

    As with :obj:`get_bad_cases` but works with ``.json`` files rather than
    input/output lists.
    """
    gal_full = run.fetch_json(path_full)["galaxies"]
    gal_out = run.fetch_json(path_out)["galaxies"]
    gal_new = get_bad_cases(gal_full, gal_out)
    run.save_as_json({"galaxies": gal_new}, path_new)


def get_subset(galaxies_out, galaxies_in):
    """Get subset of '_out' galaxies that matches set of '_in' galaxies.

    A simple set operation allowing e.g. for results to be filtered based
    on categorisation in terms of input lists.
    """
    galaxies_sub = []
    for gi in galaxies_in:
        go = next((g for g in galaxies_out if g["name"] == gi["name"]), None)
        if go is not None:
            galaxies_sub.append(go)
    return galaxies_sub


def get_complement(galaxies_out, galaxies_in):
    """Get complement of '_out' galaxies that is not in the '_in' galaxies.

    A simple set operation allowing e.g. for results to be filtered based
    on categorisation in terms of input lists.
    """
    galaxies_com = []
    for go in galaxies_out:
        gi = next((g for g in galaxies_in if g["name"] == go["name"]), None)
        if gi is None:
            galaxies_com.append(go)
    return galaxies_com


def json_subset(path_out, path_in, path_new):
    """From input file specifing subset of output file's galaxies create a
    subset output json.

    As with :obj:`get_subset` but works with ``.json`` files rather than
    input/output lists.
    """
    gal_out = run.fetch_json(path_out)["galaxies"]
    gal_in = run.fetch_json(path_in)["galaxies"]
    gal_new = get_subset(gal_out, gal_in)
    run.save_as_json({"galaxies", gal_new}, path_new)


def get_separate_in_value(galaxies, value):
    """Separate galaxies based on requested value.

    Based on the requested value to be found in galaxies' information,
    separates the inputted list into dictionary entries with different values
    found as keys.
    """
    values = {}
    for g in galaxies:
        if value in g["info"].keys():
            val = g["info"][value]
        else:
            val = None
        if val in values.keys():
            values[val].append(g)
        else:
            values[val] = [g]
    return values

def get_bins_in_value(galaxies, value, bins=5):
    """!!!
    """
    vals = []
    for g in galaxies:
        if value in g["info"].keys():
            vals.append(g["info"][value])
    lin = np.linspace(min(vals), max(vals), bins+1)
    gals = {(lin[i], lin[i+1]):[] for i in range(len(lin)-1)}
    for g in galaxies:
        if value in g["info"].keys():
            v = g["info"][value]
            for k in gals:
                if k[0] <= v < k[1]:
                    gals[k].append(g)
    return gals
            

def get_filter_or_avg(galaxy, value, filt="avg"):
    """Depending on provided parameter/name of filter either return value for
    the galaxy in a given filter or averaged across filters.

    Baseline function for getting data from the dictionaries representing
    galaxies, used by most visualisation methods in :obj:`vis`.
    Works for both input and output galaxy list formats.
    Also if `filt` is of the form "rfwXXX", takes "XXX" to be rest frame
    wavelength and uses the closest-matching filter, and if `filt` is "-1",
    then takes the filter with the highest wavelength.

    Args:
        galaxy (dict): Dictionary representing the galaxy from which the
            value is to be extracted.
        value (str): The name of the value to be extracted.
        filt (str): The type of extraction used. Can be either "avg" in which
            case an average across filters is used, "rfwXXX" in which case a
            filter closest to the specified rfw is used, "-1" for the filter
            with the longest wavelength, or directly a name of a filter.

    Returns:
        float: The desired extracted value. Can also be `None` if the
            extraction failed for whatever reason (e.g. the specified filter
            is not present for the given galaxy).
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
    if filt[:3] == "rfw":
        rfw = float(filt[3:])
        filters = [int(f[1:-1]) for f in galaxy["filters"]]
        diff = [abs(rfw - f) / f for f in filters]
        ind = diff.index(min(diff))
        filt = galaxy["filters"][ind]
    elif filt == "-1":
        filt = galaxy["filters"][-1]
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
    if value in galaxy["info"].keys():
        return galaxy["info"][value]
    return None


def get_most_filters(galaxies, no):
    """For the given set of galaxies, finds the first `no` most common
    filters and returns list of their names.
    """
    filt_names = []
    for g in galaxies:
        filt_names += g["filters"]
    unique, counts = np.unique(filt_names, return_counts=True)
    no_w_filt = zip(unique, counts)
    filt_cut = [l[0] for l in sorted(no_w_filt, key=lambda i: i[1], reverse=True)][:no]
    return filt_cut


def get_outliers(galaxies, value, sig=3, filt="avg"):
    """For provided galaxies find outliers of n-sigma in the given value.

    Using :obj:`get_filter_or_avg` gets the specified value for each galaxy,
    finds outliers in this value in the set larger than `sig`*sigma and
    returns a list of the galaxies with outlying data in the value.
    """
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
    """Prints the name of the galaxy closest to the position.

    Used to evaluate what galaxy is being clicked at in an x-y plot. From
    list of galaxy names and corresponding x and y values, finds which is
    the closest to the passed `pos` x and y values.
    """
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


def get_galaxy(name, psf_res=None):
    """Find galaxy of a given name and calculate stmo for it returning full
    results object.

    Used to quickly run the statmorph calculation for a single galaxy larger
    than larger sample. Useful for checking e.g. how the segmentation map
    looked for some problematic cases.
    """
    filj = run.fetch_json("dictionary_full.json")["galaxies"]
    gal_entry = get_galaxy_entry(filj, name)
    return run.calculate_stmo(gal_entry, psf_res=psf_res)


def get_optim_rfw(
    galaxies, M_mul=2.0, rfw_range=(0.25, 2.70), fixed_rfw=None, bad=True
):
    """Determines the optimum rest frame wavelength to be used for the
    provided set of galaxies.

    Goes through a list of possible rfw values and for each calculates what
    would be the difference for each galaxy between the observed wavelength
    and the closest filter. Then sums these differences and chooses such rfw
    which minimises the sum.
    Also directly return a list of galaxies with only the
    closest-matching filters included, and the differences as a function of
    rfw.
    By default also discriminates against middle-size band filters by
    multiplying the difference to them by 2
    All the values of wavelength worked with and returned are in microns.

    If `fixed_rfw` parameter is provided, then does not calculated best rfw
    but rather uses the value of the parameter and return a list of galaxies
    with only the closest-matching filters.
    """
    galaxies = copy.deepcopy(galaxies)
    vals = np.linspace(rfw_range[0], rfw_range[1], num=400)
    v_best = 0
    diff_best = len(galaxies)
    gals_best = []
    rfw_diff = []
    if fixed_rfw is not None:
        return get_rfw_difference(fixed_rfw, galaxies, M_mul, bad_add=bad)[1]
    for v in vals:
        diff, gals = get_rfw_difference(v, galaxies, M_mul, bad_add=bad)
        if diff < diff_best:
            v_best = v
            diff_best = diff
            gals_best = copy.deepcopy(gals)
        rfw_diff.append(diff)
    return v_best, gals_best, (vals, rfw_diff)


def get_rfw_difference(rfw, galaxies, M_mul, bad_add=True):
    """For a given rest frame wavelength and a set of galaxies, calculates
    what are the filters for each galaxy closest-matching the rfw and what is
    the sum of their differences to the rfw.
    Returns the sum of differences and a list of galaxies including only the
    closest filters.

    Should be called through :obj:`get_optim_rfw`, otherwise the input list
    of galaxies might be modified.
    """
    diff = 0
    gals = []
    for g in galaxies:
        z = g["info"]["ZBEST"]
        filts = g["filters"]
        v_r = rfw * (1 + z)
        diffs = []
        for f in filts:
            v_i = int(f[1:-1]) / 100
            if f[-1:] == "W":
                diffs.append(abs(v_r - v_i) / v_r)
            else:
                diffs.append(M_mul * abs(v_r - v_i) / v_r)
        if len(diffs) > 0:
            ind = diffs.index(min(diffs))
            diff += diffs[ind]
            gal = dict()
            for k in g:
                if type(g[k]) == list and len(g[k]) == len(diffs):
                    gal[k] = [g[k][ind]]
                else:
                    gal[k] = g[k]
            if diffs[ind] > 0.6:
                rfw_flag = 3
            elif diffs[ind] > 0.4:
                rfw_flag = 2
            elif diffs[ind] > 0.2:
                rfw_flag = 1
            else:
                rfw_flag = 0
            if filts[ind][-1:] == "M":
                M_flag = 1
            else:
                M_flag = 0
            if "info" in gal.keys():
                gal["info"]["_flag_rfw"] = rfw_flag
                gal["info"]["_flag_rfw_M"] = M_flag
            if bad_add or (not rfw_flag):
                gals.append(gal)
                diff += diffs[ind]
    return diff / len(gals) * len(galaxies), gals


def get_maximal_psf_width(galaxies, return_full=True):
    """Determines the maximal width of psf (in kpc) present in the given
    set of galaxies.

    For the set of galaxies and filters they are imaged at, finds for each
    the width (fwhm) of its psf in kiloparsecs (i.e. translated to the
    corresponding rest-frame position using cosmological calculations) and
    returns the largest one.
    Alternatively with option `return_full` can also return a list consisting
    of the largest psf, the pixel size (in kpc) of the corresponding galaxy
    and the width of the psf (again in kpc).

    Args:
        galaxies (list): List of dictionaries representing the galaxies who's
            among which the maximal fwhm distance is to be found.
        return_full (bool): If true instead of only maximum fwhm distance a
            list is returned including 1) the psf corresponding to the
            maximum psf width 2) size of one pixel for case of maximum psf
            fwhm in kpc 3) the maximum psf width in kpc.

    Returns:
        float: the maximum psf width in kpc
    """
    max_fwhm = 0.0
    psf_m = []
    for g in galaxies:
        px_size = psfm.get_pixel_size(g["info"]["ZBEST"])
        for f in g["filters"]:
            psf_fwhms_px = psfm.get_psf_fwhm(f)
            if psf_fwhms_px is not None:
                psf_fwhm_px = np.sum(psf_fwhms_px) / 2
                psf_fwhm = px_size * psf_fwhm_px
                if psf_fwhm >= max_fwhm:
                    max_fwhm = psf_fwhm
                    if return_full:
                        psf = psfm.get_psf(f)
                        psf_m = [psf, px_size, psf_fwhm]
    if psf_m:
        return psf_m
    else:
        return max_fwhm


def manual_reev_c(
    galaxies, im_path="../statmorph_images_filtered/name.png", out_path=None
):
    """Allows for reevaluation of flagging of a given set of galaxies. For
    each galaxy opens its statmorph image output if available and asks user
    for new flag in interactive cli.
    The cli accepts values 0, 1, 2, 3, 4 for flags, and <, >, <<, iXXX for
    moving around the set of galaxies.
    (the default path is very ad hoc)
    """
    gals = copy.deepcopy(galaxies)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.waitKey(0)
    i = 0
    while i < len(gals):
        g = gals[i]
        for l in range(len(g["filters"])):
            s = False
            name = g["name"] + "_" + g["filters"][l]
            ims = cv2.imread(im_path.replace("name", name))
            if ims is not None:
                cv2.imshow("img", ims)
                cv2.waitKey(1)
                inp = input(
                    f"Value for {g['name']}_{g['filters'][l]} ({i}/{len(gals)}): "
                )
                if inp == "<":
                    i -= 2
                    break
                elif inp in {">", ""}:
                    i = i
                    break
                elif inp == "<<":
                    i = -1
                    break
                elif inp in {"0", "1", "2", "3", "4"}:
                    g["frames"][l]["flag"] = int(inp)
                elif inp[:1] == "i":
                    i = int(inp[1:]) - 1
                else:
                    print("unknown input " + inp)

        i += 1
        if out_path is not None and i % 10 == 0:
            run.save_as_json({"galaxies": gals}, out_path)
    if out_path is not None:
        run.save_as_json({"galaxies": gals}, out_path)
    return gals
