"""various methods for filtering and manipulating the results
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import run


def get_galaxy_entry(galaxies_full, gal_name, fil_names):
    """gets a galaxy of certain name and certain filters from a list"""
    for g in galaxies_full:
        if g["name"] == gal_name:
            ind = []
            for f in g["filters"]:
                if f not in fil_names:
                    ind.append(g["filters"].index(f))
            if len(ind) < len(g["filters"]):
                ind.reverse()
                for i in ind:
                    g["filters"].pop(i)
                    g["files"].pop(i)
                    g["fileInfo"].pop(i)
                return g
            else:
                return None
    return None


def get_bad_frames(galaxy):
    """gets indexes of bad frames in a galaxy"""
    ind = []
    for f in galaxy["frames"]:
        if f["flag"] > 0:
            ind.append(galaxy["frames"].index(f))
    ind.reverse()
    return ind


def frames_pruning(galaxy):
    """removes bad frames from a galaxy """
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
        if g["misc"]["target_flag"] == 1:
            gal_out.append(frames_pruning(g))
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
    """from old input+output creates new input json with problematic cases """
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

def json_subset(path_out, path_in, path_new):
    """from input file specifing subset of output file's galaxies create a
    subset output json
    """
    gal_out = run.fetch_json(path_out)["galaxies"]
    gal_in = run.fetch_json(path_in)["galaxies"]
    gal_new = get_subset(gal_out, gal_in)
    run.save_as_json({"galaxies", gal_new}, path_new)
    
def get_filter_or_avg(galaxy, value, filt):
    """depending on provided parameter/name of filter either return value for
    the galaxy in a given filter or averaged across filters
    """
    if filt == "aver":
        val = 0
        for f in galaxy["frames"]:
            val += f[value]
        if val:
            val = val / len(galaxy["frames"])
            return val
        else:
            return None
    elif filt in galaxy["filters"]:
        return galaxy["frames"][galaxy["filters"].index(filt)][value]
    else:
        return None

def get_most_filters(galaxies, no):
    """for the given set of galaxies, finds the first n most common filters
    """
    filt_names = []
    for g in galaxies:
        filt_names += g["filters"]
    unique, counts = np.unique(filt_names, return_counts=True)
    no_w_filt = zip(unique, counts)
    filt_cut = [l[0] for l in sorted(no_w_filt, key= lambda i: i[1], reverse = True)][:no]
    return filt_cut    

def get_outliers(galaxies, value, sig = 3, filt = "avg"):
    """for provided galaxies find outliers of n-sigma in the given value
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
        if data[i]-mean < -sig*std or data[i]-mean> sig*std:
            gal_bad.append(galaxies[ind[i]])
    return gal_bad
