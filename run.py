"""various methods with logic for fetching and building up data for/from the
statmorph computation
"""
import json
import astropy
import stmo
import time
import matplotlib.pyplot as plt
import numpy as np


def galaxies_data(dict_path, path_out=None, return_object = False):
    """Main function that computes statmorph result for an inputted 
    dictionary
    """
    t0 = time.time()
    x = fetch_json(dict_path)
    galaxies = []
    objects = []
    for i in range(len(x["galaxies"])):
        gdata = calculate_stmo(x["galaxies"][i])
        galaxies.append(get_galaxy_data(gdata))
        if return_object:
            objects.append(gdata)
        print(
            "Finished galaxy {} in {:.2f} s ({} out of {})".format(
                galaxies[-1]["name"], time.time() - t0, i + 1, len(x["galaxies"])
            )
        )
        if i % 10 == 0 and path_out is not None:
            save_as_json({"galaxies": galaxies}, path_out)
    if path_out is not None:
        save_as_json({"galaxies": galaxies}, path_out)
    if return_object:
        return objects
    else:
        return galaxies

def calculate_stmo(galaxy):
    """Runs statmorph and returns stmo.galaxy object for the given galaxy"""
    info = galaxy["info"]
    name = galaxy["name"]
    filters, fitss = get_fitss(galaxy)
    gdata = stmo.galaxy(name, info, filters, fitss)
    return gdata

def adhoc_path(path):
    """ad-hoc modifies the path to file according to current setup
    """
    p = "../galfit_results/" + path[:-4]
    p = p.replace("big/", "big/results_")
    p = p.replace("small/", "small/results_")
    return p

def get_fitss(galaxy):
    """gets fits files of galaxy in different wavelengths
    """
    filters = []
    fitss = []
    for l in range(len(galaxy["filters"])):
        path = adhoc_path(galaxy["files"][l])
        try:
            fitss.append(astropy.io.fits.open(path))
            filters.append(galaxy["filters"][l])
        except:
            print("Couldn't locate "+path)
    return filters, fitss

def save_as_json(thing, path):
    """saves provided object as json at specified path
    """
    fil = open(path, "w")
    json.dump(thing, fil, indent=4)
    fil.close()


def fetch_json(path):
    """fetches data of json object at specified path
    """
    fil = open(path, "r")
    data = json.load(fil)
    fil.close()
    return data

def get_galaxies_data(galaxies):
    """gathers data from stmo.galaxy objects in output format 
    """
    gals = []
    for galaxy in galaxies:
        gald = get_galaxy_data(galaxy)
        gals.append(gald)
    return gals

def get_galaxy_data(galaxy):
    """extracts data from the stmo.galaxy object and formats them into the 
    output format
    """
    gald = dict()
    gald["name"] = galaxy.name
    gald["filters"] = galaxy.filters
    gald["info"] = galaxy.info
    misc = {"target_flag": galaxy.target_flag}
    gald["misc"] = misc
    frames = []
    for frame in galaxy.frames:
        data = get_frame_data(frame)
        frames.append(data)
    gald["frames"] = frames
    return gald


def get_frame_data(frame):
    """extracts data from a stmo.frame object and formats it into the output
    format
    """
    st = frame.stmo
    data = {
        "xc_centroid": float(st.xc_centroid),
        "yc_centroid": float(st.yc_centroid),
        "ellipticity_centroid": float(st.ellipticity_centroid),
        "elongation_centroid": float(st.elongation_centroid),
        "orientation_centroid": float(st.orientation_centroid),
        "xc_asymmetry": float(st.xc_asymmetry),
        "yc_asymmetry": float(st.yc_asymmetry),
        "ellipticity_asymmetry": float(st.ellipticity_asymmetry),
        "elongation_asymmetry": float(st.elongation_asymmetry),
        "orientation_asymmetry": float(st.orientation_asymmetry),
        "rpetro_circ": float(st.rpetro_circ),
        "rpetro_ellip": float(st.rpetro_ellip),
        "rhalf_circ": float(st.rhalf_circ),
        "rhalf_ellip": float(st.rhalf_ellip),
        "r20": float(st.r20),
        "r80": float(st.r80),
        "Gini": float(st.gini),
        "M20": float(st.m20),
        "F(G, M20)": float(st.gini_m20_bulge),
        "S(G, M20)": float(st.gini_m20_merger),
        "sn_per_pixel": float(st.sn_per_pixel),
        "C": float(st.concentration),
        "A": float(st.asymmetry),
        "S": float(st.smoothness),
        "sersic_amplitude": float(st.sersic_amplitude),
        "sersic_rhalf": float(st.sersic_rhalf),
        "sersic_n": float(st.sersic_n),
        "sersic_xc": float(st.sersic_xc),
        "sersic_yc": float(st.sersic_yc),
        "sersic_ellip": float(st.sersic_ellip),
        "sersic_theta": float(st.sersic_theta),
        "sersic_chi2_dof": float(st.sersic_chi2_dof),
        "sky_mean": float(st.sky_mean),
        "sky_median": float(st.sky_median),
        "sky_sigma": float(st.sky_sigma),
        "flag": st.flag,
        "flag_sersic": st.flag_sersic,
        "_name": frame.name,
        "_mask_size": frame.mask.sum(),
        "_target_size": frame.target.sum(),
        "_subtracted": not np.array_equal(frame.data, frame.data_sub),
        "_psf_used": frame.psf is not None,
        "_bg_mean": frame.bg_mean,
        "_bg_median": frame.bg_med,
        "_bg_std": frame.bg_std,
    }
    return data
