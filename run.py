"""Various methods with logic for fetching and building up data for/from the
statmorph computation.

Module holding the most central logic of the program including methods for
fetchin and saving data, getting internal representation of galaxies and
frames, calling statmorph calculation, and translating/reformating the results
of the calculation back into easily saved and accesible format.
"""

import json
import time
import warnings

import astropy
import matplotlib.pyplot as plt
import numpy as np

import stmo


def galaxies_data(
    gals, path_out=None, return_object=False, picture_path=None, psf_res=None
):
    """Main function that runs statmorph computation for an inputted list of
    galaxies, tracks the progress, saves the results, etc.

    For each of the galaxies in the inputted list, first creates its internal
    representation in terms of the :obj:`stmo.galaxy` and :obj:`stmo.frames`
    classes (decreasing the frames' resolution along the way if required)
    and runs the statmorph computation from the internal representation.
    Then based on the values of the optional arguments, can change the output
    format of the results, save them to specified path or save the visual
    outputs of statmorph. Along the calculation prints out its progress.

    Args:
        gals (list): List of galaxies represented as dictionaries with entries
            specifying their name, info, included frames, etc.
        path_out (str): Path where the results of the calculation are to be
            saved in an ``.json`` file. If `None`, the results are not saved.
        return_object (bool): If `True`, internal representation of galaxies
            is returned rather than results formatted in a dictionary for
            each galaxy. This can cause the program to run out of memory and
            be killed for large input lists.
        picture_path (str): Path to location where visual outputs of statmorph
            are to be saved. Of the format "folder_name/name.png", where
            "name" will be replaced and ".png" specifies file format. If
            `None`, then the visualisations are not generated and saved.
        psf_res (float): A target resolution in terms of stddev of psf in
            lightyears to which all frames should be adjusted before
            statmorph calculation. If `None`, no adjustment is made.

    Returns:
        list: List of either :obj:`stmo.galaxy` or dictionaries based on the
            :arg:`return_object` option, each representing one galaxy with the
            calculation results.
    """
    t0 = time.time()
    galaxies = []
    objects = []
    for i in range(len(gals)):
        gdata = calculate_stmo(gals[i], psf_res)
        galaxies.append(get_galaxy_data(gdata))
        if return_object:
            objects.append(gdata)
        print(
            "Finished galaxy {} in {:.2f} s ({} out of {})".format(
                galaxies[-1]["name"], time.time() - t0, i + 1, len(gals)
            )
        )
        if i % 10 == 0 and path_out is not None:
            save_as_json({"galaxies": galaxies}, path_out)
        if picture_path is not None:
            for f in gdata.frames:
                f.show_stmo(
                    save_path=picture_path.replace("name", (gdata.name + "_" + f.name))
                )
    if path_out is not None:
        save_as_json({"galaxies": galaxies}, path_out)
    if return_object:
        return objects
    else:
        return galaxies


def calculate_stmo(galaxy, psf_res=None):
    """Runs statmorph and returns internal representation of the given
    galaxy.

    For inputted galaxy in form of a dictionary, gets its information and
    fits files corresponding to its various frames and then uses these to
    get the galaxies' internal representation in terms of :obj:`stmo.galaxy`
    object and run statmorph calculation for each of the frames.

    Args:
        galaxy (dict): A dictionary representing the galaxy with information
            about it as well as frames/filters in which it is imaged.
        psf_res (float): Parameter noting the final physical size resolution
            to which the frames are to be adjusted before the statmorph
            calculation. Passed into/at creation of :obj:`stmo.galaxy`.

    Returns:
        :obj:`stmo.galaxy`: Internal representation of the galaxy with the
            statmorph calculation done.
    """
    info = galaxy["info"]
    name = galaxy["name"]
    filters, fitss = get_fitss(galaxy)
    gdata = stmo.galaxy(name, info, filters, fitss, psf_res=psf_res)
    return gdata


def adhoc_path(path):
    """Modifies the path to fits files ad hoc according to current setup.

    Should be moved to some config file later.
    """
    p = "../MP_cutouts/out/galfit_results/" + path[:-4]
    p = p.replace("big/", "big/results_")
    p = p.replace("small/", "small/results_")
    return p


def get_fitss(galaxy):
    """Gets fits files of galaxy in different wavelengths.

    For the passed galaxy represented by a dictionary, gets its filter names
    and path to fits files, from which it fetches the files and loads them
    with :obj:`astropy.io.fits` module. Then returns both.

    Args:
        galaxy (dict): A dictionary representing the galaxy with information
            about it as well as frames/filters in which it is imaged.

    Returns:
        tuple: A tuple consisting of:

            * *list* - List of `str` names of the filters.
            * *list* - List of :obj:`astropy.io.fits.HDUList` representation
              of the fits files.
    """
    filters = []
    fitss = []
    for l in range(len(galaxy["filters"])):
        path = adhoc_path(galaxy["files"][l])
        try:
            fitss.append(astropy.io.fits.open(path))
            filters.append(galaxy["filters"][l])
        except:
            warnings.warn(f"Couldn't locate {path}.")
    return filters, fitss


def save_as_json(thing, path):
    """Saves provided object as `json` file at specified path.

    Args:
        thing (dict, list, etc): Thing to be saved onto the path, should
            be serialisable in the ``.json`` file-structure, otherwise fails.
        path (str): Path where the object is to be saved. Should include the
            ".json" ending.
    """
    fil = open(path, "w")
    json.dump(thing, fil, indent=4)
    fil.close()


def fetch_json(path):
    """Fetches data of `json` object at specified path.

    Args:
        path (str): Full path at which the ``.json`` file is to be found.

    Returns:
        dict, list, etc.: Contents of the `json` file found at the path
            converted into equivalent python objects.
    """
    fil = open(path, "r")
    data = json.load(fil)
    fil.close()
    return data


def get_galaxies_data(galaxies):
    """Convert list of internal representations of galaxies into the
    dictionary format.

    Args:
        galaxies (list): List of internal representations, each
            :obj:`stmo.galaxy`, of galaxies.

    Returns:
        list: List of dictionary representation of galaxies with their names,
            information, results of statmorph calculation and other internal
            parameters.
    """
    gals = []
    for galaxy in galaxies:
        gald = get_galaxy_data(galaxy)
        gals.append(gald)
    return gals


def get_galaxy_data(galaxy):
    """Extracts data from the internal representation of galaxy and formats
    them into the output dictionary.

    For the given galaxy creates a dictionary into which it saves various
    information which were attributes of the internal representation,
    including list of frames and data calculated for each of them.

    Args:
        galaxy (stmo.galaxy): Internal representation of a galaxy to be
            transcribed to the output dictionary.

    Returns:
        dict: A dictionary holding information relating to the galaxy and
            results calculated for each of its frames.
    """
    gald = dict()
    gald["name"] = galaxy.name
    gald["filters"] = galaxy.filters
    gald["info"] = galaxy.info
    gald["info"]["_flag_target"] = galaxy.target_flag
    frames = []
    for frame in galaxy.frames:
        data = get_frame_data(frame)
        frames.append(data)
    gald["frames"] = frames
    return gald


def get_frame_data(frame):
    """Extracts data from an internal representation of a frame and formats
    them into the output dictionary.

    For the passed frame creates a dictionary into which it saves many
    attributes of the frame's internal representation, including the results
    of statmorph calculation, information about the segmentation map, masks,
    flags, background statistics, etc.

    Args:
        frame (stmo.frame): Internal representation of a frame to be
            translated into an output dictionary.

    Returns:
        dict: A dictionary holding information relating to the frame, its
            characteristics as well as many of the calculation results carried
            out for it.
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
        "_wavelength": float(frame.name[1:-1]),
        "_mask_size": int(frame.mask.sum()),
        "_target_size": int(frame.target.sum()),
        "_subtracted": not np.array_equal(frame.data, frame.data_sub),
        "_psf_used": frame.psf is not None,
        "_adjustment": (
            float(frame.adjustment)
            if frame.adjustment is not None
            else frame.adjustment
        ),
        "_bg_mean": float(frame.bg_mean),
        "_bg_median": float(frame.bg_med),
        "_bg_std": float(frame.bg_std),
        "_flag_seg": int(frame.flag_seg),
        "_flag_corr": int(frame.flag_corr),
    }
    return data
