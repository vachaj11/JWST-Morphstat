"""Holds various values and properties making up configuration of the script.
"""

# MAIN variables
# ==============

# Main directory in which cutout files are to be found
cutout_path = "../MP_cutouts/out/galfit_results/"


def f_cutout_path(p):
    """Function to be applied to full path to cutouts."""
    if p[-4:] == ".pdf":
        p = p[:-4]
    p = p.replace("big/", "big/results_")
    p = p.replace("small/", "small/results_")
    return p


# PHYSICS variables
# =================

# Cosmological parameters
cosm_H0 = 67.49
cosm_Om0 = 0.315
cosm_Od0 = 0.6847

# PSF variables
# =============

# Directroy in which PSFs files are to be found
psf_path = "../psf/"

# Pixel size in radians
pixel_size = 0.025 / 3600 / 180 * 3.141592653589793
# Pre-determined full-width-at-half-maximum values of PSFs of the data
psf_fwhm_sizes = {
    "F090W": (1.1995772884306106, 1.149118956666665),
    "F115W": (1.470719758687787, 1.5051056155391633),
    "F150W": (1.9493943876899091, 1.8991272065113258),
    "F182M": (2.324399468263498, 2.375521701356913),
    "F200W": (2.4781960999114885, 2.5286361972775047),
    "F210M": (2.645962560917274, 2.6954729402728326),
    "F277W": (3.430500911832727, 3.521493646912973),
    "F300M": (3.7620623355473835, 3.8648936085711054),
    "F335M": (4.22538001762926, 4.340623710243656),
    "F356W": (4.543749121910588, 4.424874502103289),
    "F410M": (5.129923450562115, 5.261257319645101),
    "F430M": (5.3889589541225735, 5.52392352664033),
    "F444W": (5.447786223407964, 5.582801195553154),
    "F460M": (5.824326385917065, 5.965375660435022),
    "F480M": (6.054312102519142, 6.199064508582882),
}
# PLOTTING variables
# ==================

# Names of galaxies to remove before calculation
remove_names = [
    "GS4_18890",
    "COS4_17600",
    "GS4_19797",
    "COS4_05758",
    "U4_28125",
    "G4_06569",
    "GS4_30028",
    "GS4_29773",
]

# Various paths specifying input/output dictionaries
dict_paths = {
    "out": "dict_out/out_full_matched_5_m.json",
    "bulge": "dict_in/n4_Bulge.json",
    "merger": "dict_in/n4_Interacting-Merger.json",
    "clumpy": "dict_in/n4_Clumpy.json",
    "full": "dict_in/dictionary_full.json",
    "ser_444": "dict_out/out_full_444.json",
    "ser_150": "dict_out/out_full_150.json",
    "ser_150_hst": "dict_out/out_full_150_hst.json",
    "ser_hst_160": "dict_out/out_hst_160_full.json",
    "ser_150_scaled": "dict_out/out_full_150_scaled.json",
    "ser_150_scaled_3": "dict_out/out_full_150_scaled_3.json",
    "ser_150_scaled_30": "dict_out/out_full_150_scaled_30.json",
    "ser_full": "dict_out/out_full_sersic.json",
}

# AD-HOC variables
# ================

# Some default values of ad hoc paths
def_image_path = "../statmorph_images_filtered/name.png"
def_dict_in = "dict_in/dictionary_full.json"
