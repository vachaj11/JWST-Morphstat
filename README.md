# *JWST-Morphstat* - Statmorph calculations for the JWST Morphology Project

Set of scripts for running [`statmorph`](https://statmorph.readthedocs.io/en/latest/) calculation and visualising results created as part of the JWST morphology project at the Max Planck Institute for Extraterrestrial Physics. Mostly written over summer 2024 and during subsequent months in preparation for publication as part of [*Espejo Salcedo et al. 2025*](https://ui.adsabs.harvard.edu/abs/2025A%26A...700A..42E/abstract).

Most of the core code is documented via docstrings attached to module-level functions, classes and attributes - although later code handling specifics of results visualisations can in part lack this. Despite this the code is currently for the most part *not* in a state where it will likely work when run without further environment-specific adjustions, for start because it makes some arbitrary assumptions about inputted data.

## Instalation

All prerequisites can be installed with

    $ python3 -m pip install astropy numpy photutils matplotlib statmorph scipy opencv-python

After which the code can be cloned to appropriate directory, and run as described bellow, assuming all paths appropriately modified.

## Usage

The scripts are created to be used in the interactive mode of the python interpreter by calling specific methods included in them.
To this purpose documentation is also included in docstring of each method/object from which it can be accessed the easiest with
`help(object)`.

The code is divided into 6 modules:
- Background logic
  - `psfm.py` - used for psf physical size-resolution matching
  - `seg.py` - used for creation of segmentation map of target and masks
  - `stmo.py` - holds classes internally representing galaxy and its frames, and runs statmorph calculation as well as calculation of its prerequisites
- `run.py` - Main module for calling the statmorph calculation. The main method here is `run.galaxies_data(...)`
- `resu.py` - Module holding methods for preparing the data for statmorph calculation as well as filtering and manipulating the resulting data
- `vis.py` - Set of methods for visualisation of the results utilising `matplotlib`
- Ad hoc modules:
  - `line_separation.py` - Methods to find line separating positions of two sets of galaxies in some parameter space

For more information on each module access its docstirng with:

    $ cd "path_to_modules"
    $ python3

    ...
    
    >>> import run
    >>> help(run)


## Standard run

Following would be a standard usage for statmorph calculation, filtering of the results, etc. :

    $ python3

    ...

    >>> import run, resu, vis
    >>> # get input dictionary data
    >>> gal_input = run.fetch_json("dictionary_full.json")["galaxies"]
    >>> # match for rfw
    >>> gal_rfw = resu.get_optim_rfw(gal_input)[1]
    >>> # find lowest psf resolution 
    >>> max_width = resu.get_maximal_psf_width(gal_rfw)
    >>> # run statmorph calculation with resolution decreased to match max_width
    >>> gal_out = run.galaxies_data(gal_rfw, psf_res = max_width)

    ...

    >>> # possible step including reflagging with resu.manual_reev_c(...)
    >>> # filtering the data according to flags
    >>> gal_filtered = resu.galaxy_pruning(gal_out, strength = "normal")
    >>> # saving the calculated data
    >>> run.save_as_json({"galaxies": gal_filtered}, "out_full.json")
    >>>
    >>> # e.g. visualisation of assymmetry in different redshift bins
    >>> # separating the list of galaxies by redshift bin
    >>> sep = resu.get_separate_in_value(gal_filtered, "z bin")
    >>> # plotting and showing the assymetry histogram
    >>> vis.plot_hist_comp([sep["high_z"],sep["low_z"]],"A", pdf=True)
    >>> import matplotlib.pyplot as plt
    >>> plt.show(block=False)
