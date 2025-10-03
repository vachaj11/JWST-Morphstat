Input dictionary example structure
----------------------------------

The scripts run based on two inputs: 1. general configuration specified in `config.py` and 2. information about the sources and their imaging provided in a form of a `.json` file.

Example of the latter can be seen bellow for a single source `U4_17483` with imaging in filters `F090W` and `F444W`. (Comments on the structure are provided as entries `"##"` etc.)
    
    {
        "galaxies": [
            {
    	        "######": "Unique name of the source",
                "name": "U4_17483",
                
                "######": "Names of filters in which imaging of the source is available.",
                "filters": [
                    "F090W",
                    "F444W"
                ],
                
                "######": "Paths to fits files corresponding to imaging in available filters.",
                "files": [
                    "high_z_high_m_low_sfr_small/U4_17483_JWST_NIRCAM_F090W_cutout_6.0_arcsec.GAB.fits",
                    "high_z_high_m_low_sfr_small/U4_17483_JWST_NIRCAM_F444W_cutout_6.0_arcsec.GAB.fits"
                ],
                
                "######": "Various general information about the source.",
                "info": {
                    "####": "Best redshift estimate",
                    "ZBEST": 2.327,
                    
                    "####": "(Optional) Coordinates of the source, used only for identification of duplicates.",
                    "RA": 34.23394999999999,
                    "DEC": -5.217676944444444,
                    
                	"####": "(Optional) Various parameters used only for a few very specific plots",
                  	"##": "Sample binning in redshift/mass/sfr/size",
                    "z bin": "high_z",
                    "M bin": "high_m",
                    "SFR bin": "low_sfr",
                    "size bin": "small",
                    
                    "##": "Stellar mass",
                    "LMSTAR": 10.93000030517578,
                    
                    "##": "Star formation rate",
                    "DMS": -0.2079,
                    
                    "##": "Parameters of Sersic profile fit",
                    "H_RE": 0.2413,
                    "H_NSERSIC": 2.7699999809265137,
                    "H_Q": 0.5498,
                    "H_PA": 21.12,
                    "VDW12_H_RE": -99.0,
                    "VDW12_H_N": -99.0,
                    "VDW12_H_Q": -99.0,
                    "VDW12_H_PA": -999.0
                }
            }
        ]
    }

And bellow, example of source `U4_36568` with imaging in `F277W` shows the most minimal functional input that can be provided.  

    {
        "galaxies": [
            {
                "name": "U4_36568",
                "filters": [
                    "F277W"
                ],
                "files": [
                    "high_z_high_m_high_sfr_small/U4_36568_JWST_NIRCAM_F277W_cutout_6.0_arcsec.GAB.fits.pdf"
                ],
                "info": {
                    "ZBEST": 2.177
                }
            }
        ]
    }

Additional entries can be arbitrarily appended to any part of the `.json` file and should not affect any operational aspect of the scripts.

Output dictionary structure description
---------------------------------------

Each element in the list under the key `"galaxies"` represents one galaxy and includes:
- `"name"` - name of galaxies
- `"filters"` - list of filters for which statmorph was run
- `"info"` - galaxy info from the original dictionary, with added entries:
  - `"_flag_target"` - Flag noting whether targets identified in images in each of the filters are overlapping. (0 - good, 1-3 - increasingly bad)
  - `"_flag_rfw"` - Flag capturing how far the wavelength of the filter selected for the galaxy is from the chosen "optimal" rest frame wavelength for the sample. (0 - < 20 % difference, 1-3 - additional 20 %)
  - `"_flag_rfw_M"` - Flag indicating whether the selected filter for the galaxy has medium width rather than being wide. (0 - wide, 1 - medium)
  - `"_e"` - Recalculated (1-e) value of HST Sersic fit ellipticity to match the statmorph value
  - `"_t"` - Recalculated (±90 degrees to radians) value of HST Sersic fit position to match the statmorph value
- `"frames"` - list of dictionaries, one for each filter (named in `"filters"); each dictionary includes calculated statmorph parameters, and furthermore added entries:
  - `"_name"` - Name of the filter.
  - `"_wavelength"` - Wavelength of the filter.
  - `"_mask_size"` - Size of the mask applied to the image in no. of pixels.
  - `"_target_size"` - Size of the target segmentation map used for the image in no. of pixels.
  - `"_subtracted"` - Whether background subtraction has been applied (by the script) to the image.
  - `"_psf_used"` - Whether a psf for the image was found and passed to statmorph.
  - `"_adjustment"` - Notes whether the `"spatial resolution"` adjustment has been applied to the image. Is None if not, 0.0 if yes and using photutils kernel-matching, and some non-0 float value if yes and using Gaussian-width matching 
  - `"_bg_mean"` - Mean value of the image background (area outside of mask and target)
  - `"_bg_median"` - Median value of the image background (area outside of mask and target)
  - `"_bg_std"` - Standard deviation of the image background (area outside of mask and target)
  - `"_flag_seg"` - Similar to `"_flag_target"` above, but here for each filter identifies whether the target found in the filter has sufficient overlap in area with targets across all the remaining filters. (0 - if yes, 1 - if no, and hence the target should be treated with suspicion)
  - `"_flag_corr"` - By looking at how many pixels of the image have the exact value of exact 0.0 (i.e. oversaturation or other error) denotes how likely is for the image to be corrupted. (0 - < 10 % of the image, 1 - < 50 %, 2 - < 80 %, 3 - > 80 %).
  - `"_MID_align"` - (Unimportant. Used at some point when transcribing some values between dictionaries) 
  - `"_radn"` - Recalculated (pixels to arcsec) value of statmorph Sersic fit radius to match the HST value 
  - `"flag"` - Furthermore the value for this parameter has been reevaluated and modified from the value provided by statmorph based on visual inspection
  - `"_flag_sm"` - The original flag provided by statmorph. Moved from `"flag"` and taking values of 0, 1 and 2, denoting increasing unreliability of results.

Filtering process
-----------------

The results are filtered based on various combination of the flags described above. With the galaxies loaded as a list, the filtering can be done with the `resu.frames_pruning(galaxies, strength="normal")` function. Various options are:
- `"normal"` - this is used by default and leaves ~85 % galaxies. The filtering is:
  - `"flag"` <= 1
  - `"flag_sersic"` <= 2
  - `"_flag_seg"` = 0
  - `"_flag_corr"` <= 2
  - `"_psf_used"` = True
- `"strict"` - more strict/careful filtering which still leave most (~76 %) galaxies. The filter is:
  - `"flag"` = 0
  - `"flag_sersic"` <= 1
  - `"_flag_seg"` = 0
  - `"_flag_corr"` <= 1
  - `"_psf_used"` = True
- `"light"` - very loose filtering. Leaves most (~88 %) of the galaxies. The filter is:
  - `"flag"` <= 2
  - `"_flag_corr"` <= 2
  - `"_psf_used"` = True
- `"sersic"` - used to filter out unreliable results in the sersic fitting specifically (additional removal of ~3 % on top of `"normal"`). The filter is:
  - `"flag"` <= 2
  - `"flag_sersic"` <= 1
  - `"_flag_corr"` <= 2
  - `"_psf_used"` = True
    
Furthermore for each galaxy it is also required that `"_flag_rfw"` = 0.
