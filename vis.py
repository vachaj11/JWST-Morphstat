"""Various methods for visualisation of the results.

Most accepts a list of result-dictionaries of galaxies as an input and plot
requested values in histogram/against each other/their differences/etc.
All of the methods utilise matplotlib.pyplot and do not call plt.show(), so
the functions can be called multiple times to overlay different data.
Each function also runs very course filter on the data to discard outliers and
most implement a method where clicking on the plot gives user a name of the
closest galaxy.
"""

import warnings

import matplotlib.image as mpi
import matplotlib.pyplot as plt
import numpy as np

import resu


def rem_bad_outliers(data, sig=5):
    """Removes data from the set that are more than n-sigma out.

    Finds outliers in data[0] values and removes them (as well as
    corresponding entries in data[i]). Also prints which galaxies are
    being removed if one of data[i] corresponds to their names.
    """
    mean = np.mean(data[0])
    std = np.std(data[0])
    ind = []
    for i in range(len(data[0])):
        if data[0][i] - mean < -sig * std or data[0][i] - mean > sig * std:
            ind.append(i)
    ind.reverse()
    if ind:
        warnings.warn(
            f"found and removed the following {len(ind)} very bad (>{sig} sigma) outliers in the data:"
        )
    for dat in data:
        for i in ind:
            if type(dat[i]) == str:
                print(dat[i], end=", ")
            dat.pop(i)
    if ind:
        print()
    return data


def plot_value(galaxies, valuex, valuey, filt="avg"):
    """Plots requested values (for a given filter or averaged) for a provided
    set of galaxies.

    For each galaxy gets its coordinates of (valuex, valuey) and puts it on
    the plot. Here `valuex` corresponds to entry for the galaxy, whereas
    `valuey` for a single or multiple of its frames (depending on `filt`).
    """
    valsy = []
    valsx = []
    valsg = []
    for g in galaxies:
        valy = resu.get_filter_or_avg(g, valuey, filt)
        if valy:
            valsy.append(valy)
            valsx.append(g["info"][valuex])
            valsg.append(g["name"])
    valsy, valsx, valsg = rem_bad_outliers([valsy, valsx, valsg])
    if filt:
        plt.plot(
            valsx,
            valsy,
            linestyle="",
            marker="+",
            label=f"filter {filt} ({len(valsx)})",
        )
    else:
        plt.plot(valsx, valsy, linestyle="", marker="+", label=f"({len(valsx)})")
    plt.xlabel = valuex
    plt.ylabel = valuey
    return zip(valsx, valsy, valsg)


def plot_correlation(galaxies, valuex, valuey, filt="avg"):
    """Plots requested values (for a given filter or averaged) against each
    other for a provided set of galaxies.

    For each galaxy gets its coordinates of (valuex, valuey) and puts it on
    the plot. Here both `valuex` and `valuey` correspond to a single or
    multiple of galaxies' frames (depending on `filt`).
    """
    valsy = []
    valsx = []
    valsg = []
    for g in galaxies:
        valy = resu.get_filter_or_avg(g, valuey, filt)
        valx = resu.get_filter_or_avg(g, valuex, filt)
        if valy and valx:
            valsy.append(valy)
            valsx.append(valx)
            valsg.append(g["name"])
    valsy, valsx, valsg = rem_bad_outliers([valsy, valsx, valsg])
    valsx, valsy, valsg = rem_bad_outliers([valsx, valsy, valsg])
    if filt:
        plt.plot(
            valsx,
            valsy,
            linestyle="",
            marker="+",
            label=f"filter {filt} ({len(valsx)})",
        )
    else:
        plt.plot(valsx, valsy, linestyle="", marker="+", label=f"({len(valsx)})")
    plt.xlabel(valuex)
    plt.ylabel(valuey)
    return zip(valsx, valsy, valsg)


def plot_value_difference(gals1, gals2, valuex, valuey, filt="avg"):
    """Plots difference in requested value (for a given filter or averaged)
    for provided two sets of galaxies (and for different filters if filt is
    provided as a list).

    Similar to :obj:`plot_value` but here to plot the difference between
    two sets of result-galaxies.
    """
    valsd = []
    valsx = []
    valsg = []
    galf1 = resu.get_subset(gals1, gals2)
    galf2 = resu.get_subset(gals2, galf1)
    for i in range(len(galf1)):
        if type(filt) == list and len(filt) == 2:
            val1 = resu.get_filter_or_avg(galf1[i], valuey, filt[0])
            val2 = resu.get_filter_or_avg(galf2[i], valuey, filt[1])
        else:
            val1 = resu.get_filter_or_avg(galf1[i], valuey, filt)
            val2 = resu.get_filter_or_avg(galf2[i], valuey, filt)
        if val1 and val2:
            valsd.append(val2 - val1)
            valsx.append(galf1[i]["info"][valuex])
            valsg.append(galf1[i]["name"])
    if type(filt) == str:
        plt.plot(
            valsx,
            valsd,
            linestyle="",
            marker="+",
            label=f"filter {filt} ({len(valsx)})",
        )
    else:
        plt.plot(valsx, valsd, linestyle="", marker="+", label=f"({len(valsx)})")
    plt.xlabel(valuex)
    plt.ylabel(valuey)
    vals = zip(valsx, valsd, valsg)
    fig = plt.gcf()
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda x: resu.print_closest([x.xdata, x.ydata], vals, fig=fig),
    )


def plot_histogram(galaxies, value, nbins=None, filt="avg"):
    """Plots a histogram of requested value (for a given filter or averaged)
    for a given set of galaxies.

    For each galaxy gets the value corresponding to `value` for either a
    single frame/filter or averaged, depending on `filt`. From all values then
    creates a histogram using basic numpy methods.
    """
    vals = []
    for g in galaxies:
        val = resu.get_filter_or_avg(g, value, filt)
        if val:
            vals.append(val)
    vals = rem_bad_outliers([vals])[0]
    if nbins is not None:
        count, bins = np.histogram(vals, nbins)
    else:
        count, bins = np.histogram(vals)
    if filt:
        plt.stairs(count / len(vals), bins, label=f"filter {filt} ({len(vals)})")
    else:
        plt.stairs(count / len(vals), bins, label=f"({len(vals)})")


def plot_hist_comp(gals_list, value, nbins=None, filt="avg", pdf=False):
    """Plots a joint histogram of requested value (for a given filter or
    averaged) for given n sets of galaxies.

    Similar to :obj:`plot_histogram`, but works for multiple galaxies.
    Also has an option of `pdf` which allows to normalisation such that
    the full area of the histogram is 1 (useful for comparing for sets
    of vastly different sizes).
    """
    for galaxies in gals_list:
        vals = []
        valsg = []
        for g in galaxies:
            val = resu.get_filter_or_avg(g, value, filt)
            if val:
                vals.append(val)
                valsg.append(g["name"])
        vals = rem_bad_outliers([vals, valsg])[0]
        if nbins is not None:
            count, bins = np.histogram(vals, nbins)
        else:
            count, bins = np.histogram(vals)
        if pdf:
            count = count / (len(vals) * (bins[1] - bins[0]))
        if filt:
            plt.stairs(count, bins, label=f"filter {filt} ({len(vals)})")
        else:
            plt.stairs(count, bins, label=f"({len(vals)})")

    if not pdf:
        plt.title(f"Histogram comparison of {value}")
    else:
        plt.title(f"Approximate distribution comparison of {value}")
    plt.legend()


def plot_value_filters(galaxies, valuex, valuey, filt=2):
    """Plots requested values for a provided set of galaxies across multiple
    filters.

    Like :obj:`plot_value` but allows for plotting with different filters
    specified by the `filt` parameter.
    Also implements the click-to-get-galaxy-name functionality.
    """
    vals = []
    if type(filt) == int:
        filts = resu.get_most_filters(galaxies, filt)
    elif type(filt) in (list, set, tuple):
        filts = list(filt)
    elif type(filt) == str and filt == "avg":
        filts = ["avg"]
    else:
        warnings.warn(f"Unrecognised type of filt: {type(filt)}.")
    for filt in filts:
        vals.extend(plot_value(galaxies, valuex, valuey, filt=filt))
    fig = plt.gcf()
    fig.canvas.mpl_connect(
        "button_press_event", lambda x: resu.print_closest([x.xdata, x.ydata], vals)
    )


def plot_correlation_filters(galaxies, valuex, valuey, filt=2):
    """Plots requested values agains each other for a provided set of galaxies
    across multiple filters.

    Like :obj:`plot_correlation` but allows for plotting with different
    filters specified by the `filt` parameter.
    Also implements the click-to-get-galaxy-name functionality.
    """
    vals = []
    if type(filt) == int:
        filts = resu.get_most_filters(galaxies, filt)
    elif type(filt) in (list, set, tuple):
        filts = list(filt)
    elif type(filt) == str and filt == "avg":
        filts = ["avg"]
    else:
        warnings.warn(f"Unrecognised type of filt: {type(filt)}.")
    for filt in filts:
        vals.extend(plot_correlation(galaxies, valuex, valuey, filt=filt))
    fig = plt.gcf()
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda x: resu.print_closest([x.xdata, x.ydata], vals, fig=fig),
    )


def plot_histogram_filters(galaxies, value, filt=2, pdf=False):
    """Plots histograms of requested value for a given set of galaxies across
    multiple filters.

    Like :obj:`plot_histogram` but allows for plotting with different
    filters specified by the `filt` parameter.
    """
    if type(filt) == int:
        filts = resu.get_most_filters(galaxies, filt)
    elif type(filt) in (list, set, tuple):
        filts = list(filt)
    elif type(filt) == str and filt == "avg":
        filts = ["avg"]
    else:
        warnings.warn(f"Unrecognised type of filt: {type(filt)}.")
    for filt in filts:
        plot_histogram(galaxies, value, filt=filt, pdf=pdf)
    plt.title(f"Histogram of {value}")


def plot_pic_value(
    galaxy, values=["C", "A", "S"], path="../galfit_results/colour_images/"
):
    """For a given galaxy show its colour picture jointly with graphs of
    requested values as a function of wavelength.

    The path here is to the folder with stored pictures named as
    "galaxy_name.png"
    """
    fig = plt.figure()
    gs = fig.add_gridspec(
        len(values) + 1, hspace=0, height_ratios=[3] + [1 for i in values]
    )
    axs = gs.subplots()
    fig.suptitle(galaxy["name"])
    pic_path = path + galaxy["name"] + ".png"
    img = mpi.imread(pic_path)
    axs[0].imshow(img)
    axs[0].set_axis_off()
    wavel = []
    data = [[] for i in values]
    for i in range(len(galaxy["filters"])):
        wavel.append(int(galaxy["filters"][i][1:-1]))
        for l in range(len(values)):
            data[l].append(galaxy["frames"][i][values[l]])
    for i in range(len(data)):
        axs[i + 1].plot(wavel, data[i])
        axs[i + 1].set(xlabel="wavelength", ylabel=values[i])
        axs[i + 1].label_outer()
        axs[i + 1].sharex(axs[1])


def plot_sersic(galsout, galsin, valueout, filt="avg"):
    """Plots a sersic fitting parameter (for a given filter or
    averaged) for provided input and output set of galaxies.

    Serves the sole purpose of comparing statmorph sersic result to galfit's.
    The list of result-dictionaries galsout should correspond (be created
    from) to galsin or be its subset.
    Also implements the click-to-get-galaxy-name functionality.
    """
    valsout = []
    valsin = []
    valsg = []
    transcript = {
        "sersic_amplitude": "m",
        "sersic_rhalf": "Re",
        "sersic_n": "n",
        "sersic_xc": "x0",
        "sersic_yc": "y0",
        "sersic_ellip": "q",
        "sersic_theta": "PA",
    }
    if valueout in transcript.keys():
        valuein = transcript[valueout]
    else:
        valuein = ""
    galfout = resu.get_subset(galsout, galsin)
    galfin = resu.get_subset(galsin, galfout)
    for i in range(len(galfout)):
        val1 = resu.get_filter_or_avg(galfout[i], valueout, filt)
        val2 = resu.get_filter_or_avg(galfin[i], valuein, filt)
        if val1 and val2:
            valsout.append(val1)
            valsin.append(val2)
            valsg.append(galfout[i]["name"])
    valsout, valsin, valsg = rem_bad_outliers([valsout, valsin, valsg])
    valsin, valsout, valsg = rem_bad_outliers([valsin, valsout, valsg])
    plt.plot(
        valsin,
        valsout,
        linestyle="",
        marker="+",
        label=f"filter {filt} ({len(valsin)})",
    )
    plt.xlabel(valuein)
    plt.ylabel(valueout)
    vals = list(zip(valsin, valsout, valsg))
    fig = plt.gcf()
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda x: resu.print_closest([x.xdata, x.ydata], vals, fig=fig),
    )
