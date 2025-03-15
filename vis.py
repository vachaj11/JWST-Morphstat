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
import matplotlib.collections as mpc
import numpy as np
import scipy.stats as stats
import seaborn as sns

import resu
import run

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern",
        "font.size": 15,
    }
)


def rem_bad_outliers(data, sig=5):
    """Removes data from the set that are more than n-sigma out.

    Finds outliers in data[0] values and removes them (as well as
    corresponding entries in data[i]). Also prints which galaxies are
    being removed if one of data[i] corresponds to their names.
    """
    datan = np.array(data[0])
    datan = datan[datan != np.array(None)]
    mean = np.mean(datan)
    std = np.std(datan)
    ind = []
    for i in range(len(data[0])):
        if (
            data[0][i] is None
            or data[0][i] - mean < -sig * std
            or data[0][i] - mean > sig * std
        ):
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
    return tuple(data)


def plot_value(galaxies, valuex, valuey, filt="avg", axis=None):
    """Plots requested values (for a given filter or averaged) for a provided
    set of galaxies.

    For each galaxy gets its coordinates of (valuex, valuey) and puts it on
    the plot. Here `valuex` corresponds to entry for the galaxy, whereas
    `valuey` for a single or multiple of its frames (depending on `filt`).
    """
    if axis is not None:
        ax = axis
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots()
    valsy = []
    valsx = []
    valsg = []
    for g in galaxies:
        valy = resu.get_filter_or_avg(g, valuey, filt)
        if valy is not None:
            valsy.append(valy)
            valsx.append(g["info"][valuex])
            valsg.append(g["name"])
    valsy, valsx, valsg = rem_bad_outliers([valsy, valsx, valsg])
    if filt:
        ax.plot(
            valsx,
            valsy,
            linestyle="",
            marker=".",
            label=f"filter {filt} ({len(valsx)})",
            alpha=0.3,
        )
    else:
        ax.plot(
            valsx, valsy, linestyle="", marker=".", alpha=0.3, label=f"({len(valsx)})"
        )
    ax.set_xlabel(valuex)
    ax.set_ylabel(valuey)
    return zip(valsx, valsy, valsg)

def plot_value_2s(galaxies1, galaxies2, valuex, valuey, axis=None):
    """!!!
    """
    if axis is not None:
        ax = axis
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots()
    gal1 = resu.get_subset(galaxies1, galaxies2)
    gal2 = resu.get_subset(galaxies2, gal1)
    valsy = []
    valsx = []
    valsg = []
    for g in gal1:
        valx = resu.get_filter_or_avg(g, valuex, "avg")
        gy = next((g2 for g2 in gal2 if g2["name"] == g["name"]), None)
        valy = resu.get_filter_or_avg(gy, valuey, "avg")
        if valx is not None and valy is not None:
            valsy.append(valy)
            valsx.append(valx)
            valsg.append(g["name"])
    valsy, valsx, valsg = rem_bad_outliers([valsy, valsx, valsg])
    ax.plot(
        valsx, valsy, linestyle="", marker=".", alpha=0.3, label=f"({len(valsx)})"
    )
    ax.set_xlabel(valuex)
    ax.set_ylabel(valuey)
    return zip(valsx, valsy, valsg)

def plot_correlation(
    galaxies, valuex, valuey, filt="avg", return_full=False, axis=None
):
    """Plots requested values (for a given filter or averaged) against each
    other for a provided set of galaxies.

    For each galaxy gets its coordinates of (valuex, valuey) and puts it on
    the plot. Here both `valuex` and `valuey` correspond to a single or
    multiple of galaxies' frames (depending on `filt`).
    """
    if axis is not None:
        ax = axis
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots()
    valsy = []
    valsx = []
    valsg = []
    valsf = []
    for g in galaxies:
        valy = resu.get_filter_or_avg(g, valuey, filt)
        valx = resu.get_filter_or_avg(g, valuex, filt)
        if valy is not None and valx is not None:
            valsy.append(valy)
            valsx.append(valx)
            valsg.append(g["name"])
            valsf.append(g)
    valsy, valsx, valsg, valsf = rem_bad_outliers([valsy, valsx, valsg, valsf])
    valsx, valsy, valsg, valsf = rem_bad_outliers([valsx, valsy, valsg, valsf])
    if filt:
        ax.plot(
            valsx,
            valsy,
            linestyle="",
            marker=".",
            label=f"filter {filt} ({len(valsx)})",
            alpha=0.3,
        )
    else:
        ax.plot(
            valsx, valsy, linestyle="", marker=".", label=f"({len(valsx)})", alpha=0.3
        )
    ax.set_xlabel(valuex)
    ax.set_ylabel(valuey)
    if not return_full:
        return zip(valsx, valsy, valsg)
    else:
        return valsx, valsy, valsf


def plot_value_difference(gals1, gals2, valuex, valuey, filt="avg", axis=None):
    """Plots difference in requested value (for a given filter or averaged)
    for provided two sets of galaxies (and for different filters if filt is
    provided as a list).

    Similar to :obj:`plot_value` but here to plot the difference between
    two sets of result-galaxies.
    """
    if axis is not None:
        ax = axis
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots()
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
        if val1 is not None and val2 is not None:
            valsd.append(val2 - val1)
            valsx.append(galf1[i]["info"][valuex])
            valsg.append(galf1[i]["name"])
    if type(filt) == str:
        ax.plot(
            valsx,
            valsd,
            linestyle="",
            marker=".",
            label=f"filter {filt} ({len(valsx)})",
            alpha=0.3,
        )
    else:
        ax.plot(
            valsx,
            valsd,
            linestyle="",
            marker=".",
            label=f"({len(valsx)})",
            alpha=0.3,
        )
    ax.set_xlabel(valuex)
    ax.set_ylabel(valuey)
    vals = zip(valsx, valsd, valsg)
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda x: resu.print_closest([x.xdata, x.ydata], vals, fig=fig),
    )


def plot_histogram(galaxies, value, nbins=None, pdf=True, filt="avg", axis=None):
    """Plots a histogram of requested value (for a given filter or averaged)
    for a given set of galaxies.

    For each galaxy gets the value corresponding to `value` for either a
    single frame/filter or averaged, depending on `filt`. From all values then
    creates a histogram using basic numpy methods.
    """
    if axis is not None:
        ax = axis
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots()
    vals = []
    for g in galaxies:
        val = resu.get_filter_or_avg(g, value, filt)
        if val is not None:
            vals.append(val)
    vals = rem_bad_outliers([vals])[0]
    if nbins is not None:
        count, bins = np.histogram(vals, nbins)
    else:
        count, bins = np.histogram(vals)
    if pdf:
        ax.stairs(count / len(vals), bins, label=f"filter {filt} ({len(vals)})")
    else:
        ax.stairs(count / len(vals), bins, label=f"({len(vals)})")


def plot_hist_comp(
    gals_list, value, bins=10, filt="avg", pdf=False, joint_bins=False, axis=None
):
    """Plots a joint histogram of requested value (for a given filter or
    averaged) for given n sets of galaxies.

    Similar to :obj:`plot_histogram`, but works for multiple galaxies.
    Also has an option of `pdf` which allows to normalisation such that
    the full area of the histogram is 1 (useful for comparing for sets
    of vastly different sizes).
    Also has an option allowing for the same bins to be used for all sets
    of galaxies, to ease comparison.
    """
    if axis is not None:
        ax = axis
    else:
        fig, ax = plt.subplots()
    if joint_bins:
        lis = [resu.get_filter_or_avg(g, value, filt) for gs in gals_list for g in gs]
        lisf = rem_bad_outliers([lis])[0]
        listn = np.array(lisf)
        valsn = listn[listn != np.array(None)]
        bins = np.histogram(valsn, bins)[1]
    for galaxies in gals_list:
        vals = []
        valsg = []
        for g in galaxies:
            val = resu.get_filter_or_avg(g, value, filt)
            if val is not None:
                vals.append(val)
                valsg.append(g["name"])
        vals = rem_bad_outliers([vals, valsg])[0]
        if filt:
            lab = f"filter {filt} ({len(vals)})"
        else:
            lab = f"({len(vals)})"
        ax.hist(vals, label=lab, density=pdf, bins=bins, histtype="step")

    if not pdf:
        ax.set_title(f"Histogram comparison of {value}")
    else:
        ax.set_title(f"Approximate distribution comparison of {value}")
    ax.legend()


def plot_smooth_comp(gals_list, value, filt="avg", pdf=False, nsig=25, axis=None):
    """Plots a smooth out distribution of requested value (for a given filter
    or averaged) for given n sets of galaxies.

    Similar to :obj:`plot_hist_comp`, but creates smooth curve by for each
    point summing neighbouring counts over a gaussian kernel.
    """
    if axis is not None:
        ax = axis
    else:
        fig, ax = plt.subplots()
    for galaxies in gals_list:
        vals = []
        valsg = []
        for g in galaxies:
            val = resu.get_filter_or_avg(g, value, filt)
            if val is not None:
                vals.append(val)
                valsg.append(g["name"])
        vals = rem_bad_outliers([vals, valsg])[0]
        if filt:
            lab = f"filter {filt} ({len(vals)})"
        else:
            lab = f"({len(vals)})"
        minx = min(vals)
        maxx = max(vals)
        sig = (maxx - minx) / nsig
        gaus = lambda x: 1 / (sig * np.sqrt(2 * np.pi)) * np.exp(-((x / sig) ** 2) / 2)
        x = np.linspace(minx, maxx, num=200)
        y = []
        if pdf:
            norm = len(vals)
        else:
            norm = nsig / (maxx - minx)
        # norm = int(pdf) * (len(vals) - 1) + 1
        for i in x:
            y.append(sum([gaus(i - c) for c in vals]) / norm)
        ax.plot(x, y, label=lab)

    if not pdf:
        ax.set_title(f"Histogram comparison of {value}")
    else:
        ax.set_title(f"Approximate distribution comparison of {value}")
    ax.legend()


def plot_value_filters(galaxies, valuex, valuey, filt="avg", axis=None):
    """Plots requested values for a provided set of galaxies across multiple
    filters.

    Like :obj:`plot_value` but allows for plotting with different filters
    specified by the `filt` parameter.
    Also implements the click-to-get-galaxy-name functionality.
    """
    if axis is not None:
        ax = axis
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots()
    vals = []
    if type(galaxies) is not tuple:
        galaxies = (galaxies,)
    if type(filt) == int:
        filts = resu.get_most_filters(sum(galaxies, []), filt)
    elif type(filt) in (list, set, tuple):
        filts = list(filt)
    elif type(filt) == str and filt == "avg":
        filts = ["avg"]
    else:
        warnings.warn(f"Unrecognised type of filt: {type(filt)}.")
    for filt in filts:
        for gs in galaxies:
            vals.extend(plot_value(gs, valuex, valuey, filt=filt, axis=ax))
    fig.canvas.mpl_connect(
        "button_press_event", lambda x: resu.print_closest([x.xdata, x.ydata], vals)
    )

def plot_values_2s(galaxies1, galaxies2, valuex, valuey, axis=None):
    """!!!
    """
    if axis is not None:
        ax = axis
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots()
    vals = []
    vals.extend(plot_value_2s(galaxies1, galaxies2, valuex, valuey, axis=ax))
    fig.canvas.mpl_connect(
        "button_press_event", lambda x: resu.print_closest([x.xdata, x.ydata], vals)
    )


def plot_correlation_filters(galaxies, valuex, valuey, filt="avg", axis=None):
    """Plots requested values agains each other for a provided set of galaxies
    across multiple filters.

    Like :obj:`plot_correlation` but allows for plotting with different
    filters specified by the `filt` parameter.
    Also implements the click-to-get-galaxy-name functionality.
    """
    if axis is not None:
        ax = axis
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots()
    vals = []
    if type(galaxies) is not tuple:
        galaxies = (galaxies,)
    if type(filt) == int:
        filts = resu.get_most_filters(sum(galaxies, []), filt)
    elif type(filt) in (list, set, tuple):
        filts = list(filt)
    elif type(filt) == str and filt == "avg":
        filts = ["avg"]
    else:
        warnings.warn(f"Unrecognised type of filt: {type(filt)}.")
    for filt in filts:
        for gs in galaxies:
            vals.extend(plot_correlation(gs, valuex, valuey, filt=filt, axis=ax))
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda x: resu.print_closest([x.xdata, x.ydata], vals, fig=fig),
    )


def plot_histogram_filters(galaxies, value, filt="avg", pdf=False, axis=None):
    """Plots histograms of requested value for a given set of galaxies across
    multiple filters.

    Like :obj:`plot_histogram` but allows for plotting with different
    filters specified by the `filt` parameter.
    """
    if axis is not None:
        ax = axis
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots()
    if type(galaxies) is not tuple:
        galaxies = (galaxies,)
    if type(filt) == int:
        filts = resu.get_most_filters(sum(galaxies, []), filt)
    elif type(filt) in (list, set, tuple):
        filts = list(filt)
    elif type(filt) == str and filt == "avg":
        filts = ["avg"]
    else:
        warnings.warn(f"Unrecognised type of filt: {type(filt)}.")
    for filt in filts:
        for gs in galaxies:
            plot_histogram(gs, value, filt=filt, pdf=pdf, axis=ax)
    ax.set_title(f"Histogram of {value}")


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


def plot_sersic(galsout, galsin, valueout, filt="avg", axis=None):
    """Plots a sersic fitting parameter (for a given filter or
    averaged) for provided input and output set of galaxies.

    Serves the sole purpose of comparing statmorph sersic result to galfit's.
    The list of result-dictionaries galsout should correspond (be created
    from) to galsin or be its subset.
    Also implements the click-to-get-galaxy-name functionality.
    """
    if axis is not None:
        ax = axis
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots()
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
        if val1 is not None and val2 is not None:
            valsout.append(val1)
            valsin.append(val2)
            valsg.append(galfout[i]["name"])
    valsout, valsin, valsg = rem_bad_outliers([valsout, valsin, valsg])
    valsin, valsout, valsg = rem_bad_outliers([valsin, valsout, valsg])
    ax.plot(
        valsin,
        valsout,
        linestyle="",
        marker=".",
        label=f"filter {filt} ({len(valsin)})",
        alpha=0.3,
    )
    ax.set_xlabel(valuein)
    ax.set_ylabel(valueout)
    vals = list(zip(valsin, valsout, valsg))
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda x: resu.print_closest([x.xdata, x.ydata], vals, fig=fig),
    )


def plot_statmorph_gini(x, y, c):
    """Extremely ad hoc function to plot one specific graph (Figure 5.) from
    the statmorph paper which relates Gini-M20 statistics with concentration.
    """
    histfull, xedges, yedges = np.histogram2d(
        x, y, bins=50, range=[[-3, 0], [0.3, 0.8]], weights=c
    )
    histcount, _, _ = np.histogram2d(x, y, bins=50, range=[[-3, 0], [0.3, 0.8]])
    hist = histfull / histcount
    plt.figure(figsize=(6, 6))
    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, hist.T, cmap="jet", vmin=2, vmax=5)
    plt.colorbar(label="Concentration, C")
    plt.plot([-3, -1.679], [0.38, 0.565], c="blue")
    plt.plot([-3, 0], [0.75, 0.33], c="orange")
    plt.xlabel("M20")
    plt.ylabel("Gini")


def plot_ref_hist(
    gals_list,
    value,
    bins=20,
    filt="avg",
    pdf=True,
    joint_bins=True,
    names=[],
    axis=None,
):
    """Plots a specific type of normalised histogram comparison with smoothed
    out trendline and corrected colours and labels!!!
    """
    if axis is not None:
        ax = axis
    else:
        fig, ax = plt.subplots()
    plot_hist_comp(
        gals_list, value, bins=bins, filt=filt, pdf=pdf, joint_bins=joint_bins, axis=ax
    )
    plot_smooth_comp(gals_list, value, filt=filt, pdf=pdf, nsig=bins, axis=ax)
    ax.set_xlabel(value)
    for i in range(len(gals_list)):
        c = list(ax.patches[i]._edgecolor)
        c[-1] = 0.7
        ax.lines[i].set_color(c)
        ax.lines[i].set_linewidth(1)
        ax.lines[i].set_linestyle("--")
        ax.lines[i].set_label("")
    L = ax.legend(loc=3)
    for i in range(len(gals_list)):
        if len(names) == len(gals_list):
            no = "(" + L.texts[i]._text.split("(")[-1]
            L.texts[i].set_text(f"{names[i]} {no}")
            # L.texts[i + len(names)].set_text(f"{names[i]} trendline")
    plt.draw()


def plot_smooth2d_comp(gals_list, valuex, valuey, filt="avg", axis=None, alpha=0.6):
    """!!!"""
    if axis is not None:
        ax = axis
    else:
        fig, ax = plt.subplots()
    for galaxies in gals_list:
        valsx = []
        valsy = []
        valsg = []
        for g in galaxies:
            valy = resu.get_filter_or_avg(g, valuey, filt)
            try:
                valx = g["info"][valuex]
            except:
                valx = None
            if valy is not None and valx is not None:
                valsx.append(valx)
                valsy.append(valy)
                valsg.append(g["name"])
        valsx, valsy, valsg = rem_bad_outliers([valsx, valsy, valsg])
        valsy, valsx, valsg = rem_bad_outliers([valsy, valsx, valsg])
        if filt:
            lab = f"filter {filt} ({len(valsx)})"
        else:
            lab = f"({len(valsx)})"
        sns.kdeplot(
            x=valsx, y=valsy, fill=True, cmap="Oranges", ax=ax, bw_adjust=0.3, alpha=alpha
        )
    else:
        ax.set_title(f"Approximate distribution comparison of {valuex}")


def plot_smooth2d_subt(gals, valuex, valuey, filt="avg", axis=None, rang=None):
    """!!!"""
    if axis is not None:
        ax = axis
    else:
        fig, ax = plt.subplots()
    vsx = []
    vsy = []
    vsg = []
    for galaxies in gals:
        valsx = []
        valsy = []
        valsg = []
        for g in galaxies:
            valy = resu.get_filter_or_avg(g, valuey, filt)
            valx = resu.get_filter_or_avg(g, valuex, filt)
            if valy is not None and valx is not None:
                valsx.append(valx)
                valsy.append(valy)
                valsg.append(g["name"])
        valsx, valsy, valsg = rem_bad_outliers([valsx, valsy, valsg])
        valsy, valsx, valsg = rem_bad_outliers([valsy, valsx, valsg])
        vsx.append(valsx)
        vsy.append(valsy)
        vsg.append(valsg)
    if filt:
        lab = f"filter {filt} ({len(valsx)})"
    else:
        lab = f"({len(valsx)})"
    kde1 = stats.gaussian_kde([vsx[0], vsy[0]], bw_method=0.5)
    kde2 = stats.gaussian_kde([vsx[1], vsy[1]], bw_method=0.5)
    if rang is None:
        xrang = (min(vsx[0] + vsx[1]), max(vsx[0] + vsx[1]))
        yrang = (min(vsy[0] + vsy[1]), max(vsy[0] + vsy[1]))
    else:
        xrang = rang[0]
        yrang = rang[1]
    x, y = np.mgrid[xrang[0] : xrang[1] : 500j, yrang[0] : yrang[1] : 500j]
    positions = np.vstack([x.ravel(), y.ravel()])

    z1 = kde1(positions)
    z2 = kde2(positions)
    z_diff = z1 - z2
    z_diff = z_diff.reshape(x.shape)
    plt.contourf(x, y, z_diff, levels=9, cmap="coolwarm")


def plot_points(
    gals_list, value, filt="avg", names=[], xranges=[], axis=None, xalpha=1, yalpha=1, percentiles = (33,67)
):
    """Plot a very simple plot with one point per set of galaxies and
    appropriate error bars. !!!
    """
    means = []
    medians = []
    stds = []
    e33 = []
    e68 = []
    if axis is not None:
        ax = axis
    else:
        fig, ax = plt.subplots()
    for galaxies in gals_list:
        vals = []
        valsg = []
        for g in galaxies:
            val = resu.get_filter_or_avg(g, value, filt)
            if val is not None:
                vals.append(val)
                valsg.append(g["name"])
        vals, valsg = rem_bad_outliers([vals, valsg])
        if filt:
            lab = f"filter {filt} ({len(vals)})"
        else:
            lab = f"({len(vals)})"
        means.append(np.mean(vals))
        medians.append(np.median(vals))
        stds.append(np.std(vals, mean=medians[-1]))
        e33.append(medians[-1] - np.percentile(vals, percentiles[0]))
        e68.append(np.percentile(vals, percentiles[1]) - medians[-1])
    if len(xranges) == len(gals_list):
        x = [(i[0] + i[1]) / 2 for i in xranges]
        xe = [abs(i[1] - i[0]) / 2 for i in xranges]
    elif len(names) == len(gals_list):
        x = range(len(names))
        xe = None
        ax.set_xlim(-0.5, len(names) - 0.5)
        ax.set_xticks(x, names)
    ax.errorbar(
        x,
        medians,
        yerr=(e33, e68),
        fmt="none",
        capsize=5,
        ecolor="black",
        alpha=0.5 * yalpha,
    )
    ax.errorbar(
        x, means, xerr=xe, fmt="none", capsize=5, ecolor="black", alpha=0.5 * xalpha
    )
    ax.plot(
        x,
        means,
        marker="D",
        c="#e70000a0",
        markeredgecolor = "#8b0000b1",
        markersize=5,
        linestyle="",
        label="mean",
        linewidth=1.5,
        alpha = None
    )
    ax.plot(
        x, medians, marker="", c="#e70000ff", markersize=12, linestyle=":", label="median", linewidth=2.5
    )
    # ax.plot(x, means, c="darkred", alpha=0.85, linewidth=1.3, linestyle="dotted")
    ax.set_ylabel(value)
    ax.legend()


def plot_points_comp(gals_list, valuex, valuey, filt="avg", axis=None, ealpha=0.5):
    """Plot a very simple plot with one point per set of galaxies and
    appropriate error bars. !!!
    """
    means = [[], []]
    medians = [[], []]
    stds = [[], []]
    e33 = [[], []]
    e68 = [[], []]
    if axis is not None:
        ax = axis
    else:
        fig, ax = plt.subplots()
    for galaxies in gals_list:
        vals = [[], []]
        valsg = []
        for g in galaxies:
            valx = resu.get_filter_or_avg(g, valuex, filt)
            valy = resu.get_filter_or_avg(g, valuey, filt)
            if valx is not None and valy is not None:
                vals[0].append(valx)
                vals[1].append(valy)
                valsg.append(g["name"])
        vals[0], vals[1], valsg = rem_bad_outliers([vals[0], vals[1], valsg])
        if filt:
            lab = f"filter {filt} ({len(vals)})"
        else:
            lab = f"({len(vals)})"
        means[0].append(np.mean(vals[0]))
        means[1].append(np.mean(vals[1]))
        medians[0].append(np.median(vals[0]))
        medians[1].append(np.median(vals[1]))
        stds[0].append(np.std(vals[0], mean=medians[0][-1]))
        stds[1].append(np.std(vals[1], mean=medians[1][-1]))
        e33[0].append(means[0][-1] - np.percentile(vals[0], 33))
        e33[1].append(means[1][-1] - np.percentile(vals[1], 33))
        e68[0].append(np.percentile(vals[0], 68) - means[0][-1])
        e68[1].append(np.percentile(vals[1], 68) - means[1][-1])

    ax.errorbar(
        means[0],
        means[1],
        xerr=(e33[0], e68[0]),
        yerr=(e33[1], e68[1]),
        fmt="none",
        capsize=5,
        ecolor="black",
        alpha=ealpha,
    )
    ax.plot(
        means[0],
        means[1],
        marker="D",
        c="darkred",
        markersize=4,
        linestyle="",
        label="mean",
    )
    ax.plot(
        medians[0],
        medians[1],
        marker="+",
        c="darkred",
        markersize=12,
        linestyle="",
        label="median",
    )
    ax.plot(
        means[0], means[1], c="darkred", alpha=0.85, linewidth=1.3, linestyle="dotted"
    )
    ax.set_xlabel(valuex)
    ax.set_ylabel(valuey)
    ax.legend()


def plot_violin(gals_list, value, filt="avg", names=[], xranges=[], axis=None):
    """Plot a very simple violin plot with one 1d histogram per set of
    galaxies!!!
    """
    if axis is not None:
        ax = axis
    else:
        fig, ax = plt.subplots()
    valss = []
    for galaxies in gals_list:
        vals = []
        valsg = []
        for g in galaxies:
            val = resu.get_filter_or_avg(g, value, filt)
            if val is not None:
                vals.append(val)
                valsg.append(g["name"])
        vals, valsg = rem_bad_outliers([vals, valsg])
        valss.append(vals)

    if len(xranges) == len(gals_list):
        x = [(i[0] + i[1]) / 2 for i in xranges]
    elif len(names) == len(gals_list):
        x = list(range(len(names)))
        ax.set_xlim(-0.5, len(names) - 0.5)
        ax.set_xticks(x, names)
    for i in range(len(gals_list)):
        sns.violinplot(
            x=x[i],
            y=valss[i] * 2,
            hue=[0] * len(valss[i]) + [1] * len(valss[i]),
            split=True,
            palette={0: "#ffa333", 1: "#ffa333"},
            ax=ax,
            alpha = 0.7,
            inner=None,
            bw_adjust=0.6,
            linecolor="#854900d5",
            linewidth=1,
        )
        if not i % 2:
            ax.collections[-2].set_alpha(0)
            ax.collections[-1].set(facecolor="#4b92c3", alpha = 0.8, edgecolor="#1d3e55")
        else:
            ax.collections[-1].set_alpha(0)
            ax.collections[-2].set(facecolor="#ffa333", alpha = 0.8)
    ax.set_ylabel(value)
    ax.legend()


def points_multi_bins(
    galaxies,
    value,
    filt="avg",
    hist=True,
    axis=None,
    no_legend=False,
    include={0, 1, 2, 3},
    balpha=0.6,
):
    """!!!"""
    il = list(include)
    il.sort()
    n = {k: il.index(k) for k in include}
    if axis is not None and len(axis) == len(include):
        axs = axis
        fig = plt.gcf()
    else:
        fig = plt.figure(figsize=(5 * (len(include) - 1), 5))
        gs = fig.add_gridspec(1, len(include), wspace=0)
        axs = gs.subplots(sharey=True)
    if 0 in include:
        z = resu.get_separate_in_value(galaxies, "z bin")
        plot_points(
            (z["low_z"], z["high_z"]),
            value,
            xranges=[[0.8, 1.3], [2, 2.5]],
            axis=axs[n[0]],
        )
        if hist:
            plot_smooth2d_comp(
                (galaxies,), "ZBEST", value, axis=axs[n[0]], alpha=balpha
            )
        axs[n[0]].set_xlabel("$z$")
    if 1 in include:
        s = resu.get_bins_in_value(galaxies, "DMS", bins=7)
        plot_points(
            [s[k] for k in s],
            value,
            xranges=[k for k in s],
            axis=axs[n[1]],
            xalpha=0.25,
            yalpha=0.7,
        )
        if hist:
            plot_smooth2d_comp((galaxies,), "DMS", value, axis=axs[n[1]], alpha=balpha)
        axs[n[1]].set_xlabel("$\\log(SFR/SFR_{MS})$")
    if 2 in include:
        m = resu.get_bins_in_value(galaxies, "LMSTAR", bins=7)
        plot_points(
            [m[k] for k in m],
            value,
            xranges=[k for k in m],
            axis=axs[n[2]],
            xalpha=0.25,
            yalpha=0.6,
        )
        if hist:
            plot_smooth2d_comp(
                (galaxies,), "LMSTAR", value, axis=axs[n[2]], alpha=balpha
            )
        axs[n[2]].set_xlabel("$\\log (M_\\star/M_\\odot)$")
    if 3 in include:
        r = resu.get_bins_in_value(galaxies, "H_RE", bins=7)
        plot_points(
            [r[k] for k in r],
            value,
            xranges=[k for k in r],
            axis=axs[n[3]],
            xalpha=0.25,
            yalpha=0.6,
        )
        if hist:
            plot_smooth2d_comp((galaxies,), "H_RE", value, axis=axs[n[3]], alpha=balpha)
        axs[n[3]].set_xlabel("$r_\\mathrm{eff}\\, (\\mathrm{arcsec})$")
    for ax in axs:
        if len(axs) > 1:
            ax.label_outer()
        ax.set_title("")
        if not no_legend:
            ax.legend(loc=1)
        elif ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.suptitle(f"Value of {value} for different sample bins")

    fig.tight_layout()


def points_multi_clas(
    galaxies, value, filt="avg", axis=None, no_legend=False, new_names=None, percentiles = (33,67)
):
    """!!!"""
    if new_names is None:
        names = {
            "bulges": "Bulge",
            "mergers": "Interacting-Merger",
            "spirals": "Spiral",
            "clumpy": "Clumpy",
        }
    else:
        names = new_names
    files = {
        k: run.fetch_json("dict_in/n4_" + v + ".json")["galaxies"]
        for (k, v) in names.items()
    }
    separ = {
        k: (resu.get_subset(galaxies, v), resu.get_complement(galaxies, v))
        for (k, v) in files.items()
    }
    if axis is not None and len(axis) == len(names):
        axs = axis
        fig = plt.gcf()
    else:
        fig = plt.figure(figsize=(5 * (len(names) - 1), 5))
        gs = fig.add_gridspec(1, len(names), wspace=0)
        axs = gs.subplots(sharey=True)
    c = 0
    for k in separ:
        plot_points(
            (separ[k][1], separ[k][0]), value, names=["non-" + k, k], axis=axs[c], percentiles = percentiles
        )
        plot_violin(
            (separ[k][1], separ[k][0]), value, names=["non-" + k, k], axis=axs[c]
        )
        axs[c].set_label(f"${k}$")
        if len(axs) > 1:
            axs[c].label_outer()
        if not no_legend:
            axs[c].legend(loc=1)
        elif axs[c].get_legend() is not None:
            axs[c].get_legend().remove()
        c += 1
    fig.suptitle(f"Value of {value} for different classification")
    fig.tight_layout()


def hist_multi_bins(galaxies, value, filt="avg", bins=20, axis=None, no_legend=False):
    """!!!"""
    if axis is not None and len(axis) == 4:
        axs = axis
        fig = plt.gcf()
    else:
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 4, wspace=0)
        axs = gs.subplots(sharex=True)
    z = resu.get_separate_in_value(galaxies, "z bin")
    s = resu.get_separate_in_value(galaxies, "SFR bin")
    m = resu.get_separate_in_value(galaxies, "M bin")
    r = resu.get_separate_in_value(galaxies, "size bin")
    plot_ref_hist(
        (z["low_z"], z["high_z"]),
        value,
        bins=bins,
        names=["low z", "high z"],
        axis=axs[0],
    )
    plot_ref_hist(
        (s["low_sfr"], s["high_sfr"]),
        value,
        bins=bins,
        names=["low sfr", "high sfr"],
        axis=axs[1],
    )
    plot_ref_hist(
        (m["low_m"], m["high_m"]),
        value,
        bins=bins,
        names=["low M", "high M"],
        axis=axs[2],
    )
    plot_ref_hist(
        (r["small"], r["big"]),
        value,
        bins=bins,
        names=["small R", "big R"],
        axis=axs[3],
    )
    for ax in axs:
        ax.set_xlabel(value)
        ax.set_title("")
        ax.yaxis.set_ticks([])
        if no_legend and ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.suptitle(f"Value of {value} for different sample bins")

    fig.tight_layout()


def hist_multi_clas(galaxies, value, filt="avg", bins=20, axis=None, no_legend=False):
    """!!!"""
    names = {
        "bulges": "Bulge",
        "interact./mergers": "Interacting-Merger",
        "spirals": "Spiral",
        "clumpy": "Clumpy",
    }
    files = {
        k: run.fetch_json("dict_in/n4_" + v + ".json")["galaxies"]
        for (k, v) in names.items()
    }
    separ = {
        k: (resu.get_subset(galaxies, v), resu.get_complement(galaxies, v))
        for (k, v) in files.items()
    }
    if axis is not None and len(axis) == len(names):
        axs = axis
        fig = plt.gcf()
    else:
        fig = plt.figure(figsize=(5 * (len(names) - 1), 5))
        gs = fig.add_gridspec(1, len(names), wspace=0)
        axs = gs.subplots(sharex=True)
    c = 0
    for k in separ:
        plot_ref_hist(
            (separ[k][1], separ[k][0]),
            value,
            bins=bins,
            names=["non-" + k, k],
            axis=axs[c],
        )
        axs[c].set_label(f"${k}$")
        axs[c].set_title("")
        axs[c].yaxis.set_ticks([])
        if no_legend and axs[c].get_legend() is not None:
            axs[c].get_legend().remove()
        c += 1
    fig.suptitle(f"Value of {value} for different classification")
    fig.tight_layout()


def plot_grided(galaxies, values, len_funct, funct, title="", *args, **kwargs):
    """!!!"""
    fig = plt.figure(figsize=(len(values) * 3.5, len_funct * 3.5))
    gs = fig.add_gridspec(len(values), len_funct, wspace=0, hspace=0)
    axs = gs.subplots(sharey="row", sharex="col")
    if len(axs.shape) == 1:
        axs = np.array([[ax] for ax in axs])
    for i in range(len(values)):
        funct(galaxies, values[i], axis=axs[i], no_legend=True, *args, **kwargs)
    for ax in axs.flatten():
        ax.patch.set_alpha(0)
    fig.suptitle(title)
    fig.tight_layout()
