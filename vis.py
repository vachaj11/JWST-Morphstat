"""various methods for visualisation of the results
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpi
import numpy as np
import resu


def rem_bad_outliers(data, sig = 5):
    """removes data from the set that are more than n-sigma out 
    """
    mean = np.mean(data[0])
    std = np.std(data[0])
    ind = []
    for i in range(len(data[0])):
        if data[0][i]-mean < -sig*std or data[0][i]-mean> sig*std:
            ind.append(i)
    ind.reverse()
    if ind:
        print(f"found {len(ind)} very bad (>{sig} sigma) outliers in the data")
    for dat in data:
        for i in ind:
            dat.pop(i)
    return data

def plot_value(galaxies, valuex, valuey, filt = "aver"):
    """plots requested values (for a given filter or averaged) for a provided set
    of galaxies
    """
    fig, ax = plt.subplots()
    valsy = []
    valsx = []
    for g in galaxies:
        valy = resu.get_filter_or_avg(g, valuey, filt)
        if valy:
            valsy.append(valy)
            valsx.append(g["info"][valuex])
    valsy, valsx = rem_bad_outliers([valsy,valsx])
    if filt:
        ax.plot(valsx, valsy, linestyle="", marker="+", label=f"filter {filt} ({len(valsx)})")
    else:
        ax.plot(valsx, valsy, linestyle="", marker="+", label=f"({len(valsx)})")
    ax.set(xlabel = valuex, ylabel = valuey)
    fig.canvas.mpl_connect('button_press_event', lambda x:print(str(x.xdata)+" "+ str(x.ydata)))
    
def plot_correlation(galaxies, valuex, valuey, filt = "aver"):
    """plots requested values (for a given filter or averaged) against each
    other or avaraged for a provided set of galaxies
    """
    valsy = []
    valsx = []
    for g in galaxies:
        valy = resu.get_filter_or_avg(g, valuey, filt)
        valx = resu.get_filter_or_avg(g, valuex, filt)
        if valy and valx:
            valsy.append(valy)
            valsx.append(valx)
    valsy, valsx = rem_bad_outliers([valsy,valsx])
    valsx, valsy = rem_bad_outliers([valsx,valsy])
    if filt:
        plt.plot(valsx, valsy, linestyle="", marker="+", label=f"filter {filt} ({len(valsx)})")
    else:
        plt.plot(valsx, valsy, linestyle="", marker="+", label=f"({len(valsx)})")
    plt.xlabel(valuex)
    plt.ylabel(valuey)

def plot_value_difference(gals1, gals2, valuex, valuey, filt = "aver"):
    """plots difference in requested value (for a given filter or averaged)
    for provided two sets of galaxies (and for different filters if filt is
    provided as a list)
    """
    valsd = []
    valsx = []
    galf1 = resu.get_subset(gals1, gals2)
    galf2 = resu.get_subset(gals2, galf1)
    for i in range(len(galf1)):
        if type(filt) == list and len(filt)==2:
            val1 = resu.get_filter_or_avg(galf1[i], valuey, filt[0])
            val2 = resu.get_filter_or_avg(galf2[i], valuey, filt[1])
        else:
            val1 = resu.get_filter_or_avg(galf1[i], valuey, filt)
            val2 = resu.get_filter_or_avg(galf2[i], valuey, filt)
        if val1 and val2:
            valsd.append(val2-val1)
            valsx.append(galf1[i]["info"][valuex])
    #valsd, valsx = rem_bad_outliers([valsd,valsx])
    if filt:
        plt.plot(valsx, valsd, linestyle="", marker="+", label=f"filter {filt} ({len(valsx)})")
    else:
        plt.plot(valsx, valsd, linestyle="", marker="+", label=f"({len(valsx)})")
    plt.xlabel(valuex)
    plt.ylabel(valuey)

def plot_histogram(galaxies, value, nbins=None, filt = "aver"):
    """plots a histogram of requested value (for a given filter or averaged)
    for a given set of galaxies
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
        plt.stairs(count/len(vals), bins, label=f"filter {filt} ({len(vals)})")
    else:
        plt.stairs(count/len(vals), bins, label=f"({len(vals)})")


def plot_hist_comp(galaxies1, galaxies2, value, nbins=None, filt = "aver"):
    """plots a joint histogram of requested value (for a given filter or
    averaged) for given two sets of galaxies
    """
    vals1 = []
    vals2 = []
    for g in galaxies1:
        val = resu.get_filter_or_avg(g, value, filt)
        if val:
            vals1.append(val)
    vals1 = rem_bad_outliers([vals1])[0]
    if nbins is not None:
        count1, bins1 = np.histogram(vals1, nbins)
    else:
        count1, bins1 = np.histogram(vals1)
    for g in galaxies2:
        val = resu.get_filter_or_avg(g, value, filt)
        if val:
            vals2.append(val)
    vals2 = rem_bad_outliers([vals2])[0]
    if nbins is not None:
        count2, bins2 = np.histogram(vals2, nbins)
    else:
        count2, bins2 = np.histogram(vals2)
    if filt:
        plt.stairs(count1 / len(vals1), bins1, label=f"filter {filt} ({len(vals1)})")
        plt.stairs(count2 / len(vals2), bins2, label=f"filter {filt} ({len(vals2)})")
    else:
        plt.stairs(count1 / len(vals1), bins1, label=f"({len(vals1)})")
        plt.stairs(count2 / len(vals2), bins2, label=f"({len(vals2)})")
    plt.title(f"Histogram comparison of {value}")
    plt.legend()

def plot_value_filters(galaxies, valuex, valuey, filt = 2):
    """plots requested values for a provided set of galaxies across multiple
    filters
    """
    if type(filt) == int:
        filts = resu.get_most_filters(galaxies, filt)
    elif type(filt) in (list,set,tuple):
        filts = list(filt)
    else:
        print("unrecognised type of filt: "+str(type(filt)))
    for filt in filts:
        plot_value(galaxies, valuex, valuey, filt=filt)

def plot_correlation_filters(galaxies, valuex, valuey, filt = 2):
    """plots requested values agains each other for a provided set of galaxies
    across multiple filters
    """
    if type(filt) == int:
        filts = resu.get_most_filters(galaxies, filt)
    elif type(filt) in (list,set,tuple):
        filts = list(filt)
    else:
        print("unrecognised type of filt: "+str(type(filt)))
    for filt in filts:
        plot_correlation(galaxies, valuex, valuey, filt=filt)
        
def plot_histogram_filters(galaxies, value, filt = 2):
    """plots histograms of requested value for a given set of galaxies across
    multiple filters
    """
    if type(filt) == int:
        filts = resu.get_most_filters(galaxies, filt)
    elif type(filt) in (list,set,tuple):
        filts = list(filt)
    else:
        print("unrecognised type of filt: "+str(type(filt)))
    for filt in filts:
        plot_histogram(galaxies, value, filt=filt)
    plt.title(f"Histogram of {value}")
    
def plot_pic_value(galaxy, values = ["C","A","S"]):
    """for a given galaxy show its colour picture jointly with graphs of
    requested values as a function of wavelength
    """
    fig = plt.figure()
    gs = fig.add_gridspec(len(values)+1, hspace=0,height_ratios=[3]+[1 for i in values])
    axs = gs.subplots()
    fig.suptitle(galaxy["name"])
    pic_path = "../colour_images/"+galaxy["name"]+".png"
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
        axs[i+1].plot(wavel, data[i])
        axs[i+1].set(xlabel="wavelength", ylabel = values[i])
        axs[i+1].label_outer()
        axs[i+1].sharex(axs[1])
