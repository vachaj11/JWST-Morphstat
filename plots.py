import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import line_separation as ls
import resu
import run
import vis
from ren_values import *

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern",
        "font.size": 15,
    }
)

"""To generate the plots, first update the paths bellow to correspond to the location of the output/input dictionaries. 
"""

remove = ["GS4_18890", "COS4_17600", "GS4_19797", "COS4_05758", "U4_28125", "G4_06569", "GS4_30028", "GS4_29773"]
rem_g = [{"name":g} for g in remove]

# main data 
raw_o = run.fetch_json("dict_out/out_full_matched_5_m.json")["galaxies"]
raw = resu.get_complement(raw_o, rem_g)
bulges = run.fetch_json("dict_in/n4_Bulge.json")["galaxies"]
mergers = run.fetch_json("dict_in/n4_Interacting-Merger.json")["galaxies"]
clumpy = run.fetch_json("dict_in/n4_Clumpy.json")["galaxies"]
ful_o = run.fetch_json("dict_in/dictionary_full.json")["galaxies"]
full = resu.get_complement(ful_o, rem_g)

# sersic data
ser_o = run.fetch_json("dict_out/out_full_444.json")["galaxies"]
sers = resu.get_complement(ser_o, rem_g)
ser150_o = run.fetch_json("dict_out/out_full_150.json")["galaxies"]
sers150 = resu.get_complement(ser150_o, rem_g)
ser150hst_o = run.fetch_json("dict_out/out_full_150_hst.json")["galaxies"]
sers150hst = resu.get_complement(ser150hst_o, rem_g)
serhst = run.fetch_json("dict_out/out_hst_160_full.json")["galaxies"]
sershst = resu.get_complement(serhst, rem_g)
ser150sc_o = run.fetch_json("dict_out/out_full_150_scaled.json")["galaxies"]
sers150sc = resu.get_complement(ser150sc_o, rem_g)
ser150sc3_o = run.fetch_json("dict_out/out_full_150_scaled_3.json")["galaxies"]
sers150sc3 = resu.get_complement(ser150sc3_o, rem_g)
sers150sc30_o = run.fetch_json("dict_out/out_full_150_scaled_30.json")["galaxies"]
sers150sc30 = resu.get_complement(sers150sc30_o, rem_g)
sersfull_o = run.fetch_json("dict_out/out_full_sersic.json")["galaxies"]
sersfull = resu.get_complement(sersfull_o, rem_g)

fil = resu.galaxy_pruning(raw)
filmb = resu.get_subset(fil, bulges)
filob = resu.get_complement(fil, bulges)
filmm = resu.get_subset(fil, mergers)
filom = resu.get_complement(fil, mergers)
filmc = resu.get_subset(fil, clumpy)
filoc = resu.get_complement(fil, clumpy)

fils = resu.galaxy_pruning(sers, strength="sersic")
fils150 = resu.galaxy_pruning(sers150, strength="sersic")
fils150hst = resu.galaxy_pruning(sers150hst, strength="sersic")
filshst = resu.galaxy_pruning(sershst, strength="sersic")
fils150sc = resu.galaxy_pruning(sers150sc, strength="sersic")
fils150sc3 = resu.galaxy_pruning(sers150sc3, strength="sersic")
fils150sc30 = resu.galaxy_pruning(sers150sc30, strength="sersic")
filsfull = resu.galaxy_pruning(sersfull, strength="sersic")

def mergers_separation(mass = None, save = None, e = None, axis = None):
    if mass is None:
        filhmm = filmm
        filhom = filom
    elif mass == "low":
        filhmm = resu.get_separate_in_value(filmm, "M bin")["low_m"]
        filhom = resu.get_separate_in_value(filom, "M bin")["low_m"]
    elif mass == "high":
        filhmm = resu.get_separate_in_value(filmm, "M bin")["high_m"]
        filhom = resu.get_separate_in_value(filom, "M bin")["high_m"]
    else:
        print("Unknown mass bin: " + str(mass))
        return None
    if axis is None:
        fig = plt.gcf()
        axs = plt.gca()
    else:
        fig = plt.gcf()
        axs = axis
    blist = []
    point, slope, diff = ls.max_sep(filhmm, filhom, "M20", "Gini", e = e, rfull=blist)
    po_co = ls.error_poly_2(blist, 0.01, (-2.3,-0.8), (0.38, 0.64),mdif=diff[0])
    poly = Polygon(po_co, color="#ffb6b6ff")
    axs.add_patch(poly)
    vis.plot_correlation_filters((filhom, filhmm), "M20", "Gini", axis = axs)
    x = np.array([-2.3, -0.8])


    axs.plot(
        x,
        -0.14 * x + 0.33,
        linestyle="--",
        linewidth=2.5,
        c="black",
        label="Lotz et al. (2008)",
    )
    axs.plot(
        x,
        (point[0] - x) * (-slope) + point[1],
        linewidth=2.5,
        c="#e70000ff",
        label="line of best separation",
    )

    mmabove = ls.get_above_line(filhmm, "M20", "Gini", [0, 0.33], -0.14)
    omabove = ls.get_above_line(filhom, "M20", "Gini", [0, 0.33], -0.14)
    tabove = ls.get_above_line(filhom+filhmm, "M20", "Gini", [0, 0.33], -0.14)
    print("\n------------------")
    print("Mergers separation")
    print("------------------")
    print("Line used in literature:")
    print(f"Gini = M_20*({-0.14:.3f})+({0.33:.3f})")
    print("Separation:")
    print(
        f"{100*mmabove[1][0]:.1f} % mergers above, {omabove[1][1]*100:.1f} % non-mergers bellow"
    )
    print(f"{tabove[2][0]*100:.1f} % merger fraction, in the sample {diff[2][1]*100:.1f} %")
    print("------------------")
    print("Best separation line:")
    print(f"Gini = M_20*({slope:.3f})+({point[1]-slope*point[0]:.3f})")
    print("Separation:")
    print(
        f"{100*diff[1][0]:.1f} % mergers above, {diff[1][1]*100:.1f} % non-mergers bellow")
    print(f"{diff[2][0]*100:.1f} % merger fraction, in the sample {diff[2][1]*100:.1f} %")
    print("------------------\n")

    axs.set_xlabel("M$_{20}$")
    axs.set_ylabel("Gini")
    axs.set_title("")  # Gini-M$_{20}$ statistics for non/mergers")
    """
    axs.lines[0].set(
        c="#034a8360", markeredgecolor="#05396f9d", markersize=9, alpha=0.37
    )
    axs.lines[1].set(
        c="#b7010154", markeredgecolor="#b701018b", markersize=9, alpha=0.37
    )
    """
    axs.lines[0].set(
        c="#267ab46b", markeredgecolor="#00346899", markersize=10, alpha = None
    )
    axs.lines[1].set(
        c="#ff910c7c", markeredgecolor="#854900d5", markersize=13, alpha = None
    )
    L = axs.legend()
    L.texts[0].set_text("others (" + L.texts[0]._text.split("(")[-1])
    L.texts[1].set_text("interact./mergers (" + L.texts[1]._text.split("(")[-1])
    axs.set_xlim(x)
    axs.set_ylim(0.38, 0.64)
    axs.tick_params(
        axis="both",
        which="major",
        bottom=True,
        top=False,
        left=True,
        right=False,
        direction="inout",
        size=6,
        labelsize=14,
    )
    if axis is None:
        fig.set_size_inches(5.5, 5)
        fig.set_layout_engine(layout="tight")
        if save is not None:
            fig.savefig(save)

def mergers_nstatistics(mass = None, save = None, e = None):
    if mass is None:
        filhmm = filmm
        filhom = filom
    elif mass == "low":
        filhmm = resu.get_separate_in_value(filmm, "M bin")["low_m"]
        filhom = resu.get_separate_in_value(filom, "M bin")["low_m"]
    elif mass == "high":
        filhmm = resu.get_separate_in_value(filmm, "M bin")["high_m"]
        filhom = resu.get_separate_in_value(filom, "M bin")["high_m"]
    else:
        print("Unknown mass bin: " + str(mass))
        return None
    blist = []
    point, slope, diff = ls.max_sep(filhmm, filhom, "M20", "Gini", param="NM", e = e, rfull=blist)
    po_co = ls.error_poly(blist, 0.01, (-2.3,-0.8), (0.38, 0.64))
    poly = Polygon(po_co, color="grey")
    axs.add_patch(poly)
    vis.plot_ref_hist(
        (filhom, filhmm), "NM", pdf=True, bins=13, names=["non-mergers", "mergers"]
    )
    x = 0
    fig = plt.gcf()
    axs = plt.gca()
    axs.plot(
        [x, x],
        [0, 14],
        linewidth=2.5,
        c="gold",
        label="line of best separation",
    )
    print("\n------------------")
    print("Mergers separation")
    print("------------------")
    print("Best separation line:")
    print(f"NF(G, M20) = {x:.3f}")
    print("Separation:")
    print(
        f"{100*(diff[1][0]):.1f} % mergers on right, {(diff[1][1])*100:.1f} % non-mergers on left"
    )
    print("------------------\n")

    axs.set_xlabel("Distance from line of best separation")
    axs.set_ylabel("Normalised distribution")
    axs.set_title("")  # Gini-M$_{20}$ statistics for non/mergers")
    axs.patches[0].set(
        edgecolor="#05396f9d", facecolor="#03498233", fill=True, linewidth=1.25
    )
    axs.patches[1].set(
        edgecolor="#b701018b", facecolor="#B7010133", fill=True, linewidth=1.25
    )
    axs.lines[0].set(c="#053970CC", linewidth=2.5)
    axs.lines[1].set(c="#B70101CC", linewidth=2.5)
    L = axs.legend()
    L.texts[0].set_text("others (" + L.texts[0]._text.split("(")[-1])
    L.texts[1].set_text("interact./mergers (" + L.texts[1]._text.split("(")[-1])
    axs.set_xlim(-0.25, 0.25)
    axs.set_ylim(0, 13)
    axs.tick_params(
        axis="both",
        which="major",
        bottom=True,
        top=False,
        left=True,
        right=False,
        direction="inout",
        size=6,
        labelsize=14,
    )
    fig.set_size_inches(5.5, 5)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)

def mergers_separation_a(mass = None, save = None, e = None):
    if mass is None:
        filhmm = filmm
        filhom = filom
    elif mass == "low":
        filhmm = resu.get_separate_in_value(filmm, "M bin")["low_m"]
        filhom = resu.get_separate_in_value(filom, "M bin")["low_m"]
    elif mass == "high":
        filhmm = resu.get_separate_in_value(filmm, "M bin")["high_m"]
        filhom = resu.get_separate_in_value(filom, "M bin")["high_m"]
    else:
        print("Unknown mass bin: " + str(mass))
        return None
    point, slope, diff = ls.max_sep(filhmm, filhom, "A", "A", e=e)
    vis.plot_ref_hist(
        (filhom, filhmm), "A", pdf=True, bins=13, names=["non-mergers", "mergers"]
    )
    x = (point[1] - point[0] * slope) / (1 - slope)
    fig = plt.gcf()
    axs = plt.gca()

    axs.plot(
        [0.35, 0.35],
        [0, 7],
        linestyle="--",
        linewidth=2.5,
        c="black",
        label="Conselice et al. (2003)",
    )
    axs.plot(
        [x, x],
        [0, 7],
        linewidth=2.5,
        c="#e70000ff",
        label="line of best separation",
    )
    mmabove = ls.get_above_line(filhmm, "A", "A", [0.35, 0.35], 2)
    omabove = ls.get_above_line(filhom, "A", "A", [0.35, 0.35], 2)
    tabove = ls.get_above_line(filhom+filhmm, "A", "A", [0.35, 0.35], 2)
    print("\n------------------")    
    print("Mergers separation")
    print("------------------")
    print("Line used in literature:")
    print(f"A = 0.35")
    print("Separation:")
    print(
        f"{100*(1-mmabove[1][0]):.1f} % mergers on right, {(1-omabove[1][1])*100:.1f} % non-mergers on left"
    )
    print(f"{tabove[2][0]*100:.1f} % merger fraction, in the sample {diff[2][1]*100:.1f} %")
    print("------------------")
    print("Best separation line:")
    print(f"A = {x:.3f}")
    print("Separation:")
    print(
        f"{100*(1-diff[1][0]):.1f} % mergers on right, {(1-diff[1][1])*100:.1f} % non-mergers on left"
    )
    print(f"{diff[2][0]*100:.1f} % merger fraction, in the sample {diff[2][1]*100:.1f} %")
    print("------------------\n")

    axs.set_xlabel("Asymmetry statistics")
    axs.set_ylabel("Normalised distribution")
    axs.set_title("")  # Gini-M$_{20}$ statistics for non/mergers")
    axs.patches[0].set(
        edgecolor="#00346899", facecolor="#1973b19d", fill=True, linewidth=1.25, alpha = None
    )
    axs.patches[1].set(
        edgecolor="#854900d5", facecolor="#ff8e049f", fill=True, linewidth=1.25, alpha = None
    )
    axs.lines[0].set(c="#004d99ff", linewidth=2.5)
    axs.lines[1].set(c="#b86500ff", linewidth=2.5)
    L = axs.legend()
    L.texts[0].set_text("others (" + L.texts[0]._text.split("(")[-1])
    L.texts[1].set_text("interact./mergers (" + L.texts[1]._text.split("(")[-1])
    axs.set_xlim(-0.1, 0.8)
    axs.set_ylim(0, 6.5)
    axs.tick_params(
        axis="both",
        which="major",
        bottom=True,
        top=False,
        left=True,
        right=False,
        direction="inout",
        size=6,
        labelsize=14,
    )
    fig.set_size_inches(5.5, 5)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)


def bulges_separation(mass = None, save = None, e = None, axis = None):
    if mass is None:
        filhmb = filmb
        filhob = filob
    elif mass == "low":
        filhmb = resu.get_separate_in_value(filmb, "M bin")["low_m"]
        filhob = resu.get_separate_in_value(filob, "M bin")["low_m"]
    elif mass == "high":
        filhmb = resu.get_separate_in_value(filmb, "M bin")["high_m"]
        filhob = resu.get_separate_in_value(filob, "M bin")["high_m"]
    else:
        print("Unknown mass bin: " + str(mass))
        return None
    if axis is None:
        fig = plt.gcf()
        axs = plt.gca()
    else:
        fig = plt.gcf()
        axs = axis
    blist = []
    point, slope, diff = ls.max_sep(filhmb, filhob, "M20", "Gini", e=e, rfull=blist)
    po_co = ls.error_poly_2(blist, 0.01, (-2.3,-0.8), (0.38, 0.64))
    poly = Polygon(po_co, color="#ffb6b6ff")
    axs.add_patch(poly)
    vis.plot_correlation_filters((filhob, filhmb), "M20", "Gini", axis = axs)
    x = np.array([-2.3, -0.8])
    """
    axs.plot(
        x,
        0.14 * x + 0.8,
        linestyle="--",
        linewidth=2.5,
        c="black",
        label="early/late boundary",
    )
    """
    axs.plot(
        x,
        (point[0] - x) * (-slope) + point[1],
        linewidth=2.5,
        c="#e70000ff",
        label="line of best separation",
    )
    
    mbabove = ls.get_above_line(filhmb, "M20", "Gini", [0, 0.8], 0.14)
    obabove = ls.get_above_line(filhob, "M20", "Gini", [0, 0.8], 0.14)
    tabove = ls.get_above_line(filhob+filhmb, "M20", "Gini", [0, 0.8], 0.14)
    print("\n-----------------")
    print("Bulges separation")
    print("-----------------")
    print("Line used in literature:")
    print(f"Gini = M_20*({0.14:.3f})+({0.8:.3f})")
    print("Separation:")
    print(
        f"{100*(1-mbabove[1][0]):.1f} % bulges bellow, {(1-obabove[1][1])*100:.1f} % non-bulges above"
    )
    print(f"{tabove[2][0]*100:.1f} % bulge fraction, in the sample {diff[2][1]*100:.1f} %")
    print("-----------------")
    print("Best separation line:")
    print(f"Gini = M_20*({slope:.3f})+({point[1]-slope*point[0]:.3f})")
    print("Separation:")
    print(
        f"{100*(1-diff[1][0]):.1f} % bulges bellow, {(1-diff[1][1])*100:.1f} % non-bulges above"
    )
    print(f"{diff[2][0]*100:.1f} % bulge fraction, in the sample {diff[2][1]*100:.1f} %")
    print("-----------------\n")

    axs.set_xlabel("M$_{20}$")
    axs.set_ylabel("Gini")
    axs.set_title("")  # Gini-M$_{20}$ statistics for non/bulges")
    axs.lines[0].set(
        c="#267ab46b", markeredgecolor="#00346899", markersize=10, alpha = None
    )
    axs.lines[1].set(
        c="#ff910c54", markeredgecolor="#8549009b", markersize=10, alpha = None
    )
    L = axs.legend()
    L.texts[0].set_text("without bulge (" + L.texts[0]._text.split("(")[-1])
    L.texts[1].set_text("with bulge (" + L.texts[1]._text.split("(")[-1])
    axs.set_xlim(x)
    axs.set_ylim(0.38, 0.64)
    axs.tick_params(
        axis="both",
        which="major",
        bottom=True,
        top=False,
        left=True,
        right=False,
        direction="inout",
        size=6,
        labelsize=14,
    )
    if axis is None:
        fig.set_size_inches(5.5, 5)
        fig.set_layout_engine(layout="tight")
        if save is not None:
            fig.savefig(save)

def bulges_nstatistics(mass = None, save = None, e = None):
    if mass is None:
        filhmb = filmb
        filhob = filob
    elif mass == "low":
        filhmb = resu.get_separate_in_value(filmb, "M bin")["low_m"]
        filhob = resu.get_separate_in_value(filob, "M bin")["low_m"]
    elif mass == "high":
        filhmb = resu.get_separate_in_value(filmb, "M bin")["high_m"]
        filhob = resu.get_separate_in_value(filob, "M bin")["high_m"]
    else:
        print("Unknown mass bin: " + str(mass))
        return None
    point, slope, diff = ls.max_sep(filhmb, filhob, "M20", "Gini", param="NM", e = e)
    vis.plot_ref_hist(
        (filhob, filhmb), "NM", pdf=True, bins=17, names=["non-bulges", "bulges"]
    )
    x = 0
    fig = plt.gcf()
    axs = plt.gca()
    axs.plot(
        [x, x],
        [0, 14],
        linewidth=2.5,
        c="gold",
        label="line of best separation",
    )
    print("\n------------------")
    print("Bulges separation")
    print("------------------")
    print("Best separation line:")
    print(f"NS(G, M20) = {x:.3f}")
    print("Separation:")
    print(
        f"{100*(1-diff[1][0]):.1f} % bulges on right, {(1-diff[1][1])*100:.1f} % non-bulges on left"
    )
    print("------------------\n")

    axs.set_xlabel("Distance from line of best separation")
    axs.set_ylabel("Normalised distribution")
    axs.set_title("")  # Gini-M$_{20}$ statistics for non/mergers")
    axs.patches[0].set(
        edgecolor="#05396f9d", facecolor="#03498233", fill=True, linewidth=1.25
    )
    axs.patches[1].set(
        edgecolor="#b701018b", facecolor="#B7010133", fill=True, linewidth=1.25
    )
    axs.lines[0].set(c="#053970CC", linewidth=2.5)
    axs.lines[1].set(c="#B70101CC", linewidth=2.5)
    L = axs.legend()
    L.texts[0].set_text("without bulge (" + L.texts[0]._text.split("(")[-1])
    L.texts[1].set_text("with bulge (" + L.texts[1]._text.split("(")[-1])
    axs.set_xlim(-0.3, 0.3)
    axs.set_ylim(0, 10)
    axs.tick_params(
        axis="both",
        which="major",
        bottom=True,
        top=False,
        left=True,
        right=False,
        direction="inout",
        size=6,
        labelsize=14,
    )
    fig.set_size_inches(5.5, 5)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
        
def clumpy_separation(mass = None, save = None, e = None, axis = None):
    if mass is None:
        filhmc = filmc
        filhoc = filoc
    elif mass == "low":
        filhmc = resu.get_separate_in_value(filmc, "M bin")["low_m"]
        filhoc = resu.get_separate_in_value(filoc, "M bin")["low_m"]
    elif mass == "high":
        filhmc = resu.get_separate_in_value(filmc, "M bin")["high_m"]
        filhoc = resu.get_separate_in_value(filoc, "M bin")["high_m"]
    else:
        print("Unknown mass bin: " + str(mass))
        return None
    if axis is None:
        fig = plt.gcf()
        axs = plt.gca()
    else:
        fig = plt.gcf()
        axs = axis
    blist = []
    point, slope, diff = ls.max_sep(filhmc, filhoc, "A", "Gini", e=e,rfull=blist)
    po_co = ls.error_poly_2(blist, 0.01, (-0.05, 0.6), (0.38, 0.64),rt=False)
    poly = Polygon(po_co, color="#ffb6b6ff")
    axs.add_patch(poly)
    vis.plot_correlation_filters((filhoc, filhmc), "A", "Gini", axis = axs)
    x = np.array([-0.05, 0.6])

    """
    axs.plot(
        x,
        0.14 * x + 0.8,
        linestyle="--",
        linewidth=2.5,
        c="black",
        label="early/late boundary",
    )
    """
    axs.plot(
        x,
        (point[0] - x) * (-slope) + point[1],
        linewidth=2.5,
        c="#e70000ff",
        label="line of best separation",
    )
    
    print("\n-----------------")
    print("Clumpy separation")
    print("-----------------")
    print("Best separation line:")
    print(f"Gini = A*({slope:.3f})+({point[1]-slope*point[0]:.3f})")
    print("Separation:")
    print(
        f"{100*(1-diff[1][0]):.1f} % bulges bellow, {(1-diff[1][1])*100:.1f} % non-bulges above"
    )
    print(f"{diff[2][0]*100:.1f} % bulge fraction, in the sample {diff[2][1]*100:.1f} %")
    print("-----------------\n")

    axs.set_xlabel("Asymmetry, $A$")
    axs.set_ylabel("Gini")
    axs.set_title("")  # Gini-M$_{20}$ statistics for non/bulges")
    axs.lines[0].set(
        c="#267ab46b", markeredgecolor="#00346899", markersize=10, alpha = None
    )
    axs.lines[1].set(
        c="#ff910c7c", markeredgecolor="#854900d5", markersize=13, alpha = None
    )
    L = axs.legend()
    L.texts[0].set_text("non-clumpy (" + L.texts[0]._text.split("(")[-1])
    L.texts[1].set_text("clumpy (" + L.texts[1]._text.split("(")[-1])
    axs.set_xlim(x)
    axs.set_ylim(0.38, 0.64)
    axs.tick_params(
        axis="both",
        which="major",
        bottom=True,
        top=False,
        left=True,
        right=False,
        direction="inout",
        size=6,
        labelsize=14,
    )
    if axs is None:
        fig.set_size_inches(5.5, 5)
        fig.set_layout_engine(layout="tight")
        if save is not None:
            fig.savefig(save)

def optimal_rfw(save = None):
    fullg = []
    # full = resu.get_rfw_range((0.5,1.37), full)
    for g in full:
        for i in range(len(g["filters"])):
            gm = {}
            for k in g:
                if type(g[k]) != list:
                    gm[k] = g[k]
                elif len(g[k]) == len(g["filters"]):
                    gm[k] = [g[k][i]]
            fullg.append(gm)
    for g in fullg:
        g["fileInfo"][0]["_wave"] = int(g["filters"][0][1:-1]) / 100
    for g in raw:
        g["frames"][0]["_wave"] = g["frames"][0]["_wavelength"]
    vis.plot_value_filters(fullg, "ZBEST", "_wave")
    fig = plt.gcf()
    axs = plt.gca()
    vis.plot_value_filters(raw, "ZBEST", "_wave", axis=axs)

    x = np.array([0.8, 2.5])
    axs.plot(
        x, (1 + x) * 1.37, linewidth=2.5, c="#e70000ff"
    )  # , label="$\lambda_{RFW}=1.37\mu$m")
    axs.plot(
        x,
        (1 + x) * 1.37 * 1.2,
        linewidth=2.5,
        c="grey",
        linestyle="--",
        label="20 \% tolerance",
    )
    axs.plot(x, (1 + x) * 1.37 * 0.8, linewidth=2.5, c="grey", linestyle="--")

    axs.set_xlabel("$z$", fontsize=18)
    axs.set_ylabel("Observed wavelength ($\mu$m)", fontsize=18)
    # axs.set_title("Optimal $\lambda_{RFW}$ for the JWST sample")
    """
    axs.lines[0].set(
        c="#0f5bb42d", markeredgecolor="#0f5bb42d", markersize=7, alpha=0.4
    )
    axs.lines[1].set(c="red", markeredgecolor="red", markersize=7, alpha=0.4)
    """
    axs.lines[0].set(
        c="#267ab44b", markeredgecolor="#0034685e", markersize=8, alpha = None
    )
    axs.lines[1].set(
        c="#ff910cab", markeredgecolor="#ba66017f", markersize=10, alpha = None
    )
    L = axs.legend(loc=8)
    L.texts[0].set_text("All available filters")
    L.texts[1].set_text("Filters closest to $\lambda_{\mathrm{RF}}$")
    # axs.set_xlim(x)
    axs.set_ylim(0, 5)
    ax2 = axs.twinx()
    ax2.set_ylim(0, 5)
    labels = [
        "F090W",
        "F115W",
        "F150W",
        "F182M",
        "F200W",
        "F210M",
        'F250M', #10
        "F277W",
        'F300M', #10
        "F335M",
        "F356W",
        "F410M",
        "F430M",
        "F444W",
        "F460M",
        "F480M",
    ]
    ticks = [int(n[1:-1]) / 100 for n in labels]
    ax2.set_yticks(ticks, labels=labels)
    axs.tick_params(
        axis="both",
        which="major",
        bottom=True,
        top=False,
        left=True,
        right=False,
        direction="inout",
        size=6,
        labelsize=14,
    )
    ax2.tick_params(
        axis="both",
        which="major",
        bottom=True,
        top=False,
        left=False,
        right=True,
        direction="inout",
        size=6,
        labelsize=10,
    )
    fig.set_size_inches(6, 5)
    fig.set_layout_engine(layout="tight")
    rot = axs.transData.transform_angles((np.arctan(1.37) * 180 / np.pi,), ((0, 0),))[0]
    axs.text(1.65, 3.42, "$\lambda_{\mathrm{RF}}=1.37\mu m$", rotation=rot, ha="center")
    if save is not None:
        fig.savefig(save)

def ren_et_al(save = None):
    vis.plot_grided(
        fil,
        ["C", "A", "Gini", "M20"],
        2,
        vis.points_multi_bins,
        title="Comparison to Ren et al. (2024) Figure 10.",
        include={0, 2},
        balpha=0.45,
    )
    fig = plt.gcf()
    axes = fig.axes
    axes[0].plot(ren_cr[0], shiftsi[0](ren_cr[1]), c="blue", linestyle=":")
    axes[1].plot(ren_cm[0], shiftsi[0](ren_cm[1]), c="blue", linestyle=":")
    axes[2].plot(ren_ar[0], shiftsi[1](ren_ar[1]), c="blue", linestyle=":")
    axes[3].plot(ren_am[0], shiftsi[1](ren_am[1]), c="blue", linestyle=":")
    axes[4].plot(
        ren_gr[0], shiftsi[2](ren_gr[1]), c="blue", linestyle=":", label="Ren et al."
    )
    axes[5].plot(ren_gm[0], shiftsi[2](ren_gm[1]), c="blue", linestyle=":")
    axes[6].plot(ren_mr[0], shiftsi[3](ren_mr[1]), c="blue", linestyle=":")
    axes[7].plot(ren_mm[0], shiftsi[3](ren_mm[1]), c="blue", linestyle=":")
    """
    axes[0].plot(ren_cr[0], ren_cr[1], c="blue", linestyle="--")
    axes[1].plot(ren_cm[0], ren_cm[1], c="blue", linestyle="--")
    axes[2].plot(ren_ar[0], ren_ar[1], c="blue", linestyle="--")
    axes[3].plot(ren_am[0], ren_am[1], c="blue", linestyle="--")
    axes[4].plot(ren_gr[0], ren_gr[1], c="blue", linestyle="--")
    axes[5].plot(ren_gm[0], ren_gm[1], c="blue", linestyle="--")
    axes[6].plot(ren_mr[0], ren_mr[1], c="blue", linestyle="--", label="Ren et al.")
    axes[7].plot(ren_mm[0], ren_mm[1], c="blue", linestyle="--")
    """
    axes[0].errorbar(
        yao_c[0],
        yao_c[1],
        yerr=yao_c[2],
        fmt="",
        c="black",
        capsize=4,
        linestyle="",
        alpha=0.5,
    )
    axes[2].errorbar(
        yao_a[0],
        yao_a[1],
        yerr=yao_a[2],
        fmt="",
        c="black",
        capsize=4,
        linestyle="",
        alpha=0.5,
    )
    axes[4].errorbar(
        yao_g[0],
        yao_g[1],
        yerr=yao_g[2],
        fmt="",
        c="black",
        capsize=4,
        linestyle="",
        alpha=0.5,
    )
    axes[6].errorbar(
        yao_m[0],
        yao_m[1],
        yerr=yao_m[2],
        fmt="",
        c="black",
        capsize=4,
        linestyle="",
        alpha=0.5,
    )
    axes[0].plot(
        yao_c[0],
        yao_c[1],
        mec="darkgreen",
        mfc="green",
        marker="D",
        markersize=5,
        c="darkgreen",
        linestyle="",
        linewidth=1.3,
        alpha = 0.6
    )
    axes[2].plot(
        yao_a[0],
        yao_a[1],
        mec="darkgreen",
        mfc="green",
        marker="D",
        markersize=5,
        c="darkgreen",
        linestyle="",
        linewidth=1.3,
        alpha = 0.6
    )
    axes[4].plot(
        yao_g[0],
        yao_g[1],
        mec="darkgreen",
        mfc="green",
        marker="D",
        markersize=5,
        c="darkgreen",
        linestyle="",
        linewidth=1.3,
        label="Yao et al.",
        alpha = 0.6    
        )
    axes[6].plot(
        yao_m[0],
        yao_m[1],
        mec="darkgreen",
        mfc="green",
        marker="D",
        markersize=5,
        c="darkgreen",
        linestyle="",
        linewidth=1.3,
        alpha = 0.6 
    )
    """
    axes[0].plot(
        yao_c[0], yao_c[1], c="darkgreen", linestyle="dotted", linewidth=1.3, alpha=0.8
    )
    axes[2].plot(
        yao_a[0], yao_a[1], c="darkgreen", linestyle="dotted", linewidth=1.3, alpha=0.8
    )
    axes[4].plot(
        yao_g[0], yao_g[1], c="darkgreen", linestyle="dotted", linewidth=1.3, alpha=0.8
    )
    axes[6].plot(
        yao_m[0], yao_m[1], c="darkgreen", linestyle="dotted", linewidth=1.3, alpha=0.8
    )
    """
    """
    for i in range(8):
        yl = axes[i].lines[0].get_ydata()
        yu = axes[i].lines[1].get_ydata()
        x = axes[i].lines[4].get_xdata()
        y = axes[i].lines[4].get_ydata()
        axes[i].errorbar(x,shifts[i//2](y), yerr=(np.abs(shifts[i//2](y)-shifts[i//2](yl)),np.abs(shifts[i//2](yu)-shifts[i//2](y))), c= "gray", capsize = 5, alpha = 0.8)
    """

    axes[0].set_ylabel("Concentration, $C$")
    axes[2].set_ylabel("Asymmetry, $A$")
    axes[6].set_ylabel("$M_{20}$")
    axes[0].set_ylim(2.25, 4)
    axes[2].set_ylim(0, 0.35)
    axes[4].set_ylim(0.425, 0.6)
    axes[6].set_ylim(-2.2, -1.25)
    axes[0].set_ylim(2.45, 3.5)
    axes[2].set_ylim(0.02, 0.3)
    axes[4].set_ylim(0.43, 0.57)
    axes[6].set_ylim(-2, -1.52)
    axes[6].set_xlim(0.6, 2.7)
    axes[7].set_xlim(9.85, 11.45)

    for ax in axes:
        ax.tick_params(
            axis="both",
            which="major",
            bottom=True,
            top=False,
            left=True,
            right=False,
            direction="inout",
            size=6,
            labelsize=14,
        )

    for i in range(8):
        axes[i].lines[4].set(linestyle="", linewidth=1.5)
        axes[i].lines[5].set(linestyle=":", linewidth=2.5)
        axes[i].lines[6].set(linewidth=2.5)
        if i % 2 == 0:
            axes[i].lines[10].set(linewidth=2.5)
        #if i % 2 == 1:
        #    axes[i].lines[4].set(marker="")

    axes[4].lines[4].set_label("_mean")
    axes[4].lines[5].set_label("_median")
    L = axes[4].legend(loc=3, fontsize=12)
    L.texts[0].set_text("Ren et al. (2024)")
    L.texts[1].set_text("Yao et al. (2023)")
    for i in range(4, len(L.texts)):
        L.texts[i].set_text("_")
    L2 = axes[6].legend(loc=2, fontsize=12)
    L2.texts[0].set_text("mean")
    L2.texts[1].set_text("median")
    for i in range(2, len(L2.texts)):
        L2.texts[i].set_text("_")

    fig.set_size_inches(5.7, 9)
    fig.suptitle("")
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)

def MID_classification(save = None):
    vis.plot_grided(
        fil,
        ["M", "I", "D"],
        2,
        vis.points_multi_clas,
        title="MID statistics for different morphology classifications",
        new_names={"bulges": "Bulge", "mergers": "Interacting-Merger"},
    )
    fig = plt.gcf()
    axes = fig.axes
    axes[0].set_ylim(-0.13, 0.6)
    axes[2].set_ylim(-0.13, 0.8)
    axes[4].set_ylim(-0.04, 0.4)
    axes[4].set_xticks([0, 1], ["without bulge", "with bulge"])
    axes[5].set_xticks([0, 1], ["others", "interact./mergers"])
    axes[0].set(ylabel="Multinode, $M$")
    axes[2].set(ylabel="Intensity, $I$")
    axes[4].set(ylabel="Deviation, $D$")
    for ax in axes:
        ax.tick_params(
            axis="both",
            which="major",
            bottom=True,
            top=False,
            left=True,
            right=False,
            direction="inout",
            size=6,
            labelsize=14,
        )
    for i in range(len(axes[4].patches)):
        axes[4].patches[i].set_label("_")
    #for i in range(len(axes)):
    #    axes[i].lines[4].set(linestyle= "")
    #    axes[i].lines[6].set(linestyle=":", linewidth = 2)
    L = axes[4].legend(loc=1)
    L.texts[0].set_text("mean")
    L.texts[1].set_text("median")
    fig.set_size_inches(6.3, 8)
    fig.suptitle("")
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)
        
def MID_classification_nb(save = None):
    vis.plot_grided(
        fil,
        ["M", "I", "D"],
        1,
        vis.points_multi_clas,
        title="MID statistics for different morphology classifications",
        new_names={"mergers": "Interacting-Merger"},
        percentiles = (16,84),
    )
    fig = plt.gcf()
    axes = fig.axes
    axes[0].set_ylim(-0.13, 0.6)
    axes[1].set_ylim(-0.13, 0.8)
    axes[2].set_ylim(-0.04, 0.4)
    axes[2].set_xlim(-0.3, 1.3)
    axes[2].set_xticks([0, 1], ["others", "interact./mergers"])
    axes[0].set(ylabel="Multinode, $M$")
    axes[1].set(ylabel="Intensity, $I$")
    axes[2].set(ylabel="Deviation, $D$")
    for ax in axes:
        ax.tick_params(
            axis="both",
            which="major",
            bottom=True,
            top=False,
            left=True,
            right=False,
            direction="inout",
            size=6,
            labelsize=14,
        )
    for i in range(len(axes[2].patches)):
        axes[2].patches[i].set_label("_")
    #for i in range(len(axes)):
    #    axes[i].lines[4].set(linestyle= "")
    #    axes[i].lines[6].set(linestyle=":", linewidth = 2)
    L = axes[2].legend(loc=2)
    L.texts[0].set_text("mean")
    L.texts[1].set_text("median")
    fig.set_size_inches(4.5, 8)
    fig.suptitle("")
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)


def sersic_comparison(f150 = 0, save = None, n_mult = 1, new = 0):
    if not f150:
        ser = resu.galaxy_pruning(fils, strength="sersic")
        ylab = "\\textit{JWST} (F444W)"
    elif f150 == 1:
        ser = resu.galaxy_pruning(fils150, strength="sersic")
        ylab = "\\textit{JWST} (F150W)"
    elif f150 == 2:
        ser = resu.galaxy_pruning(fils150hst, strength="sersic")
        ylab = "\\textit{JWST} (F150W, HST PSF)"
    elif f150 == 3:
        ser = resu.galaxy_pruning(filshst, strength="sersic")
        ylab = "\\textit{HST} (F160W)"
    elif f150 == 4:
        ser = resu.galaxy_pruning(fils150sc, strength="sersic")
        ylab = "\\textit{JWST} (F150W, scaled 90 \%)"
    elif f150 == 5:
        ser = resu.galaxy_pruning(fils150sc3, strength="sersic")
        ylab = "\\textit{JWST} (F150W, scaled 300 \%)"
    elif f150 == 6:
        ser = resu.galaxy_pruning(fils150sc30, strength="sersic")
        ylab = "\\textit{JWST} (F150W, scaled 30 \%)"
    if new == 1:
        xlab = "\\textit{HST} (F160W)"
    elif new == 0:  
        xlab = "\\textit{HST} (F160W, CANDELS)"
    elif new == 2:
        xlab = "\\textit{HST} (F160W, CANDELS VDW12)"
    for g in ser+new*filshst:
        g["frames"][0]["_sersic_rhalf"] = g["frames"][0]["sersic_rhalf"] / 40
        g["info"]["_H_Q"] = 1 - g["info"]["H_Q"]
        g["info"]["_H_PA"] = (g["info"]["H_PA"] + 90) / 180 * np.pi
        g["info"]["_H_NSERSIC"] = g["info"]["H_NSERSIC"] / n_mult
        g["info"]["_VDW12_H_Q"] = 1 - g["info"]["VDW12_H_Q"]
        g["info"]["_VDW12_H_PA"] = (g["info"]["VDW12_H_PA"] + 90) / 180 * np.pi
        g["info"]["_VDW12_H_N"] = g["info"]["VDW12_H_N"] / n_mult
    fig, axes = plt.subplots(1, 3)  # 4)
    if new == 0:
        vis.plot_value_filters(ser, "_H_NSERSIC", "sersic_n", axis=axes[0])
        vis.plot_value_filters(ser, "H_RE", "_sersic_rhalf", axis=axes[1])
        vis.plot_value_filters(ser, "_H_Q", "sersic_ellip", axis=axes[2])
        # vis.plot_value_filters(ser, "_H_PA", "sersic_theta", axis=axes[3])
    elif new == 1:
        vis.plot_values_2s(filshst, ser, "sersic_n", "sersic_n", axis=axes[0])
        vis.plot_values_2s(filshst, ser, "_sersic_rhalf", "_sersic_rhalf", axis=axes[1])
        vis.plot_values_2s(filshst, ser, "sersic_ellip", "sersic_ellip", axis=axes[2])
        # vis.plot_value_filters(filshst, ser, "sersic_theta", "sersic_theta", axis=axes[3])
    elif new == 2:
        vis.plot_value_filters(ser, "_VDW12_H_N", "sersic_n", axis=axes[0])
        vis.plot_value_filters(ser, "VDW12_H_RE", "_sersic_rhalf", axis=axes[1])
        vis.plot_value_filters(ser, "_VDW12_H_Q", "sersic_ellip", axis=axes[2])
        # vis.plot_value_filters(ser, "_H_PA", "sersic_theta", axis=axes[3])
    print(f"Number of galaxies in the  sersic plot: {len(axes[0].lines[0].get_data()[0])}")
    axes[0].plot([0, 5], [0, 5], c="#e70000ff", linestyle="--", linewidth=2)
    axes[1].plot([0, 40], [0, 40], c="#e70000ff", linestyle="--", linewidth=2)
    axes[2].plot([0, 1], [0, 1], c="#e70000ff", linestyle="--", linewidth=2)
    # axes[3].plot([0, np.pi], [0, np.pi], c="#e70000ff", linestyle="--", linewidth=2)
    for i in range(3):  # 4):
        axes[i].lines[0].set(
            markerfacecolor="#267ab46b", markeredgecolor="#00346899", alpha=None, markersize = 8
        )
        axes[i].tick_params(
            axis="both",
            which="major",
            bottom=True,
            top=False,
            left=True,
            right=False,
            direction="inout",
            size=6,
            labelsize=14,
        )

    axes[0].set(
        title="",  # Sersic n",
        xlabel="$n$ "+xlab,
        ylabel="$n$ "+ylab,
        xlim=(0, 5),
        ylim=(0, 5),
    )
    if new == 1:
        axes[0].set(xlim=(0,3.5),ylim = (0,3.5))
    axes[1].set(
        title="",  # Sersic radius",
        xlabel="$r_\mathrm{eff}$ (arcsec) "+xlab,
        ylabel="$r_\mathrm{eff}$ (arcsec) "+ylab,
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[2].set(
        title="",  # Sersic ellipticity",
        xlabel="$e$ "+xlab,
        ylabel="$e$ "+ylab,
        xlim=(0, 1),
        ylim=(0, 1),
    )
    """
    axes[3].set(
        title="", #Sersic position angle",
        xlabel=r"$\theta_\mathrm{PA}$ "+xlab,
        ylabel=r"$\theta_\mathrm{PA}$ "+ylab,
        xlim=(0, np.pi),
        ylim=(0, np.pi),
    )
    """
    fig.suptitle(
        ""  # Sersic fit comparison: HST (F160W) vs JWST ($\lambda_{RFW}=1.37\mu$m)"
    )
    fig.set_size_inches(13.3, 4.5)  # 17.5, 4.5)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)

def ginim20_class(save=None):
    vis.plot_grided(
        fil,
        ["S(G, M20)", "F(G, M20)"],
        2,
        vis.points_multi_clas,
        title="Standard morphological classification indicators",
        new_names={"bulges": "Bulge", "mergers": "Interacting-Merger"},
    )
    fig = plt.gcf()
    axes = fig.axes
    axes[0].set_ylim(-0.18, 0.1)
    axes[2].set_ylim(-1.3, 0.8)
    axes[2].set_xticks([0, 1], ["without bulge", "with bulge"])
    axes[3].set_xticks([0, 1], ["others", "interact./mergers"])
    axes[0].set_ylabel("Merger st. $S(G, M_{20}$)")
    axes[2].set_ylabel("Bulge st. $F(G, M_{20}$)")
    axes[0].patch.set_alpha(0)
    axes[3].patch.set_alpha(0)
    for i in {"top", "bottom", "left", "right"}:
        axes[1].spines[i].set(color="#b70101", linewidth=3.5, linestyle=":")
        axes[2].spines[i].set(color="#b70101", linewidth=3.5, linestyle=":")
    axes[0].spines["right"].set(linestyle="")
    axes[0].spines["bottom"].set(linestyle="")
    axes[3].spines["left"].set(linestyle="")
    axes[3].spines["top"].set(linestyle="")
    for ax in axes:
        ax.tick_params(
            axis="both",
            which="major",
            bottom=True,
            top=False,
            left=True,
            right=False,
            direction="inout",
            size=6,
            labelsize=14,
        )
    for i in range(len(axes[0].patches)):
        axes[0].patches[i].set_label("_")
    L = axes[0].legend(loc=2)
    L.texts[0].set_text("mean")
    L.texts[1].set_text("median")
    fig.set_size_inches(6.3, 6)
    fig.suptitle("")
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)

def ginim20_class_2(save=None):
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2)
    axes = gs.subplots()
    vis.points_multi_clas(fil, "F(G, M20)", axis = [axes[1]], no_legend = True, new_names = {"bulges":  "Bulge"})
    vis.points_multi_clas(fil, "S(G, M20)", axis = [axes[0]], no_legend = True, new_names = {"mergers": "Interacting-Merger"})
    axes[1].set_ylim(-1.3, 0.8)
    axes[0].set_ylim(-0.18, 0.1)
    axes[1].set_xticks([0, 1], ["without bulge", "with bulge"])
    axes[0].set_xticks([0, 1], ["others", "interact./mergers"])
    axes[1].set_xlim(-0.3, 1.3)
    axes[0].set_xlim(-0.3, 1.3)
    axes[1].set_ylabel("")#Bulge stat. $F(G, M_{20}$)")
    axes[0].set_ylabel("")#Merger stat. $S(G, M_{20}$)")
    axes[1].set_title("$F(G, M_{20})$")
    axes[0].set_title("$S(G, M_{20})$")    
    for ax in axes:
        ax.tick_params(
            axis="both",
            which="major",
            bottom=True,
            top=False,
            left=True,
            right=False,
            direction="inout",
            size=6,
            labelsize=14,
        )
    for i in range(len(axes[0].patches)):
        axes[0].patches[i].set_label("_")
    L = axes[0].legend(loc=2)
    L.texts[0].set_text("mean")
    L.texts[1].set_text("median")
    fig.set_size_inches(7, 3.6)
    fig.suptitle("")
    #fig.set_layout_engine(layout="tight")
    plt.subplots_adjust(top = 0.87, bottom = 0.12, left=0.115, right=0.955, wspace=0.28)
    if save is not None:
        fig.savefig(save)


def separation_table(gal1, gal2, parameters):
    """Generates a table with separation values in various 2d parameter
    spaces formatted as latex `tabular` environment with coloured cells. The
    sets of galaxies between which the separation is to be made are the `gal1`
    and `gal2` arguments, while the `parameters` argument defines the
    parameter spaces to be considered.
    """
    print("This might take some time...")
    vals = ls.best_parameters(gal1, gal2, parameters)
    vald = {k: vals[k][2] for k in vals}
    vmax = max([vald[k] for k in vald])
    vmin = min([vald[k] for k in vald])
    cmap = lambda c: int((c - vmin) / (vmax - vmin) * 70)
    value = "\hline \hline & " + " & ".join(parameters) + "\\\\\hline\n"
    values = []
    for nr in parameters:
        row = []
        for nc in parameters:
            val = 0
            for k in vald:
                if set([nc, nr]) == set(k):
                    val = vald[k]
            sval = f"\cellcolor{{red!{cmap(val)}}} {val:.3f}"
            if parameters.index(nc) >= parameters.index(nr):
                row.append(sval)
            else:
                row.append("")
        values.append(" & ".join([nr] + row))
    table = value + " \\\\\n".join(values) + "\\\\\hline\n"
    table = table.replace("sersic_n", "$n$")
    table = table.replace("sersic_rhalf", "$r_{\\text{eff}}$")
    table = table.replace("M20", "$\\text{M}_{20}$")
    starts = f"\\begin{{tabular}}{{{"c"*(len(parameters)+1)}}}\n"
    ends = "\end{tabular}"
    table = starts + table + ends
    print("\n----------\n")
    print(table)
    print("\n----------\n")


def masking_examples_6(galaxies, save = None):
    fig = plt.figure(figsize=(7.77, 5.7))
    gs = fig.add_gridspec(
        2, 3, wspace=0.065, hspace=0.175, left=0.03, right=0.97, top=0.925, bottom=0.045
    )
    axs = gs.subplots().flatten()
    galaxies = galaxies[:6]
    gal_in = [resu.get_galaxy_entry(full, g) for g in galaxies]
    gal_ps = resu.get_optim_rfw(gal_in, fixed_rfw=1.33)
    gal_data = run.galaxies_data(gal_ps, return_object=True)

    for i in range(len(gal_data)):
        plot_segmentation(gal_data[i], axis=axs[i])
        axs[i].set_title(galaxies[i])
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    axs[3].plot([], [], linestyle="--", c="red", linewidth=1.7, label="Masked area")
    axs[3].plot([], [], linestyle="--", c="blue", linewidth=1.7, label="Target area")
    L = axs[3].legend(loc=1)
    fig.suptitle("")  # Examples of target identification and masking")
    if save is not None:
        fig.savefig(save)


def masking_examples_4(galaxies, save = None):
    fig = plt.figure(figsize=(10.35, 2.95))
    gs = fig.add_gridspec(
        1,
        4,
        wspace=0.065,
        hspace=0.175,
        left=0.015,
        right=0.985,
        top=0.88,
        bottom=0.045,
    )
    axs = gs.subplots().flatten()
    galaxies = galaxies[:4]
    gal_in = [resu.get_galaxy_entry(full, g) for g in galaxies]
    gal_ps = resu.get_optim_rfw(gal_in, fixed_rfw=1.33)
    gal_data = run.galaxies_data(gal_ps, return_object=True)

    for i in range(len(gal_data)):
        plot_segmentation(gal_data[i], axis=axs[i])
        axs[i].set_title(galaxies[i])
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].plot([], [], linestyle="--", c="red", linewidth=1.7, label="Masked area")
    axs[0].plot([], [], linestyle="--", c="blue", linewidth=1.7, label="Target area")
    L = axs[0].legend(loc=4)
    fig.suptitle("")  # Examples of target identification and masking")
    if save is not None:
        fig.savefig(save)


def plot_segmentation(g, axis=None):
    if axis is None:
        axis = plt.gca()
    f = g.frames[0]
    l = np.log(f.convolved)
    ln = np.nan_to_num(np.log(f.data), nan=-100)
    r = np.nanmax(l) - np.nanmin(l)
    axis.imshow(ln, cmap="gray", vmin=np.nanmin(l) + r / 2, vmax=np.nanmax(l))
    axis.contour(f.target, levels=1, colors="blue", linestyles="--", linewidths=1.7)
    axis.contour(f.mask, levels=1, colors="red", linestyles="--", linewidths=1.7)


def sersic_filters(save = None):
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, hspace=0)
    axes = gs.subplots(sharex=True)
    fullh = filsfull
    filters = ["F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F444W"]
    nf = {f:float(f[1:-1])/100 for f in filters}
    sernf = dict()
    serrf = dict()
    for f in filters:
        valsnraw = [resu.get_filter_or_avg(g, "sersic_n", filt = f) for g in fullh]
        valuesn = np.array([v for v in valsnraw if v is not None])
        valsrraw = [resu.get_filter_or_avg(g, "sersic_rhalf", filt = f) for g in fullh]
        valuesr = np.array([v for v in valsrraw if v is not None])/40
        
        n_bootstrap = 1000  # Number of bootstrap samples
        # Sérsic index
        medians_n = [np.nanmedian(np.random.choice(valuesn, size=len(valuesn), replace=True))
                            for _ in range(n_bootstrap)]
        medians_r = [np.nanmedian(np.random.choice(valuesr, size=len(valuesr), replace=True))
                            for _ in range(n_bootstrap)]

        sernf[f] = [np.percentile(medians_n, v) for v in [50,16,84]]
        serrf[f] = [np.percentile(medians_r, v) for v in [50,33,67]]
        
    axes[0].errorbar([nf[f] for f in filters], [sernf[f][0] for f in filters], [[sernf[f][0]-sernf[f][1] for f in filters],[sernf[f][2]-sernf[f][0] for f in filters]],ecolor="black", capsize = 5, linestyle = "")
    axes[0].plot([nf[f] for f in filters], [sernf[f][0] for f in filters])
    
    
    axes[1].errorbar([nf[f] for f in filters], [serrf[f][0] for f in filters], [[serrf[f][0]-serrf[f][1] for f in filters],[serrf[f][2]-serrf[f][0] for f in filters]],ecolor="black", capsize = 5, linestyle="")
    axes[1].plot([nf[f] for f in filters], [serrf[f][0] for f in filters])

    axes[0].lines[3].set(linestyle=":", c= "black", marker = "o", markersize = 10, markerfacecolor= "#4b92c3")
    axes[1].lines[3].set(linestyle=":", c= "black", marker = "o", markersize = 10, markerfacecolor= "#4b92c3")

    axes[1].set(xlabel="$\lambda$ ($\mu m$)",ylabel="$\left<r_{\mathrm{eff}}\\right>$ (arcsec)")
    axes[0].set(ylabel="$\left<n\\right>$")
    for f in filters:
        axes[0].annotate(f,(nf[f],sernf[f][0]),ha='left',textcoords="offset points", size = 10,xytext=(7,-10))
        axes[1].annotate(f,(nf[f],serrf[f][0]),ha='left',textcoords="offset points", size = 10,xytext=(7,0))
    
    axes[0].texts[-1].set(ha='right',position=(-4,5))
    axes[0].texts[-2].set(position=(7,-13))
    axes[1].texts[-1].set(ha='right',position=(-6,-10))

    fig.set_size_inches(4.5, 6.5)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)

def combined_separation(save = None):
    fig = plt.figure()
    gs = fig.add_gridspec(1, 3, wspace=0)
    axes = gs.subplots(sharey=True)
    bulges_separation(mass = None, axis = axes[0], e = 0.01)
    mergers_separation(mass = None, axis = axes[1], e = 0.01)
    clumpy_separation(mass = None, axis = axes[2])
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    fig.set_size_inches(14, 5)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)

if __name__ == "__main__":
    """Comment out lines corresponding to plots you want to create."""
    parameters = [
        "Gini",
        "M20",
        "C",
        "A",
        "S",
        "M",
        "I",
        "D",
        "sersic_n",
        "sersic_rhalf",
    ]
    # separation_table(filmm, filom, parameters)
    # separation_table(filmb, filob, parameters)
    # separation_table(filmc, filoc, parameters)
    # mergers_separation(mass = None, save = "../../out/pdfs/josef_mergers.pdf")
    # mergers_separation_a(mass = None, save = "../../out/pdfs/josef_mergers_a.pdf")    
    # bulges_separation(mass = None, save = "../../out/pdfs/josef_bulges.pdf")
    # clumpy_separation(mass = None, save = "../../out/pdfs/josef_clumpy.pdf")
    # mergers_separation(mass = None, save = "../../out/pdfs/josef_mergers_frac.pdf", e = 0.01)
    # mergers_separation_a(mass = None, save = "../../out/pdfs/josef_mergers_a_frac.pdf", e = 0.005)    
    # bulges_separation(mass = None, save = "../../out/pdfs/josef_bulges_frac.pdf", e = 0.01)
    # optimal_rfw(save = "../../out/pdfs/josef_rfw.pdf")
    # ren_et_al(save = "../../out/pdfs/josef_Ren.pdf")
    # MID_classification(save = "../../out/pdfs/josef_MID.pdf")
    # MID_classification_nb(save = "../../out/pdfs/josef_MID_2.pdf")
    # sersic_comparison(save = "../../out/pdfs/josef_sersic.pdf", new = 1)
    # sersic_comparison(f150= 1, save = "../../out/pdfs/josef_sersic_150.pdf", new = 1)
    # sersic_comparison(f150=1, save = "../../out/pdfs/josef_sersic_candels")
    # ginim20_class(save = "../../out/pdfs/josef_class.pdf")
    # ginim20_class_2(save = "../../out/pdfs/josef_class_2.pdf")
    # sersic_filters(save = "../../out/pdfs/josef_sersic_wavelength.pdf")
    # combined_separation(save = "../../out/pdfs/josef_combined_separation.pdf")

    names = [
        "COS4_02049",
        "COS4_02167",
        "U4_26324",
        "U4_21440",
    ]
    #masking_examples_4(names, save = "../../out/pdfs/josef_masks.pdf")
    #mergers_nstatistics(mass = None, save = "../../out/pdfs feedback/josef_mergers_distance.pdf")
    #mergers_nstatistics(mass = None, save = "../../out/pdfs feedback/pngs/josef_mergers_distance.png")
    #bulges_nstatistics(mass = None, save = "../../out/pdfs feedback/josef_bulges_distance.pdf")
    #bulges_nstatistics(mass = None, save = "../../out/pdfs feedback/pngs/josef_bulges_distance.png")
    #sersic_comparison(f150=True, save = "../../out/pdfs feedback/josef_sersic_150.pdf")
    #sersic_comparison(f150=1, save = "../../out/pdfs feedback/pngs/josef_sersic_150.png")
    #sersic_comparison(f150=2, save = "../../out/pdfs feedback/josef_sersic_150hst.pdf")
    #sersic_comparison(f150=2, save = "../../out/pdfs feedback/pngs/josef_sersic_150_hst.png", n_mult = 1)
    names = [
        "COS4_02049",
        "COS4_02167",
        "COS4_17389",
        "COS4_20910",
        "U4_26324",
        "U4_21440",
    ]
    #masking_examples_6(names, save = "../../out/supl/josef_masks_6.png")
    #mergers_separation(mass = None, save = "../../out/supl/josef_mergers_frac.png", e = 0.01)
    #mergers_separation_a(mass = None, save = "../../out/supl/josef_mergers_frac_a.png", e = 0.005)    
    #bulges_separation(mass = None, save = "../../out/supl/josef_bulges_frac.png", e = 0.01)
    #sersic_comparison(f150=2, save = "../../out/supl/josef_sersic_150hst_n.png", new = True)
    #sersic_comparison(f150=3, save = "../../out/supl/josef_sersic_160_hst.png")
    plt.show()
