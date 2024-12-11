import json

import matplotlib.pyplot as plt
import numpy as np

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

raw = run.fetch_json("dict_out/out_full_matched_5_m.json")["galaxies"]
bulges = run.fetch_json("dict_in/n4_Bulge.json")["galaxies"]
mergers = run.fetch_json("dict_in/n4_Interacting-Merger.json")["galaxies"]
sers = run.fetch_json("dict_out/out_full_444.json")["galaxies"]
full = run.fetch_json("dict_in/dictionary_full.json")["galaxies"]

fil = resu.galaxy_pruning(raw)
filmb = resu.get_subset(fil, bulges)
filob = resu.get_complement(fil, bulges)
filmm = resu.get_subset(fil, mergers)
filom = resu.get_complement(fil, mergers)
fils = resu.galaxy_pruning(sers, strength="strict")


def mergers_separation():
    point, slope, diff = ls.max_sep(filmm, filom, "M20", "Gini")
    vis.plot_correlation_filters((filom, filmm), "M20", "Gini")
    x = np.array([-2.3, -0.8])
    fig = plt.gcf()
    axs = plt.gca()

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
        c="gold",
        label="line of best separation",
    )
    print("\n------------------")
    print("Mergers separation")
    print("------------------")
    print("Line used in literature:")
    print(f"Gini = M_20*({-0.14:.3f})+({0.33:.3f})")
    print("Separation:")
    mmabove = ls.get_above_line(filmm, "M20", "Gini", [0, 0.33], -0.14)
    omabove = ls.get_above_line(filom, "M20", "Gini", [0, 0.33], -0.14)
    print(
        f"{100*mmabove:.1f} % mergers above, {(1-omabove)*100:.1f} % non-mergers bellow"
    )
    print("Best separation line:")
    print(f"Gini = M_20*({slope:.3f})+({point[1]-slope*point[0]:.3f})")
    print("Separation:")
    print(
        f"{100*diff[1][0]:.1f} % mergers above, {diff[1][1]*100:.1f} % non-mergers bellow"
    )
    print("------------------\n")

    axs.set_xlabel("M$_{20}$")
    axs.set_ylabel("Gini")
    axs.set_title("")  # Gini-M$_{20}$ statistics for non/mergers")
    axs.lines[0].set(
        c="#034a8360", markeredgecolor="#05396f9d", markersize=7, alpha=0.5
    )
    axs.lines[1].set(
        c="#b7010154", markeredgecolor="#b701018b", markersize=7, alpha=0.5
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
    fig.set_size_inches(5.5, 5)
    fig.set_layout_engine(layout="tight")


def mergers_separation_a():
    point, slope, diff = ls.max_sep(filmm, filom, "A", "A")
    vis.plot_ref_hist(
        (filom, filmm), "A", pdf=True, bins=13, names=["non-mergers", "mergers"]
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
        label="Lotz et al. (2008)",
    )
    axs.plot(
        [x, x],
        [0, 7],
        linewidth=2.5,
        c="gold",
        label="line of best separation",
    )
    print("\n------------------")
    print("Mergers separation")
    print("------------------")
    print("Line used in literature:")
    print(f"A = 0.35")
    print("Separation:")
    mmabove = ls.get_above_line(filmm, "A", "A", [0.35, 0.35], 2)
    omabove = ls.get_above_line(filom, "A", "A", [0.35, 0.35], 2)
    print(
        f"{100*(1-mmabove):.1f} % mergers on right, {omabove*100:.1f} % non-mergers on left"
    )
    print("Best separation line:")
    print(f"A = {x:.3f}")
    print("Separation:")
    print(
        f"{100*(1-diff[1][0]):.1f} % mergers on right, {(1-diff[1][1])*100:.1f} % non-mergers on left"
    )
    print("------------------\n")

    axs.set_xlabel("Asymmetry statistics")
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


def bulges_separation():
    point, slope, diff = ls.max_sep(filmb, filob, "M20", "Gini")
    vis.plot_correlation_filters((filob, filmb), "M20", "Gini")
    x = np.array([-2.3, -0.8])
    fig = plt.gcf()
    axs = plt.gca()

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
        c="gold",
        label="line of best separation",
    )
    print("\n-----------------")
    print("Bulges separation")
    print("-----------------")
    print("Line used in literature:")
    print(f"Gini = M_20*({0.14:.3f})+({0.8:.3f})")
    print("Separation:")
    mbabove = ls.get_above_line(filmm, "M20", "Gini", [0, 0.8], 0.14)
    obabove = ls.get_above_line(filom, "M20", "Gini", [0, 0.8], 0.14)
    print(
        f"{100*mbabove:.1f} % bulges bellow, {(1-obabove)*100:.1f} % non-bulges above"
    )
    print("Best separation line:")
    print(f"Gini = M_20*({slope:.3f})+({point[1]-slope*point[0]:.3f})")
    print("Separation:")
    print(
        f"{100*diff[1][0]:.1f} % bulges bellow, {diff[1][1]*100:.1f} % non-bulges above"
    )
    print("-----------------\n")

    axs.set_xlabel("M$_{20}$")
    axs.set_ylabel("Gini")
    axs.set_title("")  # Gini-M$_{20}$ statistics for non/bulges")
    axs.lines[0].set(
        c="#034a8360", markeredgecolor="#05396f9d", markersize=7, alpha=0.5
    )
    axs.lines[1].set(
        c="#b7010154", markeredgecolor="#b701018b", markersize=7, alpha=0.5
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
    fig.set_size_inches(5.5, 5)
    fig.set_layout_engine(layout="tight")


def optimal_rfw():
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
        x, (1 + x) * 1.37, linewidth=2.5, c="green"
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
    axs.set_ylabel("observed wavelength ($\mu$m)", fontsize=18)
    # axs.set_title("Optimal $\lambda_{RFW}$ for the JWST sample")
    axs.lines[0].set(
        c="#0f5bb42d", markeredgecolor="#0f5bb42d", markersize=7, alpha=0.4
    )
    axs.lines[1].set(c="red", markeredgecolor="red", markersize=7, alpha=0.4)
    L = axs.legend(loc=8)
    L.texts[0].set_text("All available filters")
    L.texts[1].set_text("Selected filters")
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
        "F277W",
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


def ren_et_al():
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
        markersize=4,
        c="darkgreen",
        linestyle="dotted",
        linewidth=1.3,
    )
    axes[2].plot(
        yao_a[0],
        yao_a[1],
        mec="darkgreen",
        mfc="green",
        marker="D",
        markersize=4,
        c="darkgreen",
        linestyle="dotted",
        linewidth=1.3,
    )
    axes[4].plot(
        yao_g[0],
        yao_g[1],
        mec="darkgreen",
        mfc="green",
        marker="D",
        markersize=4,
        c="darkgreen",
        linestyle="dotted",
        linewidth=1.3,
        label="Yao et al.",
    )
    axes[6].plot(
        yao_m[0],
        yao_m[1],
        mec="darkgreen",
        mfc="green",
        marker="D",
        markersize=4,
        c="darkgreen",
        linestyle="dotted",
        linewidth=1.3,
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
        axes[i].lines[4].set(linestyle="--", linewidth=1.5)
        axes[i].lines[6].set(linewidth=2)
        if i % 2 == 0:
            axes[i].lines[10].set(linewidth=2)

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


def MID_classification():
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
    L = axes[4].legend(loc=1)
    L.texts[0].set_text("mean")
    L.texts[1].set_text("median")
    fig.set_size_inches(6.3, 8)
    fig.suptitle("")
    fig.set_layout_engine(layout="tight")


def sersic_comparison():
    ser = resu.galaxy_pruning(fils, strength="sersic")
    print(f"Number of galaxies in the  sersic plot: {len(ser)}")
    for g in ser:
        g["frames"][0]["_sersic_rhalf"] = g["frames"][0]["sersic_rhalf"] / 40
        g["info"]["_H_Q"] = 1 - g["info"]["H_Q"]
        g["info"]["_H_PA"] = (g["info"]["H_PA"] + 90) / 180 * np.pi
    fig, axes = plt.subplots(1, 3)  # 4)
    vis.plot_value_filters(ser, "H_NSERSIC", "sersic_n", axis=axes[0])
    vis.plot_value_filters(ser, "H_RE", "_sersic_rhalf", axis=axes[1])
    vis.plot_value_filters(ser, "_H_Q", "sersic_ellip", axis=axes[2])
    # vis.plot_value_filters(ser, "_H_PA", "sersic_theta", axis=axes[3])
    axes[0].plot([0, 5], [0, 5], c="red", linestyle="--", linewidth=2)
    axes[1].plot([0, 1], [0, 1], c="red", linestyle="--", linewidth=2)
    axes[2].plot([0, 1], [0, 1], c="red", linestyle="--", linewidth=2)
    # axes[3].plot([0, np.pi], [0, np.pi], c="red", linestyle="--", linewidth=2)
    for i in range(3):  # 4):
        axes[i].lines[0].set(
            markerfacecolor="#034a8360", markeredgecolor="#05396f9d", alpha=0.4
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
        xlabel="$n$ \\textit{HST} (F160W)",
        ylabel="$n$ \\textit{JWST} (F444W)",
        xlim=(0, 5),
        ylim=(0, 5),
    )
    axes[1].set(
        title="",  # Sersic radius",
        xlabel="$r_\mathrm{eff}$  \\textit{HST} (F160W)",
        ylabel="$r_\mathrm{eff}$ \\textit{JWST} (F444W)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[2].set(
        title="",  # Sersic ellipticity",
        xlabel="$e$  \\textit{HST} (F160W)",
        ylabel="$e$ \\textit{JWST} (F444W)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    """
    axes[3].set(
        title="", #Sersic position angle",
        xlabel=r"$\theta_\mathrm{PA}$  \textit{HST} (F160W)",
        ylabel=r"$\theta_\mathrm{PA}$ \textit{JWST} (F444W)",
        xlim=(0, np.pi),
        ylim=(0, np.pi),
    )
    """
    fig.suptitle(
        ""  # Sersic fit comparison: HST (F160W) vs JWST ($\lambda_{RFW}=1.37\mu$m)"
    )
    fig.set_size_inches(13.3, 4.5)  # 17.5, 4.5)
    fig.set_layout_engine(layout="tight")


def ginim20_class():
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


def masking_examples_6(galaxies):
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


def masking_examples_4(galaxies):
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
    # mergers_separation()
    # mergers_separation_a()
    # bulges_separation()
    # optimal_rfw()
    # ren_et_al()
    # MID_classification()
    # sersic_comparison()
    # ginim20_class()
    # ginim20_redshift()
    # ca_redshift()
    names = [
        "COS4_02049",
        "COS4_02167",
        "COS4_17389",
        "COS4_20910",
        "U4_26324",
        "U4_21440",
    ]
    # masking_examples_6(names)
    names = [
        "COS4_02049",
        "COS4_02167",
        "U4_26324",
        "U4_21440",
    ]
    # masking_examples_4(names)
    sup_mid_bins()
    plt.show()
