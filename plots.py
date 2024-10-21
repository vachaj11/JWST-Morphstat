import vis, resu, run
import line_separation as ls
import matplotlib.pyplot as plt
import numpy as np
import json
from ren_values import *

"""To generate the plots, first update the paths bellow to correspond to the location of the output/input dictionaries. 
"""

raw = run.fetch_json("dict_out/out_full_matched_2_k.json")["galaxies"]
bulges = run.fetch_json("dict_in/n4_Bulge.json")["galaxies"]
mergers = run.fetch_json("dict_in/n4_Interacting-Merger.json")["galaxies"]
full = run.fetch_json("dict_in/dictionary_full.json")["galaxies"]

fil = resu.galaxy_pruning(raw)
filmb = resu.get_subset(fil, bulges)
filob = resu.get_complement(fil, bulges)
filmm = resu.get_subset(fil, mergers)
filom = resu.get_complement(fil, mergers)

def mergers_separation():
    point, slope, _ = ls.max_sep(filmm, filom, "M20", "Gini")
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
        label="non/merger boundary",
    )
    axs.plot(
        x,
        (point[0] - x) * (-slope) + point[1],
        linewidth=2.5,
        c="gold",
        label="line of best separation",
    )

    axs.set_xlabel("M20")
    axs.set_ylabel("Gini")
    axs.set_title("Gini-M20 statistics for non/mergers")
    axs.lines[0].set(
        c="#034a8360", markeredgecolor="#05396f9d", markersize=7, alpha=0.5
    )
    axs.lines[1].set(
        c="#b7010154", markeredgecolor="#b701018b", markersize=7, alpha=0.5
    )
    L = axs.legend()
    L.texts[0].set_text("non-mergers (" + L.texts[0]._text.split("(")[-1])
    L.texts[1].set_text("interact./mergers (" + L.texts[1]._text.split("(")[-1])
    axs.set_xlim(x)
    axs.set_ylim(0.38, 0.64)
    fig.set_size_inches(5.5, 5)
    fig.set_layout_engine(layout="tight")


def bulges_separation():
    point, slope, _ = ls.max_sep(filmb, filob, "M20", "Gini")
    vis.plot_correlation_filters((filob, filmb), "M20", "Gini")
    x = np.array([-2.3, -0.8])
    fig = plt.gcf()
    axs = plt.gca()

    axs.plot(
        x,
        0.14 * x + 0.8,
        linestyle="--",
        linewidth=2.5,
        c="black",
        label="early/late boundary",
    )
    axs.plot(
        x,
        (point[0] - x) * (-slope) + point[1],
        linewidth=2.5,
        c="gold",
        label="line of best separation",
    )

    axs.set_xlabel("M20")
    axs.set_ylabel("Gini")
    axs.set_title("Gini-M20 statistics for non/bulges")
    axs.lines[0].set(
        c="#034a8360", markeredgecolor="#05396f9d", markersize=7, alpha=0.5
    )
    axs.lines[1].set(
        c="#b7010154", markeredgecolor="#b701018b", markersize=7, alpha=0.5
    )
    L = axs.legend()
    L.texts[0].set_text("non-bulges (" + L.texts[0]._text.split("(")[-1])
    L.texts[1].set_text("bulges (" + L.texts[1]._text.split("(")[-1])
    axs.set_xlim(x)
    axs.set_ylim(0.38, 0.64)
    fig.set_size_inches(5.5, 5)
    fig.set_layout_engine(layout="tight")


def optimal_rfw():
    fullg = []
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
        g["frames"][0]["_wave"] = g["frames"][0]["_wavelength"] / 100
    vis.plot_value_filters(fullg, "ZBEST", "_wave")
    fig = plt.gcf()
    axs = plt.gca()
    vis.plot_value_filters(raw, "ZBEST", "_wave", axis=axs)

    x = np.array([0.8, 2.5])
    axs.plot(
        x, (1 + x) * 1.37, linewidth=2.5, c="green", label="$\lambda_{RFW}=1.37\mu$m"
    )
    axs.plot(
        x,
        (1 + x) * 1.37 * 1.2,
        linewidth=2.5,
        c="grey",
        linestyle="--",
        label="outlier threshold",
    )
    axs.plot(x, (1 + x) * 1.37 * 0.8, linewidth=2.5, c="grey", linestyle="--")

    axs.set_xlabel("$z$")
    axs.set_ylabel("observed wavelength ($\mu$m)")
    axs.set_title("Optimal $\lambda_{RFW}$ for the JWST sample")
    axs.lines[0].set(
        c="#0f5bb42d", markeredgecolor="#0f5bb42d", markersize=7, alpha=0.4
    )
    axs.lines[1].set(c="red", markeredgecolor="red", markersize=7, alpha=0.4)
    L = axs.legend(loc=8)
    L.texts[0].set_text("all galaxies all filters")
    L.texts[1].set_text("all galaxies selected filters")
    # axs.set_xlim(x)
    axs.set_ylim(0, 5)
    fig.set_size_inches(5.5, 5)
    fig.set_layout_engine(layout="tight")


def ren_et_al():
    vis.plot_grided(
        fil,
        ["C", "Gini", "M20"],
        2,
        vis.points_multi_bins,
        title="Comparison to Ren et al. (2024) Figure 10.",
        include={0, 2},
    )
    fig = plt.gcf()
    axes = fig.axes
    axes[0].plot(ren_cr[0], ren_cr[1], c="blue")
    axes[1].plot(ren_cm[0], ren_cm[1], c="blue")
    axes[2].plot(ren_gr[0], ren_gr[1], c="blue")
    axes[3].plot(ren_gm[0], ren_gm[1], c="blue")
    axes[4].plot(ren_mr[0], ren_mr[1], c="blue", label="Ren et al.")
    axes[5].plot(ren_mm[0], ren_mm[1], c="blue")

    axes[0].set_ylabel("Concentration, C")
    axes[0].set_ylim(2, 4)
    axes[2].set_ylim(0.4, 0.62)
    axes[4].set_ylim(-2.2, -1)
    axes[4].set_xlim(0.6, 2.7)
    axes[5].set_xlim(9.85, 11.45)
    L = axes[4].legend(loc=2)
    L.texts[0].set_text("median")
    L.texts[1].set_text("mean")
    L.texts[2].set_text("Ren et al.")
    for i in range(3, len(L.texts)):
        L.texts[i].set_text("_")
    fig.set_size_inches(6.3, 8)
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
    axes[5].set_xticks([0, 1], ["non-merg.", "interac./merg."])
    for i in range(len(axes[4].patches)):
        axes[4].patches[i].set_label("_")
    L = axes[4].legend(loc=1)
    L.texts[0].set_text("median")
    L.texts[1].set_text("mean")
    fig.set_size_inches(6.3, 8)
    fig.set_layout_engine(layout="tight")


def sersic_comparison():
    ser = resu.galaxy_pruning(fil, strength="sersic")
    for g in ser:
        g["frames"][0]["_sersic_rhalf"] = g["frames"][0]["sersic_rhalf"] / 40
        g["info"]["_H_Q"] = 1 - g["info"]["H_Q"]
        g["info"]["_H_PA"] = (g["info"]["H_PA"] + 90) / 180 * np.pi
    fig, axes = plt.subplots(1, 4)
    vis.plot_value_filters(ser, "H_NSERSIC", "sersic_n", axis=axes[0])
    vis.plot_value_filters(ser, "H_RE", "_sersic_rhalf", axis=axes[1])
    vis.plot_value_filters(ser, "_H_Q", "sersic_ellip", axis=axes[2])
    vis.plot_value_filters(ser, "_H_PA", "sersic_theta", axis=axes[3])
    axes[0].plot([0, 5], [0, 5], c="red", linestyle="--", linewidth=2)
    axes[1].plot([0, 1], [0, 1], c="red", linestyle="--", linewidth=2)
    axes[2].plot([0, 1], [0, 1], c="red", linestyle="--", linewidth=2)
    axes[3].plot([0, np.pi], [0, np.pi], c="red", linestyle="--", linewidth=2)
    for i in range(4):
        axes[i].lines[0].set(
            markerfacecolor="#034a8360", markeredgecolor="#05396f9d", alpha=0.4
        )

    axes[0].set(
        title="Sersic n", xlabel="n HST", ylabel="n JWST", xlim=(0, 5), ylim=(0, 5)
    )
    axes[1].set(
        title="Sersic radius",
        xlabel="r HST (arcsec)",
        ylabel="r JWST (arcsec)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[2].set(
        title="Sersic ellipticity",
        xlabel="e HST",
        ylabel="e JWST",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[3].set(
        title="Sersic position angle",
        xlabel=r"$\theta$ HST (rad)",
        ylabel=r"$\theta$ JWST (rad)",
        xlim=(0, np.pi),
        ylim=(0, np.pi),
    )
    fig.suptitle(
        "Sersic fit comparison: HST (F160W) vs JWST ($\lambda_{RFW}=1.37\mu$m)"
    )
    fig.set_size_inches(17.5, 5)
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
    axes[3].set_xticks([0, 1], ["non-merg.", "interac./merg."])
    axes[0].set_ylabel("Merger st. S(G, M20)")
    axes[2].set_ylabel("Bulge st. F(G, M20)")
    axes[0].patch.set_alpha(0)
    axes[3].patch.set_alpha(0)
    for i in {"top", "bottom", "left", "right"}:
        axes[1].spines[i].set(color="#b70101", linewidth=3.5, linestyle=":")
        axes[2].spines[i].set(color="#b70101", linewidth=3.5, linestyle=":")
    axes[0].spines["right"].set(linestyle="")
    axes[0].spines["bottom"].set(linestyle="")
    axes[3].spines["left"].set(linestyle="")
    axes[3].spines["top"].set(linestyle="")
    for i in range(len(axes[0].patches)):
        axes[0].patches[i].set_label("_")
    L = axes[0].legend(loc=2)
    L.texts[0].set_text("median")
    L.texts[1].set_text("mean")
    fig.set_size_inches(6.3, 6)
    fig.set_layout_engine(layout="tight")


if __name__ == "__main__":
    """Comment out lines corresponding to plots you want to create."""
    # mergers_separation()
    # bulges_separation()
    # optimal_rfw()
    # ren_et_al()
    # MID_classification()
    # sersic_comparison()
    # ginim20_class()
    plt.show()
