import json

import matplotlib.pyplot as plt
import numpy as np

import resu
import run
import vis
from ren_values import *

"""To generate the plots, first update the paths bellow to correspond to the location of the output/input dictionaries. 
"""
remove = ["GS4_18890", "COS4_17600", "GS4_19797", "COS4_05758", "U4_28125", "G4_06569", "GS4_30028", "GS4_29773"]
rem_g = [{"name":g} for g in remove]

raw_o = run.fetch_json("dict_out/out_full_matched_5_m.json")["galaxies"]
raw = resu.get_complement(raw_o, rem_g)

fil = resu.galaxy_pruning(raw)


def ginim20_redshift(save = None):
    fig, axs = plt.subplots()
    z = resu.get_separate_in_value(fil, "z bin")
    rang = ((-2.3, -0.8), (0.38, 0.64))
    vis.plot_smooth2d_subt(
        (z["high_z"], z["low_z"]), "M20", "Gini", axis=axs, rang=rang
    )
    vis.plot_correlation_filters((fil,), "M20", "Gini", axis=axs)
    vis.plot_points_comp((z["high_z"],), "M20", "Gini", axis=axs, ealpha=0.8)
    vis.plot_points_comp((z["low_z"],), "M20", "Gini", axis=axs, ealpha=0.8)

    axs.errorbar(
        yao_m[1],
        yao_g[1],
        xerr=(yao_m[2][0], yao_m[2][1]),
        yerr=(yao_g[2][0], yao_g[2][1]),
        capsize=5,
        c="black",
        alpha=0.8,
        fmt="none",
    )
    axs.plot(
        yao_m[1],
        yao_g[1],
        c="darkgreen",
        marker="D",
        linestyle="",
        markersize=4,
        label="Yao et al. (2023)",
    )
    axs.plot(yao_m[1], yao_g[1], c="darkgreen", linestyle=":")
    axs.plot(ren_mg[0], ren_mg[1], c="blue", label="Ren et al. (2024)")

    axs.lines[0].set(marker=".", markersize=1.5, c="black", alpha=0.4)
    axs.collections[0].set(alpha=0.5)
    axs.set_xlabel("$M_{20}$")
    axs.set_ylabel("Gini")
    axs.set_title("")
    axs.set_xlim(rang[0])
    axs.set_ylim(rang[1])
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
    axs.lines[0].set(label="_")
    axs.lines[12].set(c="darkblue")
    axs.lines[13].set(c="darkblue")
    L = axs.legend(fontsize=12)
    L.texts[0].set_text("high $z$ mean")
    L.texts[1].set_text("high $z$ median")
    L.texts[2].set_text("low $z$ mean")
    L.texts[3].set_text("low $z$ median")
    axs.set_ylim(0.45, 0.58)
    axs.set_xlim(-2, -1.4)

    fig.set_size_inches(5.5, 5)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)

def ca_redshift(save = None):
    fig, axs = plt.subplots()
    z = resu.get_separate_in_value(fil, "z bin")
    rang = ((2.5, 3.5), (0, 0.3))
    vis.plot_smooth2d_subt((z["high_z"], z["low_z"]), "C", "A", axis=axs, rang=rang)
    vis.plot_correlation_filters((fil,), "C", "A", axis=axs)
    vis.plot_points_comp((z["high_z"],), "C", "A", axis=axs, ealpha=0.8)
    vis.plot_points_comp((z["low_z"],), "C", "A", axis=axs, ealpha=0.8)

    axs.errorbar(
        yao_c[1],
        yao_a[1],
        xerr=(yao_c[2][0], yao_c[2][1]),
        yerr=(yao_a[2][0], yao_a[2][1]),
        capsize=5,
        c="black",
        alpha=0.8,
        fmt="none",
    )
    axs.plot(
        yao_c[1],
        yao_a[1],
        c="darkgreen",
        marker="D",
        linestyle="",
        markersize=4,
        label="Yao et al. (2023)",
    )
    axs.plot(yao_c[1], yao_a[1], c="darkgreen", linestyle=":")
    axs.plot(ren_ca[0], ren_ca[1], c="blue", label="Ren et al. (2024)")

    axs.lines[0].set(marker=".", markersize=1.5, c="black", alpha=0.4)
    axs.collections[0].set(alpha=0.5)
    axs.set_xlabel("Concentration, $C$")
    axs.set_ylabel("Asymmetry, $A$")
    axs.set_title("")
    axs.set_xlim(rang[0])
    axs.set_ylim(rang[1])
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
    axs.lines[0].set(label="_")
    axs.lines[12].set(c="darkblue")
    axs.lines[13].set(c="darkblue")
    L = axs.legend(fontsize=12)
    L.texts[0].set_text("high $z$ mean")
    L.texts[1].set_text("high $z$ median")
    L.texts[2].set_text("low $z$ mean")
    L.texts[3].set_text("low $z$ median")

    fig.set_size_inches(5.5, 5)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)

def mid_bins(save = None):
    vis.plot_grided(
        fil,
        ["M", "I", "D"],
        4,
        vis.points_multi_clas,
        title="MID statistics for different visual morphology classifications",
    )
    fig = plt.gcf()
    axes = fig.axes
    axes[0].set_ylim(-0.1, 0.5)
    axes[4].set_ylim(-0.1, 0.8)
    axes[8].set_ylim(-0.05, 0.45)
    axes[0].set_xticks([0, 1], ["without bulge", "with bulge"])
    axes[1].set_xticks([0, 1], ["others", "inter./mergers"])
    axes[0].set_ylabel("Multinode st., $M$")
    axes[4].set_ylabel("Intensity st., $I$")
    axes[8].set_ylabel("Deviation st., $D$")

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
    for i in range(len(axes[8].patches)):
        axes[8].patches[i].set_label("_")
    L = axes[8].legend(loc=1)
    L.texts[0].set_text("mean")
    L.texts[1].set_text("median")
    fig.set_size_inches(9.5, 7.4)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)

def cas_bins(save = None):
    vis.plot_grided(
        fil,
        ["C", "A", "S"],
        4,
        vis.points_multi_clas,
        title="CAS statistics for different visual morphology classifications",
    )
    fig = plt.gcf()
    axes = fig.axes
    axes[0].set_ylim(1.8, 4)
    axes[4].set_ylim(-0.1, 0.65)
    axes[8].set_ylim(-0.015, 0.065)
    axes[0].set_xticks([0, 1], ["without bulge", "with bulge"])
    axes[1].set_xticks([0, 1], ["others", "inter./mergers"])
    axes[0].set_ylabel("Concentration st., $C$")
    axes[4].set_ylabel("Asymmetry st., $A$")
    axes[8].set_ylabel("Smoothness st., $S$")

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
    for i in range(len(axes[8].patches)):
        axes[8].patches[i].set_label("_")
    L = axes[8].legend(loc=1)
    L.texts[0].set_text("mean")
    L.texts[1].set_text("median")
    fig.set_size_inches(9.5, 7.4)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)

def ser_bins(save = None):
    vis.plot_grided(
        fil,
        ["sersic_n", "sersic_rhalf", "sersic_amplitude"],
        4,
        vis.points_multi_clas,
        title="Sersic fit values for different visual morphology classifications",
    )
    fig = plt.gcf()
    axes = fig.axes
    axes[0].set_ylim(-0.3, 2.5)
    axes[4].set_ylim(0, 35)
    axes[8].set_ylim(-0.2, 4)
    axes[0].set_xticks([0, 1], ["without bulge", "with bulge"])
    axes[1].set_xticks([0, 1], ["others", "inter./mergers"])
    axes[0].set_ylabel("Sersic index, $n$")
    axes[4].set_ylabel("Sersic radius (px), $r_\mathrm{half}$")
    axes[8].set_ylabel("Sersic amplitude")

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
    for i in range(len(axes[8].patches)):
        axes[8].patches[i].set_label("_")
    L = axes[8].legend(loc=1)
    L.texts[0].set_text("mean")
    L.texts[1].set_text("median")
    fig.set_size_inches(9.5, 7.4)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)

def gm20_bins(save = None):
    vis.plot_grided(
        fil,
        ["Gini", "M20", "S(G, M20)", "F(G, M20)"],
        4,
        vis.points_multi_clas,
        title="Gini-$M_{20}$ statistics for different visual morphology classifications",
    )
    fig = plt.gcf()
    axes = fig.axes
    axes[0].set_ylim(0.38, 0.62)
    axes[4].set_ylim(-2.2, -0.8)
    axes[8].set_ylim(-0.18, 0.1)
    axes[12].set_ylim(-1.25, 0.7)
    axes[0].set_xticks([0, 1], ["without bulge", "with bulge"])
    axes[1].set_xticks([0, 1], ["others", "inter./mergers"])
    axes[0].set_ylabel("Gini")
    axes[4].set_ylabel("$M_{20}$")
    axes[8].set_ylabel("Bulge st., $S(G, M_{20})$")
    axes[12].set_ylabel("Merger st., $F(G, M_{20})$")

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
    for i in range(len(axes[12].patches)):
        axes[12].patches[i].set_label("_")
    L = axes[12].legend(loc=1)
    L.texts[0].set_text("mean")
    L.texts[1].set_text("median")
    fig.set_size_inches(9.5, 9.5)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)

def mid_evol(save = None):
    vis.plot_grided(
        fil,
        ["M", "I", "D"],
        4,
        vis.points_multi_bins,
        title="Evolution of MID statistics in $z$, $SFR$, $M_\star$ and $r_\mathrm{eff}$",
    )
    fig = plt.gcf()
    axes = fig.axes
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

    axes[0].set_ylim(-0.05, 0.25)
    axes[4].set_ylim(-0.05, 0.4)
    axes[8].set_ylim(-0.02, 0.25)
    axes[0].set_ylabel("Multinode st., $M$")
    axes[4].set_ylabel("Intensity st., $I$")
    axes[8].set_ylabel("Deviation st., $D$")

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
    L = axes[8].legend(loc=1)
    L.texts[0].set_text("mean")
    L.texts[1].set_text("median")
    fig.set_size_inches(9.5, 7.4)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)

def cas_evol(save = None):
    vis.plot_grided(
        fil,
        ["C", "A", "S"],
        4,
        vis.points_multi_bins,
        title="Evolution of CAS statistics in $z$, $SFR$, $M_\star$ and $r_\mathrm{eff}$",
    )
    fig = plt.gcf()
    axes = fig.axes
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

    axes[0].set_ylim(2.2, 3.7)
    axes[4].set_ylim(0, 0.4)
    axes[8].set_ylim(-0.004, 0.047)
    axes[0].set_ylabel("Concentration st., $C$")
    axes[4].set_ylabel("Asymmetry st., $A$")
    axes[8].set_ylabel("Smoothness st., $S$")

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
    L = axes[8].legend(loc=1)
    L.texts[0].set_text("mean")
    L.texts[1].set_text("median")
    fig.set_size_inches(9.5, 7.4)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)

def ser_evol(save = None):
    vis.plot_grided(
        fil,
        ["sersic_n", "sersic_rhalf", "sersic_amplitude"],
        4,
        vis.points_multi_bins,
        title="Evolution of Sersic fit values in $z$, $SFR$, $M_\star$ and $r_\mathrm{eff}$",
    )
    fig = plt.gcf()
    axes = fig.axes
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

    axes[0].set_ylim(-0.15, 2.2)
    axes[4].set_ylim(2, 32)
    axes[8].set_ylim(-0.1, 3.6)
    axes[0].set_ylabel("Sersic index, $n$")
    axes[4].set_ylabel("Sersic radius (px), $r_\mathrm{half}$")
    axes[8].set_ylabel("Sersic amplitude")

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
    L = axes[8].legend(loc=1)
    L.texts[0].set_text("mean")
    L.texts[1].set_text("median")
    fig.set_size_inches(9.5, 7.4)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)

def gm20_evol(save = None):
    vis.plot_grided(
        fil,
        ["Gini", "M20", "S(G, M20)", "F(G, M20)"],
        4,
        vis.points_multi_bins,
        title="Evolution of Gini-$M_{20}$ statistics in $z$, $SFR$, $M_\star$ and $r_\mathrm{eff}$",
    )
    fig = plt.gcf()
    axes = fig.axes
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

    axes[0].set_ylim(0.42, 0.6)
    axes[4].set_ylim(-2.15, -1.2)
    axes[8].set_ylim(-0.15, 0.028)
    axes[12].set_ylim(-0.85, 0.4)
    axes[0].set_ylabel("Gini")
    axes[4].set_ylabel("$M_{20}$")
    axes[8].set_ylabel("Bulge st., $S(G, M_{20})$")
    axes[12].set_ylabel("Merger st., $F(G, M_{20})$")
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
    L = axes[12].legend(loc=1)
    L.texts[0].set_text("mean")
    L.texts[1].set_text("median")
    fig.set_size_inches(9.5, 9.5)
    fig.set_layout_engine(layout="tight")
    if save is not None:
        fig.savefig(save)

if __name__ == "__main__":
    """Comment out lines corresponding to plots you want to create."""
    #ginim20_redshift(save = "../../out/supl/josef_z_gm20.png")
    #ca_redshift(save = "../../out/supl/josef_z_cas.png")
    #mid_bins(save = "../../out/supl/josef_mid_bins.png")
    #cas_bins(save = "../../out/supl/josef_cas_bins.png")
    #ser_bins(save = "../../out/supl/josef_ser_bins.png")
    #gm20_bins(save = "../../out/supl/josef_gm20_bins.png")
    #mid_evol(save = "../../out/supl/josef_mid_evol.png")
    #cas_evol(save = "../../out/supl/josef_cas_evol.png")
    #ser_evol(save = "../../out/supl/josef_ser_evol.png")
    #gm20_evol(save = "../../out/supl/josef_gm20_evol.png")
    plt.show()
