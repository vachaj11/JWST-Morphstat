import vis, resu, run
import line_separation as ls
import matplotlib.pyplot as plt
import numpy as np
import json
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
        label="non/merger boundary",
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
    mmabove = ls.get_above_line(filmm, "M20", "Gini", [0,0.33], -0.14)
    omabove = ls.get_above_line(filom, "M20", "Gini", [0,0.33], -0.14)
    print(f"{100*mmabove:.1f} % mergers above, {(1-omabove)*100:.1f} % non-mergers bellow")
    print("Best separation line:")
    print(f"Gini = M_20*({slope:.3f})+({point[1]-slope*point[0]:.3f})")
    print("Separation:")
    print(f"{100*diff[1][0]:.1f} % mergers above, {(1-diff[1][1])*100:.1f} % non-mergers bellow")
    print("------------------\n")

    axs.set_xlabel("M$_{20}$")
    axs.set_ylabel("Gini")
    axs.set_title("Gini-M$_{20}$ statistics for non/mergers")
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
    point, slope, diff = ls.max_sep(filmb, filob, "M20", "Gini")
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
    print("\n-----------------")
    print("Bulges separation")
    print("-----------------")
    print("Line used in literature:")
    print(f"Gini = M_20*({0.14:.3f})+({0.8:.3f})")
    print("Separation:")
    mbabove = ls.get_above_line(filmm, "M20", "Gini", [0,0.8], 0.14)
    obabove = ls.get_above_line(filom, "M20", "Gini", [0,0.8], 0.14)
    print(f"{100*(1-mbabove):.1f} % bulges bellow, {obabove*100:.1f} % non-bulges above")
    print("Best separation line:")
    print(f"Gini = M_20*({slope:.3f})+({point[1]-slope*point[0]:.3f})")
    print("Separation:")
    print(f"{100*(1-diff[1][0]):.1f} % bulges bellow, {diff[1][1]*100:.1f} % non-bulges above")
    print("-----------------\n")

    axs.set_xlabel("M$_{20}$")
    axs.set_ylabel("Gini")
    axs.set_title("Gini-M$_{20}$ statistics for non/bulges")
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
    axes[4].set_ylabel("M$_{20}$")
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
    axes[4].set_ylabel("M$_{20}$")
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

def separation_table(gal1, gal2, parameters):
    """Generates a table with separation values in various 2d parameter
    spaces formatted as latex `tabular` environment with coloured cells. The
    sets of galaxies between which the separation is to be made are the `gal1`
    and `gal2` arguments, while the `parameters` argument defines the 
    parameter spaces to be considered.
    """
    print("This might take some time... (~3 mins)")
    vals = ls.best_parameters(gal1, gal2, parameters)
    vald = {k:vals[k][2] for k in vals}
    vmax = max([vald[k] for k in vald])
    vmin = min([vald[k] for k in vald])
    cmap = lambda c: int((c-vmin)/(vmax-vmin)*70)
    values = ["\hline & \\textbf{"+"} & \\textbf{".join(parameters)+"}"]
    for nr in parameters:
        row = []
        for nc in parameters:
            val = 0
            for k in vald:
                if nc in k and nr in k:
                    val = vald[k]
            sval = f"\cellcolor{{red!{cmap(val)}}} {val:.3f}"
            if parameters.index(nc)>=parameters.index(nr):
                row.append(sval)
            else:
                row.append("")
        values.append(" & ".join(["\\textbf{"+nr+"}"]+row))
    table = " \\\\\hline\n".join(values+[""])
    table = table.replace("\\textbf{sersic_n}","$\\text{\\textbf{sersic}}_n$")
    table = table.replace("\\textbf{sersic_rhalf}","$\\text{\\textbf{sersic}}_r$")
    table = table.replace("\\textbf{M20}","$\\text{\\textbf{M}}_{20}$")
    starts = f"\\begin{{tabular}}{{{"|"+"c|"*(len(parameters)+1)}}}\n"
    ends = "\end{tabular}"
    table = starts+table+ends
    print("\n----------\n")
    print(table)
    print("\n----------\n")    

def masking_examples(galaxies):
    fig = plt.figure(figsize=(7.65, 6))
    gs = fig.add_gridspec(2, 3, wspace=0.06, hspace=0.17, left= 0.04, right= 0.96, top =0.87, bottom=0.06)
    axs = gs.subplots().flatten()
    galaxies = galaxies[:6]
    gal_in = [resu.get_galaxy_entry(full, g) for g in galaxies]
    gal_ps = resu.get_optim_rfw(gal_in, fixed_rfw = 1.33)
    gal_data = run.galaxies_data(gal_ps, return_object=True)

    for i in range(len(gal_data)):
        plot_segmentation(gal_data[i], axis=axs[i])
        axs[i].set_title(galaxies[i])
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    axs[3].plot([],[],linestyle="--",c="red",linewidth=1.7,label="Masked area")
    axs[3].plot([],[],linestyle="--",c="blue",linewidth=1.7,label="Target area")
    L = axs[3].legend(loc=1)
    fig.suptitle("Examples of target identification and masking")

    
def plot_segmentation(g, axis=None):
    if axis is None:
        axis = plt.gca()
    f = g.frames[0]
    l = np.log(f.convolved)
    ln = np.nan_to_num(np.log(f.data), nan=-100)
    r = np.nanmax(l)-np.nanmin(l)
    axis.imshow(ln, cmap="gray", vmin=np.nanmin(l)+r/2,vmax=np.nanmax(l))
    axis.contour(f.target, levels=1, colors= "blue", linestyles="--", linewidths=1.7)
    axis.contour(f.mask, levels=1, colors= "red", linestyles="--", linewidths=1.7)

if __name__ == "__main__":
    """Comment out lines corresponding to plots you want to create."""
    #mergers_separation()
    #bulges_separation()
    #optimal_rfw()
    #ren_et_al()
    #MID_classification()
    #sersic_comparison()
    #ginim20_class()
    names = ["COS4_02049", "COS4_02167", "COS4_17389", "COS4_20910", "U4_26324", "U4_21440"]
    #masking_examples(names)
    plt.show()
    parameters = ["Gini", "M20", "C","A","S", "M","I","D", "sersic_n", "sersic_rhalf"]
    #separation_table(filmm, filom, parameters)
    #separation_table(filmb, filob, parameters)

