import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, List

from consts_visualizer import *
from preprocessing import *

ModeNorm = Literal["peak", "two_peaks", "mask_unused", "none"]

def _mask_None(x):
    return np.zeros_like(x)

def processing(
        NAME_DATA: str,
        norm_mode: ModeNorm="peak",
        mask=_mask_None,
        center=1003,
        left=950,
        right=1050,
        half_width=8
    ) -> pd.DataFrame:
    path_data = DIR_DATA / NAME_DATA

    data = pd.read_csv(path_data, sep='	 ', header=None, engine='python')

    data.columns = ["Raman_shift", "CCD"]

    ########################

    id_lazer = data["CCD"].idxmax()
    data = data.loc[id_lazer:]

    ccd_dark = data["CCD"].min()
    id_dark = data["CCD"].idxmin()
    data = data.loc[id_dark:]
    data["CCD"] -= ccd_dark

    data = data[data["Raman_shift"] >= SHIFT_MIN]
    data = data[data["Raman_shift"] <= SHIFT_MAX]

    data["CCD"] = remove_spikes(data["CCD"], kernel_size=15, threshold=8)
    data["CCD_clean"] = remove_negative_artifacts(data["Raman_shift"], data["CCD"], threshold=3.5)[0]
    data["CCD_corrected"], data["Baseline"] = correct_baseline(data["CCD_clean"], lam=1e6, p=0.03)
    
    if norm_mode == "peak":
        data["CCD_shift"] = shift_peak(
            data,
            x_col="Raman_shift",
            y_col="CCD_corrected",
            left=left,
            right=right,
            target=center
        )

        data["CCD_norm"] = normalize_to_peak(
            data["Raman_shift"],
            data["CCD_shift"],
            center=center,
            half_width=half_width
        )
        data["CCD_res"] = data["CCD_norm"]
    elif norm_mode == "two_peaks":
        data["CCD_shift"], half_width = shift_two_peak(
            data, 
            x_col="Raman_shift",
            y_col="CCD_corrected",
            left=left,
            right=right,
            target=center
        )

        data["CCD_norm"] = normalize_to_peak(
            data["Raman_shift"],
            data["CCD_shift"],
            center=center,
            half_width=half_width)
        data["CCD_res"] = data["CCD_norm"]
    elif norm_mode == "mask_unused":
        data["CCD_shift"] = shift_peak(
            data,
            x_col="Raman_shift",
            y_col="CCD_corrected"
        )

        data = normalize_by_unchanged_parts(
            data,
            x_col="Raman_shift",
            y_col="CCD_shift",
            mask=mask(data["Raman_shift"]),
            mode="vector",
            new_col="CCD_norm",
        )[0]
        data["CCD_res"] = data["CCD_norm"]
    else:
        data["CCD_res"] = data["CCD_corrected"]

    return data

def average_shifted_data(shifted_data, x_col="Raman_shift", y_col="CCD"):
    # Берём общую сетку Raman_shift из первого спектра
    x = shifted_data[0][x_col].to_numpy(dtype=float)

    ys = []

    for df in shifted_data:
        x_cur = df[x_col].to_numpy(dtype=float)

        if not np.allclose(x_cur, x):
            raise ValueError("Сетки Raman_shift не совпадают")

        y = df[y_col].to_numpy(dtype=float)
        ys.append(y)

    # Матрица размера: количество спектров × количество точек
    ys = np.vstack(ys)

    # Оставляем только те точки, где у всех спектров есть значения
    good = ~np.any(np.isnan(ys), axis=0)

    x_good = x[good]
    ys_good = ys[:, good]

    mean_y = np.mean(ys_good, axis=0)
    std_y = np.std(ys_good, axis=0, ddof=1)

    result = pd.DataFrame({
        x_col: x_good,
        y_col + "_mean": mean_y,
        "std": std_y,
    })

    return result, ys_good

def plot_once_graph(
        NAME: str,
        SUB_DIR: str="",
        norm_mode: ModeNorm= "peak",
        mask=_mask_None,
        draw_raw=True,
        left=950,
        right=1050,
        center=1003,
        half_wigth=8
    ) -> pd.DataFrame:
    NAME_GRAPH = NAME + ".png"
    NAME_DATA = NAME + ".txt"
    GRAPH_PATH = DIR_GRAPHS
    if SUB_DIR != "":
        GRAPH_PATH /= SUB_DIR
    GRAPH_PATH /= NAME_GRAPH

    data = processing(
        NAME_DATA,
        norm_mode=norm_mode,
        mask=mask,
        left=left,
        right=right,
        center=center,
        half_width=half_wigth
    )

    ########################
    if draw_raw:
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(10, 6))

        ax1.plot(data["Raman_shift"], data["CCD_res"], "", color="royalblue")

        ax2.plot(data["Raman_shift"], data["CCD"], "", color="red")
        ax2.plot(data["Raman_shift"], data["Baseline"], "", color="orange")
        ax2.plot(data["Raman_shift"], data["CCD_clean"], "", color="royalblue")
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

        ax1.plot(data["Raman_shift"], data["CCD_res"], "", color="royalblue")

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAPH_PATH)
    return data

def plot_mean_graph(NAME: str,
                    data: pd.DataFrame,
                    x_col="Raman_shift",
                    y_col="CCD",
                    SUB_DIR: str="") -> pd.DataFrame:
    NAME_GRAPH = NAME + ".png"
    GRAPH_PATH = DIR_GRAPHS
    if SUB_DIR != "":
        GRAPH_PATH /= SUB_DIR
    GRAPH_PATH /= NAME_GRAPH

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    ax1.plot(data[x_col], data[y_col], "", color="royalblue")

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAPH_PATH)
    return data

def plot_mean_graphs(
        NAME: str,
        datas,
        x_col="Raman_shift",
        y_col="CCD",
        SUB_DIR: str=""
):
    NAME_GRAPH = NAME + ".png"
    GRAPH_PATH = DIR_GRAPHS
    if SUB_DIR != "":
        GRAPH_PATH /= SUB_DIR
    GRAPH_PATH /= NAME_GRAPH
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    for i in datas:
        ax1.plot(i[0][x_col], i[0][y_col], label=i[1], color=COLORS[i[1]])
    
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    ax1.legend()
    fig.savefig(GRAPH_PATH)