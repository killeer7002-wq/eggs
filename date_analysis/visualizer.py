import re
from typing import List

from consts_visualizer import *
from processing_visualizer import *

def give_data_for_averaging(
        names: List[str],
        norm_mode: ModeNorm = "peak",
        mask=None,
        left=950,
        right=1050,
        center=1003,
        half_width=8
):
    data = []
    for regex in names:
        pattern = re.compile(regex)
        files = [
            path.name
            for path in DIR_DATA.iterdir()
            if path.is_file() and pattern.search(path.name)
        ]
        for file_name in files:
            if mask == None:
                data.append(processing(
                    file_name,
                    norm_mode=norm_mode,
                    left=left,
                    right=right,
                    center=center,
                    half_width=half_width
                ))
            else:
                data.append(processing(
                    file_name,
                    norm_mode=norm_mode,
                    mask=mask,
                    left=left,
                    right=right,
                    center=center,
                    half_width=half_width
                ))
    return data

def make_changed_mask(x):
    x = np.asarray(x, dtype=float)

    mask = (
        ((x >= 970) & (x <= 1050)) |
        ((x >= 1200) & (x <= 1500)) |
        ((x >= 1600) & (x <= SHIFT_MAX))
    )

    return mask

def interesting_peak(x):
    return (x >= 1420) & (x <= 1480)

def metrics(data, data_average, name) -> pd.DataFrame:
    interest_space = data_average[interesting_peak(data_average["Raman_shift"])]
    peak = data_average.loc[interest_space["CCD_res_mean"].idxmax()]
    areas = []
    for i in data:
        areas.append(area_peak(
            i["Raman_shift"],
            i["CCD_res"],
            center=peak["Raman_shift"],
            half_width=int(len(interest_space) / 2)
        ))
    res = pd.DataFrame([{
        "Type": name,
        "Raman_shift": peak["Raman_shift"],
        "CCD": peak["CCD_res_mean"],
        "Mean_Area": np.mean(areas),
        "Std_Area": np.std(areas)
    }])
    return res

if __name__ == "__main__":
    for name in NAMES:
        plot_once_graph(
            name, 
            SUB_DIR=re.match(r"^\D*", name).group(),
            norm_mode="peak",
            mask=make_changed_mask,
            draw_raw=False
            )
    
    datas = []

    res_alb = pd.DataFrame(columns=[
        "Type",
        "Raman_shift",
        "CCD",
        "Mean_Area",
        "Std_Area",
    ])
    for name in MEAN_NAMES_ALBUMIN.keys():

        data = give_data_for_averaging(
            MEAN_NAMES_ALBUMIN[name],
            norm_mode="peak",
            mask=make_changed_mask,
        )
        try:
            data_average = average_shifted_data(
                data,
                x_col="Raman_shift",
                y_col="CCD_res"
            )[0]
            datas.append((data_average, name))
            plot_mean_graph(
                name,
                data_average,
                x_col="Raman_shift",
                y_col="CCD_res_mean",
                SUB_DIR="mean"
            )
        except:
            print(name, len(data))
        
        res_alb = pd.concat([res_alb, metrics(data, data_average, name)], ignore_index=True)
    
    plot_mean_graphs(
        "Albumin",
        datas,
        x_col="Raman_shift",
        y_col="CCD_res_mean",
        SUB_DIR="mean"
    )

    print(res_alb)

    datas = []

    res_yolk = pd.DataFrame(columns=[
        "Type",
        "Raman_shift",
        "CCD",
        "Mean_Area",
        "Std_Area",
    ])
    for name in MEAN_NAMES_YOLK.keys():

        data = give_data_for_averaging(
            MEAN_NAMES_YOLK[name],
            norm_mode="none",
            mask=None,
            # mask=make_changed_mask,
            left=1600,
            right=1800,
            center=1655,
            half_width=10
        )
        try:
            data_average = average_shifted_data(
                data,
                x_col="Raman_shift",
                y_col="CCD_res"
            )[0]
            datas.append((data_average, name))
            plot_mean_graph(
                name,
                data_average,
                x_col="Raman_shift",
                y_col="CCD_res_mean",
                SUB_DIR="mean"
            )
        except:
            print(name, len(data))
        
        res_yolk = pd.concat([res_yolk, metrics(data, data_average, name)], ignore_index=True)

    plot_mean_graphs(
        "Yolk",
        datas,
        x_col="Raman_shift",
        y_col="CCD_res_mean",
        SUB_DIR="mean"
    )

    left_b = 1000
    right_b = 1200

    data_av = datas[0][0]
    peak = data_av.loc[data_av["CCD_res_mean"][(data_av["Raman_shift"] <= right_b) & (data_av["Raman_shift"] >= left_b)].idxmax()]
    print(f"left={left_b},",
          f"right={right_b},",
          f"center={round(peak["Raman_shift"])},",
          sep="\n" + " " * 12)

    print(res_yolk)
