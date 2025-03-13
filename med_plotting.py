import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from utils.general_utils import create_function, get_param_values
from utils.plotting_utils import create_heatmap, create_circle_heatmap, create_contour_plot

DIM_DICT = {
    "r": "Radial",
    "z": "Axial",
    "x": "Horizontal (x)",
    "y": "Horizontal (y)"
}

sweep = "amp-fill"
tag = "r1"
dim = "r"
p1_name, p2_name = sweep.split("-")
hof_path = f"post/med_{sweep}_{tag}_{dim}/hall_of_fame.csv"
eq_df = pd.read_csv(hof_path)

k_path = f"k_csvs/{sweep}/fitted_k_values_{tag}.csv"
k_df = pd.read_csv(k_path)
k_df = k_df[["study name", f"{dim} lacey k"]]
k_df["param_values"] = k_df["study name"].apply(get_param_values)


k_df[p1_name] = k_df["param_values"].apply(lambda x: x[0][1])
k_df[p2_name] = k_df["param_values"].apply(lambda x: x[1][1]) 


comp = 13
eq_str = eq_df.loc[eq_df["Complexity"] == comp, "Equation"].squeeze()

func = create_function(eq_str, p1_name, p2_name)

k_df["pred_k"] = k_df.apply(lambda row: func(row[p1_name], row[p2_name]), axis=1)
k_df["percentage_error"] = 100 * np.abs(k_df[f"{dim} lacey k"] - k_df["pred_k"]) / k_df[f"{dim} lacey k"]

hm_kwargs = {
    "title": f"Complexity {comp} {DIM_DICT[dim]} Performance on {sweep}",
    "xlabel": "Amplitude (m)",
    "ylabel": "Number of Particles",
    "cbar_label": f"Mixing Rate Constant ({DIM_DICT[dim]}) Percentage Error"
}

plot_var = f"pred_k"
plot_fun = create_circle_heatmap

heatmap_plot = plot_fun(k_df, p1_name, p2_name, plot_var, colmin=None, colmax=None, **hm_kwargs)
plt.show()