from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.interpolate import griddata


def create_heatmap(df: pd.DataFrame, x_axis: str, y_axis: str, color_variable: str, colmin=None, colmax=None, **kwargs):
    """
    Creates a heatmap from a DataFrame using specified x and y axes and a color variable.
    Keyword arguments (passed as **kwargs):
    - title: Title for the heatmap (default is f"Heatmap of {color_variable}")
    - xlabel: Label for the x-axis (default is x_axis)
    - ylabel: Label for the y-axis (default is y_axis)
    - cbar_label: Label for the colorbar (default is color_variable)
    - fontsize: Font size for labels and ticks (default is 12)
    """
    df[color_variable] = pd.to_numeric(df[color_variable], errors="coerce")

    font_size = kwargs.get("fontsize", 12)

    title = kwargs.get("title", f"Heatmap of {color_variable}")
    xlabel = kwargs.get("xlabel", x_axis)
    ylabel = kwargs.get("ylabel", y_axis)
    cbar_label = kwargs.get("cbar_label", color_variable)

    heatmap_data = df.pivot_table(
        values=color_variable, 
        index=y_axis, 
        columns=x_axis, 
        aggfunc="mean" 
    )

    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(
        heatmap_data, 
        annot=False,
        vmin=colmin,
        vmax=colmax,
        cmap="inferno",  
        cbar_kws={"label": cbar_label}
    )

    cbar = heatmap.collections[0].colorbar
    cbar.set_label(cbar_label, fontsize=font_size, rotation=90)

    cbar.ax.tick_params(labelsize=font_size)

    plt.title(title, fontsize=font_size)
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    
    plt.xticks(rotation=45, ha="right", fontsize=font_size)
    plt.yticks(fontsize=font_size) 
    plt.tight_layout()

    return plt


def create_contour_plot(df: pd.DataFrame, x_axis: str, y_axis: str, color_variable: str, **kwargs):
    """
    Creates a contour plot from a DataFrame using specified x and y axes and a color variable,
    with black circles representing the datapoint locations.
    
    Keyword arguments (passed as **kwargs):
    - title: Title for the plot (default is f"Contour Plot of {color_variable}")
    - xlabel: Label for the x-axis (default is x_axis)
    - ylabel: Label for the y-axis (default is y_axis)
    - cbar_label: Label for the colorbar (default is color_variable)
    - fontsize: Font size for labels and ticks (default is 12)
    - levels: Number of contour levels (default is 20)
    - point_size: Size of the black circles representing data points (default is 20)
    - point_alpha: Transparency of the black circles (default is 0.6)
    """
    df[color_variable] = pd.to_numeric(df[color_variable], errors="coerce")
    
    font_size = kwargs.get("fontsize", 12)
    title = kwargs.get("title", f"Contour Plot of {color_variable}")
    xlabel = kwargs.get("xlabel", x_axis)
    ylabel = kwargs.get("ylabel", y_axis)
    cbar_label = kwargs.get("cbar_label", color_variable)
    levels = kwargs.get("levels", 20)
    point_size = kwargs.get("point_size", 20)
    point_alpha = kwargs.get("point_alpha", 0.6)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_data = df[x_axis].values
    y_data = df[y_axis].values
    z_data = df[color_variable].values
    
    x_min, x_max = x_data.min(), x_data.max()
    y_min, y_max = y_data.min(), y_data.max()
    
    xi = np.linspace(x_min, x_max, 100)
    yi = np.linspace(y_min, y_max, 100)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    zi_grid = griddata((x_data, y_data), z_data, (xi_grid, yi_grid), method='cubic')
    
    contour = ax.contourf(xi_grid, yi_grid, zi_grid, levels=levels, cmap='inferno',
                         extent=[x_min, x_max, y_min, y_max])
    
    ax.scatter(x_data, y_data, color='black', s=point_size, alpha=point_alpha, 
              edgecolors='white', linewidths=0.5)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(cbar_label, fontsize=font_size, rotation=90)
    cbar.ax.tick_params(labelsize=font_size)

    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    
    fig.tight_layout()
    
    return plt


def create_circle_heatmap(df: pd.DataFrame, x_axis: str, y_axis: str, color_variable: str, colmin=None, colmax=None, **kwargs):
    """
    Creates a heatmap with circular data points using a DataFrame.
    Keyword arguments (passed as **kwargs):
    - title: Title for the heatmap (default is f"Heatmap of {color_variable}")
    - xlabel: Label for the x-axis (default is x_axis)
    - ylabel: Label for the y-axis (default is y_axis)
    - cbar_label: Label for the colorbar (default is color_variable)
    - fontsize: Font size for labels and ticks (default is 12)
    """
    df[color_variable] = pd.to_numeric(df[color_variable], errors="coerce")
    
    font_size = kwargs.get("fontsize", 12)
    title = kwargs.get("title", f"Heatmap of {color_variable}")
    xlabel = kwargs.get("xlabel", x_axis)
    ylabel = kwargs.get("ylabel", y_axis)
    cbar_label = kwargs.get("cbar_label", color_variable)
    
    heatmap_data = df.pivot_table(values=color_variable, index=y_axis, columns=x_axis, aggfunc="mean")
    
    x_labels = heatmap_data.columns
    y_labels = heatmap_data.index
    
    fig, ax = plt.subplots(figsize=(12, 8))
    norm = plt.Normalize(colmin if colmin is not None else heatmap_data.min().min(), 
                         colmax if colmax is not None else heatmap_data.max().max())
    cmap = plt.get_cmap("inferno")
    
    for i, y in enumerate(y_labels):
        for j, x in enumerate(x_labels):
            value = heatmap_data.at[y, x]
            if not np.isnan(value):
                circle = plt.Circle((j, -i), 0.4, color=cmap(norm(value)))
                ax.add_patch(circle)
    
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.set_ylim(-len(y_labels) + 0.5, 0.5)
    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(-len(y_labels) + 1, 1))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=font_size)
    ax.set_yticklabels(y_labels, fontsize=font_size)
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(cbar_label, fontsize=font_size, rotation=90)
    cbar.ax.tick_params(labelsize=font_size)
    
    plt.title(title, fontsize=font_size)
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    plt.tight_layout()
    
    return plt
