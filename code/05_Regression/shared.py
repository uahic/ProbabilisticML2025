from matplotlib.colors import LinearSegmentedColormap

cmap_wr = LinearSegmentedColormap.from_list(
    "white_red", ["white", "red"]
)
cmap_wg = LinearSegmentedColormap.from_list(
    "white_green", ["white", "green"]
)

class rgb:
    tue_red  = (0.75, 0.0, 0.0)
    tue_blue = (0.0, 0.3, 0.6)
    tue_dark = (0.2, 0.2, 0.2)

