import jax
from matplotlib.colors import LinearSegmentedColormap


class rgb:
    tue_red = (0.75, 0.0, 0.0)
    tue_blue = (0.0, 0.3, 0.6)
    tue_dark = (0.2, 0.2, 0.2)
    tue_orange = (0.9, 0.45, 0.0)
    tue_gray = (0.64, 0.64, 0.64)


cmap_wr = LinearSegmentedColormap.from_list("white_red", ["white", "red"])
cmap_wg = LinearSegmentedColormap.from_list("white_green", ["white", "green"])
cmap_wd = LinearSegmentedColormap.from_list("white_green", ["white", "black"])

cmap_bwo = LinearSegmentedColormap.from_list(
    "bwo", [rgb.tue_blue, "w", rgb.tue_orange], N=1024
)

KEY = jax.random.PRNGKey(0)
