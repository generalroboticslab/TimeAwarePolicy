"""Small, dependency-light style shared by result reproduction plots."""

from matplotlib import pyplot as plt


FontSize = 20
AxisLabelSize = FontSize * 1.65
LegendSize = FontSize

TimeawareColor = "#3A6EA5"
VanillaColor = "#E39C37"
TimeOptimalColor = "#7B6CA8"
TimeInputColor = "#6E9F6B"
FillBlueColor = "#C7D7EB"
FillVioletColor = "#D9D2E9"
NeutralEdgeColor = "#555555"
AxisSpineColor = "#444444"
GridColor = "#999999"

BaselineLineWidth = 3.5
BarEdgeWidth = 0.5

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FontSize,
    "axes.labelsize": AxisLabelSize,
    "axes.titlesize": FontSize,
    "legend.fontsize": LegendSize,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def style_axis(axis):
    """Apply the compact paper-figure axis style."""
    axis.grid(True, alpha=0.12, linewidth=0.6, color=GridColor)
    axis.set_facecolor("white")
    axis.spines["left"].set_color(AxisSpineColor)
    axis.spines["bottom"].set_color(AxisSpineColor)
