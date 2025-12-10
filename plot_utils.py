from __future__ import annotations    # keeps forward-refs as strings; harmless ≤3.7
from typing import Iterable, List, Optional, Sequence
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation
plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

import numpy as np
import os
from utils import to_numpy, _format_time
# import seaborn as sns
# sns.set_theme()

import rerun as rr # type: ignore
import rerun.blueprint as rrb # type: ignore



class RerunVis:
    def __init__(
        self,
        dt,
        num_joints: int = 9,
        num_series: int = 1,
        name: str = "live_streaming",
        sliding_window: float = 5,
		init_time_ratio = 1.,
        time_ratios = np.linspace(0.2, 1.0, 9),
		timeaware_layout = False,
        simple_layout = False,
    ) -> None:
        rr.init(name, spawn=True)

        self.timeaware_layout = timeaware_layout
        self.simple_layout = simple_layout
        self.num_joints: int = num_joints
        self.num_series: int = num_series
        self.joint_paths: List[str] = [f"Joint_{i}" for i in range(num_joints)]

        self.sliding_window = sliding_window
        self.dt: float = dt
        self._t_joint: float = 0.0  # timeline for joint grid

        if timeaware_layout:
            self.keyboard_ctrl = None
            self.cur_time_ratio = init_time_ratio
            self._compute_time_ratio_dots(time_ratios)
            self._set_line_type("observed_tte_scalar", "", [255, 255, 255], 2)
            self._set_line_type("instability", "", [255, 255, 255], 2)
            self._set_line_type("instability", "upper", [255, 255, 0], 5)
        else:
            # static styling
            self._set_line_type_group("value", [255, 255, 255], 2)
            self._set_line_type_group("extra", [255, 0, 0], 2)
            self._set_line_type_group("upper", [255, 255, 0], 5)
            self._set_line_type_group("lower", [255, 255, 0], 5)
        # send merged blueprint
        self.blueprint = self._build_blueprint(sliding_window)
        rr.send_blueprint(self.blueprint)

    # ------------------------------------------------------------------
    # blueprints
    # ------------------------------------------------------------------
    def _build_joint_grid(self, win: float) -> rrb.Horizontal:
        # left column - camera
        left = rrb.Vertical(
            rrb.Spatial2DView(name="camera", origin="camera", contents=["camera"]),
            name="camera-column",
        )

        views: List[rrb.TimeSeriesView] = []
        sliding_window = rrb.VisibleTimeRange(
                            "time",
                            start=rrb.TimeRangeBoundary.cursor_relative(seconds=-win),
                            end=rrb.TimeRangeBoundary.cursor_relative(),
                        )
        for path in self.joint_paths:
            views.append(
                rrb.TimeSeriesView(
                    origin=path,
                    plot_legend=rrb.PlotLegend(visible=False),
                    time_ranges=[sliding_window],
                )
            )
        right = rrb.Grid(contents=views)
        return rrb.Horizontal(
			left, right, column_shares=[4, 3]
		)


    def _build_timeaware_simple_layout(self, win: float) -> rrb.Horizontal:
        # left column – camera
        left = rrb.Vertical(
            rrb.Spatial2DView(name="Camera", origin="camera", contents=["camera"]),
            name="camera-column",
        )

        # right column – clocks, curves
        clocks_row = rrb.Horizontal(
            rrb.TextDocumentView(name="Timer", origin="time_text"),
            rrb.Spatial2DView(
                name="Time Ratio",
                origin="time_ratio_panel",
                background=(0, 0, 0, 255),
            ),
            column_shares=[1, 2],
        )

        sliding_window = rrb.VisibleTimeRange(
                    "time",
                    start=rrb.TimeRangeBoundary.cursor_relative(seconds=-win),
                    end=rrb.TimeRangeBoundary.cursor_relative(),
                )
        
        instab_curve = rrb.TimeSeriesView(
			name="Instability", 
			origin="instability",
            plot_legend=rrb.PlotLegend(visible=False),
            time_ranges=[sliding_window],
            axis_y=rrb.ScalarAxis(range=(0., 0.8))
        )

        right = rrb.Vertical(
            clocks_row,  
			instab_curve, 
			row_shares=[1, 2], 
			name="Metrics"
        )

        return rrb.Horizontal(
			left, 
			right, 
			column_shares=[0, 1]
		)
    
    
    def _build_timeaware_layout(self, win: float) -> rrb.Horizontal:
        # left column – camera
        left = rrb.Vertical(
            rrb.Spatial2DView(name="camera", origin="camera", contents=["camera"]),
            name="camera-column",
        )

        # right column – clocks, curves
        clocks_row = rrb.Horizontal(
            rrb.TextDocumentView(name="observed time clock", origin="observed_tte_text"),
            rrb.TextDocumentView(name="Timer", origin="time_text"),
            rrb.Spatial2DView(
                name="speed ratio",
                origin="time_ratio_panel",
                background=(0, 0, 0, 255),
            ),
            column_shares=[1, 1, 1],
        )

        sliding_window = rrb.VisibleTimeRange(
                    "time",
                    start=rrb.TimeRangeBoundary.cursor_relative(seconds=-win),
                    end=rrb.TimeRangeBoundary.cursor_relative(),
                )
        
        tte_curve = rrb.TimeSeriesView(
            name="observed time curve", 
			origin="observed_tte_scalar",
            plot_legend=rrb.PlotLegend(visible=False),
            time_ranges=[sliding_window],
        )
        instab_curve = rrb.TimeSeriesView(
			name="instability", 
			origin="instability",
            plot_legend=rrb.PlotLegend(visible=False),
            time_ranges=[sliding_window],
        )

        right = rrb.Vertical(
            clocks_row, 
			tte_curve, 
			instab_curve, 
			row_shares=[1, 2, 2], 
			name="metrics"
        )

        return rrb.Horizontal(
			left, 
			right, 
			column_shares=[1, 1]
		)

    def _build_blueprint(self, win: float) -> rrb.Blueprint:
        if self.timeaware_layout:
            if self.simple_layout:
                layout = self._build_timeaware_simple_layout(win)
            else:
                layout = self._build_timeaware_layout(win)
        else:
            layout = self._build_joint_grid(win)
        
        return rrb.Blueprint(
			layout,
            rrb.BlueprintPanel(expanded=False),
            rrb.SelectionPanel(expanded=False),
            rrb.TimePanel(expanded=False),
            auto_space_views=False,
			collapse_panels=True,
        )

    # ------------------------------------------------------------------
    # 1) Joint grid logging
    # ------------------------------------------------------------------
    def log_joint(
        self,
        values: Iterable[float],
        lowers: Optional[Iterable[float]] = None,
        uppers: Optional[Iterable[float]] = None,
        extras: Optional[Iterable[float]] = None,
		img: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ) -> None:
        if dt is None:
            dt = self.dt
        self._t_joint += dt
        rr.set_time_seconds("time", self._t_joint)
		
        if img is not None:
            rr.log("camera", rr.Image(img))

        v = to_numpy(values)
        for idx, path in enumerate(self.joint_paths[: len(v)]):
            rr.log(f"{path}/value", rr.Scalar(v[idx]))

            if lowers is not None:
                rr.log(f"{path}/lower", rr.Scalar(to_numpy(lowers)[idx]))
            if uppers is not None:
                rr.log(f"{path}/upper", rr.Scalar(to_numpy(uppers)[idx]))
            if extras is not None and idx < len(extras):
                rr.log(f"{path}/extra", rr.Scalar(to_numpy(extras)[idx]))

    # ------------------------------------------------------------------
    # 2) Time-aware dashboard logging
    # ------------------------------------------------------------------
    def log_timeaware_step(
        self,
        *,
        img=None,
        real_tte=None,
        observed_tte=None,
		cur_time_ratio=None,
        instability=None,
        instability_lim=None,
        wall_time: Optional[float] = None,
        done = False
    ) -> None:

        if wall_time is None:
            self._t_joint += self.dt
            wall_time = self._t_joint

        rr.set_time_seconds("time", wall_time)

        # camera image
        if img is not None:
            rr.log("camera", rr.Image(img))

        # observed time-to-end curve
        if observed_tte is not None:
            rr.log("observed_tte_scalar", rr.Scalar(observed_tte))
            # digital clocks
            rr.log(
                "time_text",
                rr.TextDocument(
                    f"### Observed Remaining Time:\n# {_format_time(observed_tte)}",
                    media_type=rr.MediaType.MARKDOWN,
                ),
            )
			
        else:
            real_tte = _format_time(real_tte) if real_tte is not None else "N/A"
            rr.log(
                "time_text",
                rr.TextDocument(
                    f"### Real Left Time:\n# {real_tte}",
                    media_type=rr.MediaType.MARKDOWN,
                ),
            )

        # speed-ratio half-circle
        self._refresh_time_ratio_panel()
        self.cur_time_ratio = cur_time_ratio if cur_time_ratio is not None else self.cur_time_ratio
        self.cur_time_ratio = round(self.cur_time_ratio, 1)
        rr.log(
            "time_ratio_panel/arrow",
            rr.Arrows2D(
                origins=[[0.0, 0.0]],
                vectors=[[v*0.8 for v in self.panel_dots[self.cur_time_ratio]]],
                radii=0.05,
                colors=[[255, 255, 255, 255]],
                labels=[f"{self.cur_time_ratio:.2f}"],
            ),
        )

        # instability
        if instability is not None:
            rr.log("instability", rr.Scalar(instability))
        if instability_lim is not None:
            rr.log("instability/upper", rr.Scalar(instability_lim))

        # Reset time cursor if done signal is received
        if done:
            # Log some idle data to reset the plot
            for i in range(int(2*self.sliding_window//self.dt)):
                self._t_joint += self.dt
                rr.set_time_seconds("time", self._t_joint)
                rr.log("instability", rr.Scalar(-1))
                rr.log("instability/upper", rr.Scalar(-1))
            rr.log("instability", rr.Clear(recursive=True))
            rr.log("instability/upper", rr.Clear(recursive=True))
		
    
    # ------------------------------------------------------------------
    # Utils functions
    # ------------------------------------------------------------------
    def _compute_time_ratio_dots(self, panel_values: Sequence[float]):
        self.panel_dots = {}
        for i, pv in enumerate(panel_values):
            ang = -180 + (180 * i / (len(panel_values) - 1))
            position = [np.cos(np.radians(ang)) * 1.5, np.sin(np.radians(ang)) * 1.5]
            time_ratio = round(pv, 1)
            self.panel_dots[time_ratio] = position

    
    def _refresh_time_ratio_panel(self) -> None:
        """Register the speed ratio panel with the given values."""
        dot_positions = list(self.panel_dots.values())
        rr.log(
            "time_ratio_panel/points",
            rr.Points2D(
                positions=dot_positions,
                radii=0.1,
                colors=[[255, 255, 255, 255]],
                show_labels=False,
            ),
        )

        label_positions = [dot_positions[0], dot_positions[-1]]
        rr.log(
            "time_ratio_panel/labels",
            rr.Points2D(
                positions=label_positions,
                radii=0.,
                colors=[[255, 255, 255, 255]],
                labels=["Slow", "Fast"],
                show_labels=True,
            ),
        )

        boundary = 2
        rr.log(
            "time_ratio_panel/boundary",
            rr.Points2D(
                positions=[[-boundary, -boundary], [boundary, boundary/3]],
                radii=0.0,
                colors=[[0, 0, 0, 0]],
            ),
        )

    
    def _set_line_type_group(self, suffix: str, color: Sequence[int], width: int) -> None:
        for idx, path in enumerate(self.joint_paths):
            rr.log(
                f"{path}/{suffix}",
                rr.SeriesLine(color=color, name=f"Joint{idx}_{suffix}", width=width),
                static=True,
            )

    
    def _set_line_type(self, origin, suffix, color, width):
        rr.log(
            f"{origin}/{suffix}",
            rr.SeriesLine(color=color, name=f"{origin}_{suffix}", width=width),
            static=True
        )


class ValueVisualizer:
    def __init__(self, agent, xlim=(-0.2, 3.5), ylim=(0, 5.), vflip=False) -> None:
        self.agent = agent
        self.fig, self.ax = plt.subplots()
        self.fig.suptitle("Value Function Visualization", fontsize=16, y=0.95)  # y controls vertical position
        # Set x-axis label
        self.ax.set_xlabel("State")
        self.ax.set_ylabel("Value")
        # Set the limits
        self.xlim = xlim
        self.ylim = ylim
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        # Create a line
        self.line, = self.ax.plot([], [], label="Value Function", color="blue", linewidth=2)
        # Create a point
        self.point, = self.ax.plot([], [], 'ro', markersize=8)
        self.max_point, = self.ax.plot([], [], 'go', label="Max Value", markersize=6)
        # Create a vertical line
        self.vline = self.ax.axvline(x=0, color='r', linestyle='--', label="Goal")
        # Add legend
        self.ax.legend()
        # vertical flip
        if vflip:
            self.ax.invert_xaxis()
        

    def draw_value(self, state, scur_idx, lstm_state=None, done=None, step=0.1, s_goal=None, pause=0.01, 
                   s_name="Time"):
        """
        scur_idx: the index of the current state
        """
        curstate = []; value = []; 
        scur = state[0, scur_idx].item()
        vcur = self.agent.get_value(state)[0].item() if lstm_state is None else self.agent.get_value(state, lstm_state, done)[0].item()
        for s in np.arange(*self.xlim, step):
            state[0, scur_idx] = s # manually change the current time
            v = self.agent.get_value(state)[0].item() if lstm_state is None else self.agent.get_value(state, lstm_state, done)[0].item()
            curstate.append(s)
            value.append(v)
        max_value_idx = np.argmax(value)

        # Update the line with the new data
        self.line.set_data(curstate, value)
        # Update the point
        self.point.set_data(scur, vcur)
        self.point.set_label(f"Current {s_name}: {scur:.2f}")
        # Update the max point
        self.max_point.set_data(curstate[max_value_idx], value[max_value_idx])
        self.max_point.set_label(f"Max Value {s_name}: {curstate[max_value_idx]:.2f}")
        # Update the vertical line
        if s_goal is not None:
            self.vline.set_xdata(s_goal)
            self.vline.set_label(f"Thred {s_name}: {s_goal:.2f}")

        # Update legend
        self.ax.legend()
        # Pause for a certain number of seconds
        plt.pause(pause)



def plot_utime_dataset(utime_bins, utime_counts, save_dir=None, figsize=(10, 6), spacing_factor=0.2):
    """
    Plot the Utime distribution with improved visualization and automatic bar width adjustment
    
    Parameters:
    -----------
    utime_bins : array-like
        The bin values for the x-axis
    utime_counts : array-like
        The count values for the y-axis
    save_dir : str, optional
        Directory to save the figure
    figsize : tuple, optional
        Figure size (width, height) in inches
    spacing_factor : float, optional
        Controls the spacing between bars (0-1). Higher values create more space.
    """
    # Create figure and axis with specified size
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate dynamic bar width based on data range and number of bars
    n_bars = len(utime_bins)
    if n_bars > 1:
        # If we have multiple bars, calculate width based on their spacing
        x_range = max(utime_bins) - min(utime_bins)
        bar_width = (x_range / (n_bars - 1)) * (1 - spacing_factor)
    else:
        # Default width if there's only one bar
        bar_width = 0.8
    
    # Create bars with calculated width
    bars = ax.bar(utime_bins, utime_counts, width=bar_width, align='center', 
                 color='skyblue', edgecolor='navy', alpha=0.8)
    
    # Add count labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02*max(utime_counts),
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Customize appearance
    ax.set_title("Utime Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Utime", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set the x-axis limits a bit wider to ensure bars don't touch the edges
    x_padding = 0.5 * bar_width
    if n_bars > 1:
        ax.set_xlim(min(utime_bins) - x_padding, max(utime_bins) + x_padding)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save figure if requested
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "utime_distribution.pdf"), dpi=300, bbox_inches='tight')



def show_heatmap(arr, yticklabels=None, cmap="viridis", save_path=None, fig=None, ax=None, cbar=None):
    """
    arr: numpy array of shape (N, M, 1) or (N, M)
    """
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr2d = arr[..., 0]
    elif arr.ndim == 2:
        arr2d = arr
    else:
        raise ValueError("Expected shape (N, M, 1) or (N, M)")

    arr2d = np.concatenate([arr2d[:, :18], arr2d[:, -2:]], axis=1)  # only show the first 18 and last 2 columns
    if fig is None or ax is None:
        M, N = arr2d.shape
        ratio = N / M
        fig, ax = plt.subplots(figsize=(5*ratio, 5))
    im = ax.imshow(arr2d, cmap=cmap, aspect="auto")

    # Colorbar
    if cbar is not None:
        # update existing colorbar
        cbar.update_normal(im)
    else:
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    # Build y-ticks and separators
    groups = [
        ("CubeA Pose", 7),
        ("CubeA to CubeB Trans", 3),
        ("EEF Pose", 7),
        ("a_g", 1),
        # ("JointP", 7),
        # ("Prev Cmd JointP", 7),
        # ("Prev Delta JointP", 7),
        ("r^t", 1),
        ("t", 1),
    ]
    
    starts = np.cumsum([0] + [l for _, l in groups[:-1]])  # start index of each group
    centers = starts + np.array([l for _, l in groups]) / 2.0 - 0.5

    ax.set_xticks(centers)
    ax.set_xticklabels([name for name, _ in groups])
    if yticklabels is not None:
        ax.set_yticks(np.linspace(0, arr2d.shape[0]-1, num=len(yticklabels)))
        ax.set_yticklabels([f"{x:.2f}" for x in yticklabels])

    # Optional separators between groups
    for i, start_pos in enumerate(starts[1:-2]):
        ax.axvline(start_pos - 0.5, color="white", linewidth=2.0, alpha=1)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
    
    # plt.show()
    ax.cla()

    return fig, ax, cbar