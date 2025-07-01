"""
High-level plotting utilities used throughout the HEIMDALL project.

The functions in this module create publication-quality figures for:

* Source and station geometry (`plot_sources`)
* Graph neural-network topology (`plot_graph`, `plot_locator_map`)
* Waveform, pick and prediction diagnostics (`plot_gnn_predictions`,
  `plot_source_pdf_and_waveforms`)
* 3-D locator probabilities in Plotly (`plot_source_pdf`) and
  Matplotlib (`plot_source_pdf_images_simple`)
* End-to-end inspection flows for single windows or continuous streams
  (`plot_heimdall_flow`, `plot_heimdall_flow_continuous`)
* Miscellaneous helpers for evaluating PDF consistency
  (`extract_max_info`, )

All matplotlib plots follow a unified style set in the global
configuration at import time.
"""

import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from obspy import UTCDateTime as UTC

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from io import BytesIO

from heimdall import custom_logger as CL
logger = CL.init_logger(Path(sys.argv[0]).name, lvl="INFO")

plt.style.use('seaborn-v0_8-white')  # ggplot / seaborn-v0_8-white
plt.rcParams.update({
    'font.size': 10,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'axes.edgecolor': 'gray',
    'axes.linewidth': 0.8,
})


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


def plot_sources(sources_xyzsdrm, stations_xyz, show=False):
    """
    Plots the spatial positions of sources and stations in a 3D scatter plot.

    This function takes the XYZ coordinates of sources and stations, and optionally
    a boolean to determine if the plot should be displayed immediately. It adjusts
    the Z coordinate of sources by making it negative (to represent depth). It also
    adds a semi-transparent gray plane at z=0 to serve as a reference depth.

    Args:
        sources_xyzsdrm (np.ndarray): An array of shape (N, M) where N is the number
            of sources and M is at least 3 (for XYZ coordinates). Additional columns
            are allowed but ignored.
        stations_xyz (np.ndarray): An array of shape (K, 3) where K is the number
            of stations, representing their XYZ coordinates.
        show (bool, optional): If True, display the plot immediately. Defaults to False.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    assert isinstance(stations_xyz, np.ndarray)
    assert isinstance(sources_xyzsdrm, np.ndarray)
    sources_xyzsdrm[:, 2] = sources_xyzsdrm[:, 2] * -1  # negative depth
    _coords_list = np.concatenate(
                        (sources_xyzsdrm[:, :3], stations_xyz[:, :3]), axis=0)
    xmin, xmax = np.min(_coords_list[:, 0]), np.max(_coords_list[:, 0])
    xlim_min = xmin - 0.05
    xlim_max = xmax + 0.05

    ymin, ymax = np.min(_coords_list[:, 1]), np.max(_coords_list[:, 1])
    ylim_min = ymin - 0.01
    ylim_max = ymax + 0.01

    zmin, zmax = np.min(_coords_list[:, 2]), np.max(_coords_list[:, 2])
    zlim_min = zmin - 1
    zlim_max = zmax + 0.2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sources_xyzsdrm[:, 0],
               sources_xyzsdrm[:, 1],
               sources_xyzsdrm[:, 2], c='teal',
               marker='o', label='sources')
    ax.scatter(stations_xyz[:, 0],
               stations_xyz[:, 1],
               stations_xyz[:, 2]+3000, s=70, c='orange',
               marker='^', label='stations')

    # Add a semi-transparent gray plane at a fixed depth
    fixed_depth = 0  # Set the desired depth
    xx, yy = np.meshgrid(
                    np.arange(xlim_min, xlim_max+0.01, 0.01),
                    np.arange(ylim_min, ylim_max+0.01, 0.01))
    zz = fixed_depth * np.ones_like(xx)
    ax.plot_surface(xx, yy, zz, color='darkgray', alpha=0.01)

    # Set axis limits / labels
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_zlim(zlim_min, zlim_max)
    ax.set_xlabel('longitude (dec.deg)')
    ax.set_ylabel('latitude (dec.deg)')
    ax.set_zlabel('depth (m)')
    #
    ax.legend()
    if show:
        plt.show()
    return fig


def plot_graph(nodes, edges, clusters=None, fig_title="Graph Network",
               show=False, store=False, names={}, store_name="GNN.pdf"):
    """
    Plots a graph with nodes and edges, optionally displaying cluster information,
    and can save the figure to a file.

    The function visualizes a graph where nodes and edges are given by their
    coordinates and connectivity. Nodes can be colored according to the cluster
    they belong to, and the graph can be annotated with names.

    Args:
        nodes (np.ndarray): An array of shape (N, 2) containing the XY coordinates
            of the nodes.
        edges (np.ndarray): An array of shape (2, M) where each column represents
            an edge as a pair of node indices.
        clusters (np.ndarray, optional): An array of shape (N,) where each element
            is an integer indicating the cluster of the corresponding node. If None,
            all nodes are considered as part of a single cluster. Defaults to None.
        fig_title (str, optional): Title of the graph. Defaults to "Graph Network".
        show (bool, optional): If True, display the plot immediately. Defaults to False.
        store (bool, optional): If True, save the plot to a file. Defaults to False.
        names (dict, optional): A dictionary where keys are node labels and values
            are indices in the `nodes` array. Defaults to {}.
        store_name (str, optional): The filename where the figure should be saved if
            `store` is True. Defaults to "GNN.pdf".

    Returns:
        matplotlib.figure.Figure: The figure object containing the graph plot.
    """

    # All ICELAND
    lat_range = [63, 67]
    lon_range = [-25, -13]

    # Investigation AREA
    lat_range = [63.8, 64.25]
    lon_range = [-22, -20.7]

    X = nodes[:, 0:2]
    fig = plt.figure(figsize=(8, 5))  # Adjust the figure size as needed

    # --- Plot edges
    assert edges.shape[1] == 2
    for edge in edges:
        node1 = nodes[edge[0]]
        node2 = nodes[edge[1]]
        plt.plot([node1[0], node2[0]], [node1[1], node2[1]],
                 color="darkgray", linestyle="-",
                 linewidth=0.3, zorder=-1)  #, alpha=0.7)

    # --- Plot Nodes
    if isinstance(clusters, (list, tuple, np.ndarray)):
        unique_clusters = np.unique(clusters)
        for cluster in unique_clusters:
            if cluster == -1:
                # Plot noise points in black
                plt.scatter(X[clusters == cluster, 0], X[clusters == cluster, 1],
                            c='red', marker="*", label='Noise', s=40)
            else:
                plt.scatter(X[clusters == cluster, 0], X[clusters == cluster, 1],
                            s=30, label=f'Cluster {cluster}')
    else:
        plt.scatter(nodes[:, 0], nodes[:, 1],
                    s=30, label="nodes")

    # --- Names
    if names:
        for kk, vv in names.items():
            plt.text(nodes[vv, 0].item(), nodes[vv, 1].item(), kk,
                     fontsize=11, color='red', ha='center', va='center')

    # --- Annotating
    plt.title(fig_title)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.xlim(lon_range)
    plt.ylim(lat_range)
    plt.legend()

    # --- Closing
    plt.tight_layout()
    if store:
        fig.savefig(store_name)
    if show:
        plt.show()
    #
    return fig


def plot_locator_map(eqcoord, gnn, picks, plot_edges=True, show=False):
    """Visualise triggered and idle stations together with an event location.

    Args:
        eqcoord (Sequence[float]): Tuple ``(lon, lat, dep, pdf_val)`` returned
            by the locator.  Depth is in kilometres.
        gnn (dict): Serialized GNN object containing the keys
            ``{"nodes", "edges", "stations_order", "stations_coordinate"}``.
        picks (list): List of ``(station_idx, windows)`` pairs where *windows*
            is the list of picked intervals for that station.
        plot_edges (bool, optional): Draw the inter-station graph edges.
            Defaults to ``True``.
        show (bool, optional): Immediately display the figure with
            ``plt.show()``.  Defaults to ``False``.

    Returns:
        tuple: ``(fig, ax)`` – the Matplotlib figure and its main axis.
    """
    stations_dict = gnn['stations_order'].item()
    stations_coord_dict = gnn['stations_coordinate'].item()
    nodes, edges = gnn["nodes"], gnn["edges"]

    # Iterate through picks to populate the sets
    coordinates_with_observations, coordinates_without_observations = [], []
    for (station_index, observations) in picks:
        # Get station name
        _name = [kk for kk, vv in stations_dict.items() if vv==station_index]
        assert len(_name) == 1
        _name = _name[0]
        # Get station coord
        _coord = stations_coord_dict[_name]

        if observations:
            coordinates_with_observations.append(_coord)
        else:
            coordinates_without_observations.append(_coord)
    #
    coordinates_with_observations_mat = np.array(coordinates_with_observations)
    coordinates_without_observations_mat = np.array(coordinates_without_observations)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # --> Edges
    if len(edges) > 0:
        try:
            assert edges.shape[1] == 2
        except AssertionError:
            edges = edges.T
        #
        for edge in edges:
            node1 = nodes[edge[0]]
            node2 = nodes[edge[1]]
            plt.plot([node1[0], node2[0]], [node1[1], node2[1]],
                     color="darkgray", linestyle="-",
                     linewidth=0.3, zorder=-1)  #, alpha=0.7)

    # --> NO obs
    if len(coordinates_without_observations_mat):
        ax.scatter(
            coordinates_without_observations_mat[:, 0],
            coordinates_without_observations_mat[:, 1],
            s=40, marker="o", edgecolor="black", facecolor="white",
            zorder=1, label="stations")

    # --> YES obs
    if len(coordinates_with_observations_mat) > 0:
        ax.scatter(coordinates_with_observations_mat[:, 0],
                   coordinates_with_observations_mat[:, 1],
                   s=55, marker="^", edgecolor="black", facecolor="black",
                   zorder=2, label="triggered stations")

    # --> EQARTHQUAKE
    ax.scatter([eqcoord[0],], [eqcoord[1],], s=100, marker="*",
               edgecolor="black", facecolor="orange",
               label="earthquake", zorder=3)

    # beauty
    ax.set_xlabel("Longitude (dec.deg)")
    ax.set_ylabel("Latitude (dec.deg)")
    ax.set_title("Lon: %.5f - Lat: %.5f - Dep: %.1f / PDF: %0.2f" % (
                eqcoord[0], eqcoord[1], eqcoord[2], eqcoord[-1]
            ))
    plt.legend()
    plt.tight_layout()

    if show:
        plt.show()

    return (fig, ax)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def plot_gnn_predictions(trX, trY, trR, figtitle, stations_dict,
                         pred=None, detect=None, detect_bound=None,
                         rows_per_figure=5, store_dir="./outfigures",
                         use_batch=None, order=None,
                         scale_zero_to_one=True, show=False):
    """Plot CF stacks, real waveforms, labels, predictions and detections.

    The routine creates one or more multi-panel figures, each covering
    *rows_per_figure* stations.  Figures are stored to *store_dir* as PNG
    **and** PDF.

    Args:
        trX (np.ndarray): CF tensor with shape
            ``(batch, station, cf_chan, npts)``
            **or** ``(station, cf_chan, npts)``.
        trY (np.ndarray): Label tensor with the same trailing dimensions as
            *trX*.
        trR (np.ndarray): Real waveform tensor with shape compatible to
            *trX*.
        figtitle (str): Base title and base filename for the figures.
        stations_dict (dict): Mapping ``{station_name: station_idx}``.
        pred (np.ndarray, optional): Model predictions with the same shape as
            *trY*.  Defaults to ``None``.
        detect (list | None): Per-station list of picked intervals
            ``[(start, end), ...]``.  Defaults to ``None``.
        detect_bound (np.ndarray | None): Two-column array with the lower and
            upper detection threshold indices for each station.  Defaults to
            ``None``.
        rows_per_figure (int, optional): Number of stations per figure.
            Defaults to ``5``.
        store_dir (str | Path, optional): Output directory.  Created if it
            does not exist.  Defaults to ``"./outfigures"``.
        use_batch (int | None): Batch index to visualise when *trX* is 4-D.
            Required in that case.
        order (dict | None): Custom plotting order
            ``{station_name: rank}``.  Defaults to alphabetical order.
        scale_zero_to_one (bool, optional): Normalise each CF to ``[0, 1]``.
            Defaults to ``True``.
        show (bool, optional): Call ``plt.show()`` for every figure.
            Defaults to ``False``.

    Returns:
        list[matplotlib.figure.Figure]: List of created figure objects.
    """

    order = order or {name: i for i, name in enumerate(stations_dict)}
    sorted_indices = [stations_dict[name] for name in sorted(stations_dict, key=order.get)]
    sorted_indices_ylabel = [name for name in sorted(stations_dict, key=order.get)]

    Path(store_dir).mkdir(parents=True, exist_ok=True)

    if len(trX.shape) == 4:
        if use_batch is None:
            logger.error("Specify use_batch when trX has a batch dimension.")
            sys.exit()
        indices = (use_batch, sorted_indices, slice(None), slice(None))
        suffix = '_batch'
    else:
        indices = (sorted_indices, slice(None), slice(None))
        suffix = ''

    trX, trY, trR = trX[indices], trY[indices], trR[indices]
    if pred is not None:
        trP = pred[indices]
    if detect is not None:
        trD = [detect[idx] for idx in sorted_indices]
    if detect_bound is not None:
        trDidx = detect_bound[use_batch, sorted_indices, :, :] if len(detect_bound.shape) == 4 else [detect_bound[idx] for idx in sorted_indices]

    nrows = trX.shape[0]
    num_figures = (nrows - 1) // rows_per_figure + 1
    figs_list = []

    for fig_num in range(1, num_figures + 1):
        start_idx = (fig_num - 1) * rows_per_figure
        end_idx = start_idx + rows_per_figure
        rows_in_this_fig = trX[start_idx:end_idx].shape[0]

        fig, axs = plt.subplots(rows_in_this_fig, 1, figsize=(10, 2 * rows_in_this_fig), squeeze=False)
        plt.subplots_adjust(hspace=0)

        for i, ax in enumerate(axs.flatten()):
            idx = start_idx + i
            ax.plot(trR[idx][1] + trR[idx][2], color="darkgray", alpha=0.6, label="real_NE")
            ax.plot(trR[idx][0], color="black", alpha=0.6, label="real_Z")

            lf_ax2 = ax.twinx()
            for _cfs in range(trX.shape[1]):
                if scale_zero_to_one:
                    lf_ax2.plot((trX[idx][_cfs]+10**-9)/np.max(trX[idx][_cfs]),
                                alpha=0.7, color=f"C{_cfs}", label=f"CF_{_cfs}")
                else:
                    lf_ax2.plot(trX[idx][_cfs], alpha=0.7,
                                color=f"C{_cfs}", label=f"CF_{_cfs}")
            #
            lf_ax2.plot(trY[idx][0], alpha=0.7, label="Y", ls="--", color="#ff8099")
            try:
                lf_ax2.plot(trY[idx][1], alpha=0.7, label="P", ls="--")
                lf_ax2.plot(trY[idx][2], alpha=0.7, label="S", ls="--")
            except:
                pass

            if pred is not None:
                lf_ax2.plot(trP[idx][0], alpha=0.7, label="prediction", color="purple")
            if detect is not None:
                plt.axhline(y=0.5, color='gray', linestyle='--')
                for _daje, (_ss, _ee) in enumerate(trD[idx]):
                    lf_ax2.axvspan(_ss, _ee, color='gold', alpha=0.3, label='detections' if _daje == 0 else '')

            if detect_bound is not None:
                lf_ax2.axvline(trDidx[idx][0], color='darkgray')
                lf_ax2.axvline(trDidx[idx][1], color='darkgray')

            lf_ax2.set_ylim([-0.2, 1.2])
            ax.set_ylabel(sorted_indices_ylabel[idx])

            if i == 0:
                ax.legend(loc='upper left')
                lf_ax2.legend(loc='upper right')

        fig.suptitle(f"{figtitle}  Fig: {fig_num}/{num_figures}", fontsize=16)
        fig.savefig(f"{store_dir}/{figtitle}_Fig_{fig_num}{suffix}.png",
                    bbox_inches='tight', dpi=310)
        fig.savefig(f"{store_dir}/{figtitle}_Fig_{fig_num}{suffix}.pdf",
                    bbox_inches='tight')
        figs_list.append(fig)
        if show:
            plt.show()
        plt.close(fig)

    return figs_list


def plot_source_pdf_images_simple(
        x_grid, y_grid, z_grid,
        xy_image, xz_image, yz_image,
        reference_locations=None, figtitle=None):
    """Quick Matplotlib visualisation of PDF projections.

    Args:
        x_grid (np.ndarray): 1-D X-coordinate grid (metres).
        y_grid (np.ndarray): 1-D Y-coordinate grid (metres).
        z_grid (np.ndarray): 1-D Z-coordinate grid (metres).
        xy_image (np.ndarray): 2-D XY slice of the PDF (shape
            ``len(x_grid) × len(y_grid)``).
        xz_image (np.ndarray): 2-D XZ slice of the PDF.
        yz_image (np.ndarray): 2-D YZ slice of the PDF.
        reference_locations (Sequence[Sequence[float]] | None):
            Optional list of reference points in **grid** coordinates
            ``[(x, y, z), ...]``.  Defaults to ``None``.
        figtitle (str | None): Overall figure title.

    Returns:
        tuple: ``(fig, axs)`` where *axs* is the ``(1, 3)`` ndarray of axes.
    """

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Configuration for plots
    projections = [
        (axs[0], x_grid, y_grid, xy_image, 'XY Projection', 'X', 'Y', (0, 1)),
        (axs[1], x_grid, z_grid, xz_image, 'XZ Projection', 'X', 'Z', (0, 2)),
        (axs[2], y_grid, z_grid, yz_image, 'YZ Projection', 'Y', 'Z', (1, 2)),
    ]

    # Plot each projection
    for _de, (ax, x, y, image, title, xlabel, ylabel, ref_indices) in enumerate(projections):
        ax.pcolormesh(x, y, image.T, cmap='viridis', vmin=0.0, vmax=1.0,
                      edgecolors='none', rasterized=True)  # shading='gouraud',)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if _de in (1, 2):
            ax.invert_yaxis()
        # Plot reference locations if provided
        if reference_locations:
            ref_x, ref_y = ref_indices
            ax.scatter(
                [loc[ref_x] for loc in reference_locations],
                [loc[ref_y] for loc in reference_locations],
                facecolor='darkred',
                edgecolor='black',
                marker='o',
                s=20
            )

    # Add a colorbar
    im3 = axs[2].collections[0]  # Reusing the last plot's collection for the colorbar
    cbar = fig.colorbar(im3, ax=axs, orientation='horizontal',
                        fraction=0.03, pad=0.1)
    cbar.ax.set_position([0.2, 0.1, 0.6, 0.02])  # [left, bottom, width, height]

    # Add title if provided
    if figtitle:
        fig.suptitle(figtitle, fontsize=16)

    return (fig, axs)


def plot_source_pdf(x_grid, y_grid, z_grid, sources_pdfs,
                    isosurface_values=[0.9, ], reference_locations=None,  # XYZ
                    show=False, title=None):

    """
    Plots the probability density function (PDF) of sources using a 3D isosurface
    and 2D heatmap slices.

    This function visualizes the source PDF in both 3D and 2D perspectives. The 3D
    view shows isosurfaces for specified values, and the 2D views show heatmaps for
    the XY, XZ, and YZ projections of the PDF.

    Args:
        x_grid (np.ndarray): 1D array specifying the grid points along the X axis.
        y_grid (np.ndarray): 1D array specifying the grid points along the Y axis.
        z_grid (np.ndarray): 1D array specifying the grid points along the Z axis.
        sources_pdfs (np.ndarray): A 3D array of shape (len(x_grid), len(y_grid), len(z_grid))
            representing the PDF values at each point in the grid.
        isosurface_values (list, optional): Values at which isosurfaces should be drawn.
            Defaults to [0.9].
        reference_location (list, optional): A list / tuple of 3D point (X, Y, Z)
            indicating a reference location to be marked on the plot. Defaults to None.
        show (bool, optional): If True, display the plot immediately. Defaults to False.
        title (str, optional): Title of the plot. If None, a default title is used.
            Defaults to None.

    Returns:
        plotly.graph_objs.Figure: The figure object containing the 3D isosurface and 2D slices.
    """
    reference_locations_colors = ["darkred", "white", "black"]
    fig = make_subplots(rows=1, cols=4,
                        # column_widths=[0.4, 0.2, 0.2, 0.2],
                        specs=[[{'type': 'surface'}, {'type': 'heatmap'},
                                {'type': 'heatmap'}, {'type': 'heatmap'}]])

    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
    for value in isosurface_values:
        fig.add_trace(
            go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=sources_pdfs.flatten(),
                isomin=value,
                isomax=value,
                colorscale=[[0, "darkred"], [1, "darkred"]],  # Fixed color for the isosurface
                caps=dict(x_show=False, y_show=False, z_show=False),
                showscale=False  # Hide the color scale
            ), row=1, col=1
        )

        for _rloc, reference_location in enumerate(reference_locations):
            if isinstance(reference_location, (np.ndarray, list, tuple)):
                fig.add_trace(
                    go.Scatter3d(
                        x=[reference_location[-3]],
                        y=[reference_location[-2]],
                        z=[reference_location[-1]],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=reference_locations_colors[_rloc],
                            line=dict(
                                width=5,  # Adjust the width of the outline as needed --> not working!
                                color='black'
                            ),
                            symbol='circle',
                            opacity=1
                        ),
                    ),
                    row=1, col=1
                )

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    xy_sum = np.sum(sources_pdfs, axis=2)  # Sum over the Z-axis to get the XY plane
    xz_sum = np.sum(sources_pdfs, axis=1)  # Sum over the Y-axis to get the XZ plane
    yz_sum = np.sum(sources_pdfs, axis=0)  # Sum over the X-axis to get the YZ plane
    xy_sum /= np.max(xy_sum)
    xz_sum /= np.max(xz_sum)
    yz_sum /= np.max(yz_sum)

    for _col, _mat, (xgr, ygr), (label_x, label_y) in zip(
                (2, 3, 4),
                (xy_sum.T, xz_sum.T, yz_sum.T),
                ((x_grid, y_grid), (x_grid, z_grid), (y_grid, z_grid)),
                (('X (m)', 'Y (m)'),
                 ('X (m)', 'Z (m)'),
                 ('Y (m)', 'Z (m)'))):

        fig.add_trace(go.Heatmap(x=xgr, y=ygr, z=_mat,
                                 colorscale='Viridis', showscale=True if _col == 4 else False),
                      row=1, col=_col)

        for iso_val in isosurface_values:
            fig.add_trace(go.Contour(
                x=xgr, y=ygr, z=_mat,
                showscale=False,  # Hide color scale for contours
                contours=dict(
                    type='constraint',  # Use 'constraint' to specify exact values
                    value=iso_val,  # Specify the exact value for contour
                    showlabels=True  # Shows labels on contour lines
                ),
                line=dict(
                    color='darkred',  # Set contour line color
                    width=2,  # Set contour line width
                ),
            ), row=1, col=_col)

        for _rloc, reference_location in enumerate(reference_locations):
            if isinstance(reference_location, (np.ndarray, list, tuple)):
                if _col == 2:
                    idx_X, idx_Y = -3, -2
                elif _col == 3:
                    idx_X, idx_Y = -3, -1
                elif _col == 4:
                    idx_X, idx_Y = -2, -1
                else:
                    raise ValueError("Unkown error!")

                fig.add_trace(
                    go.Scatter(
                        x=[reference_location[idx_X],],
                        y=[reference_location[idx_Y],],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=reference_locations_colors[_rloc],
                            line=dict(
                                width=1.5,
                                color='black'
                            ),
                            symbol='circle',
                            opacity=1
                        ),
                    ),
                    row=1, col=_col
                )

        fig.update_xaxes(title=label_x, range=[np.min(xgr), np.max(xgr)],
                         row=1, col=_col)
        fig.update_yaxes(title=label_y, range=[np.min(ygr), np.max(ygr)],
                         row=1, col=_col)

        if _col in (3, 4):
            fig.update_yaxes(autorange="reversed", row=1, col=_col)

    # ====================================================================
    # ====================================================  FINAL UPDATES

    # Update the layout to specify titles and reverse the z-axis
    fig.update_layout(
        title=title if title else 'Isosurface with 2D Slices and Contours',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            zaxis=dict(autorange='reversed'),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.25, y=-2.1, z=1.5)  # Use different values to change view angle
            )
        ),
        width=1700,
        height=455,
        #
        showlegend=False,
    )

    if show:
        fig.show()
    return fig


def plot_source_pdf_and_waveforms(
            plotly_fig, trX, trR, figtitle, stations_dict,
            pred=None, detect=None, detect_bound=None,
            rows_per_figure=5, store_dir="./outfigures",
            use_batch=None, order=None, show=False):

    img_bytes = pio.to_image(plotly_fig, format='png')
    img = Image.open(BytesIO(img_bytes))
    nrows = rows_per_figure
    #
    order = order or {name: i for i, name in enumerate(stations_dict) if i < nrows}
    if len(order) != len(stations_dict):
        _last_idx_ = np.max([_v for _v in order.values()])
        for key, value in stations_dict.items():
            if key not in order.keys():
                _last_idx_ = _last_idx_+1
                order[key] = _last_idx_
    sorted_indices = [stations_dict[name] for name in sorted(stations_dict, key=order.get)]
    sorted_indices_ylabel = [name for name in sorted(stations_dict, key=order.get)]

    Path(store_dir).mkdir(parents=True, exist_ok=True)

    if len(trX.shape) == 4:
        if use_batch is None:
            logger.error("Specify use_batch when try has a batch dimension.")
            sys.exit()
        indices = (use_batch, sorted_indices, slice(None), slice(None))
    else:
        indices = (sorted_indices, slice(None), slice(None))

    trX, trR = trX[indices], trR[indices]
    if pred is not None:
        trP = pred[indices]
    if detect is not None:
        trD = [detect[idx] for idx in sorted_indices]
    if detect_bound is not None:
        trDidx = detect_bound[use_batch, sorted_indices, :, :] if len(detect_bound.shape) == 4 else [detect_bound[idx] for idx in sorted_indices]

    fig, axs = plt.subplots(
                        nrows+1, 1,
                        figsize=(7, 10), squeeze=False)
    axs_flat = axs.flatten()
    plt.subplots_adjust(hspace=0)

    # =========================  SET UP PLOTLY IMAGE
    axs_flat[0].imshow(img, aspect='auto',
                       # extent=(-0.1, 1.1, -0.1, 1.1),
                       extent=(-0.05, 1.05, -0.05, 1.05),
                       transform=axs_flat[0].transAxes)  # Enlarge image beyond the subplot box
    axs_flat[0].axis('off')  # Turn off axis

    # select only the first nrows with PICKS (both P+S)
    # plot_idx = [ii for ii in indices if np.max(trX[ii][0]) >= 0.9][:nrows]

    for idx, ax in enumerate(axs_flat[1:]):
        ax.plot(trR[idx][1] + trR[idx][2], color="darkgray", alpha=0.6, label="real_NE")
        ax.plot(trR[idx][0], color="black", alpha=0.6, label="real_Z")
        lf_ax2 = ax.twinx()
        lf_ax2.plot(trX[idx][0], alpha=0.7, label="Y", ls="--", color="#ff8099")
        if pred is not None:
            lf_ax2.plot(trP[idx][0], alpha=0.7, label="prediction", color="purple")
        if detect is not None:
            plt.axhline(y=0.5, color='gray', linestyle='--')
            for _daje, (_ss, _ee) in enumerate(trD[idx]):
                lf_ax2.axvspan(_ss, _ee, color='gold', alpha=0.3, label='detections' if _daje == 0 else '')

        if detect_bound is not None:
            lf_ax2.axvline(trDidx[idx][0], color='darkgray')
            lf_ax2.axvline(trDidx[idx][1], color='darkgray')

        lf_ax2.set_ylim([-0.2, 1.2])
        ax.set_ylabel(sorted_indices_ylabel[idx])

        if idx == 0:
            ax.legend(loc='upper left')
            lf_ax2.legend(loc='upper right')

    fig.suptitle(f"{figtitle}", fontsize=16)
    fig.savefig(f"{store_dir}/{figtitle}.png",
                bbox_inches='tight', dpi=310)
    if show:
        plt.show()
    return fig


def plot_images(xgr, ygr, zgr, matrix_list1, matrix_list2,
                matrix_list3, figtitle="LOCATOR - TRAINING"):
    """Create a 3x3 panel comparing label, prediction and delta images.

    Args:
        xgr (np.ndarray): X grid (metres or kilometres – labels added by caller).
        ygr (np.ndarray): Y grid.
        zgr (np.ndarray): Z grid.
        matrix_list1 (Sequence[np.ndarray]): Three 2-D arrays (XY, XZ, YZ)
            with the target labels.
        matrix_list2 (Sequence[np.ndarray]): Corresponding prediction arrays.
        matrix_list3 (Sequence[np.ndarray]): Element-wise difference
            ``|label − prediction|``.
        figtitle (str, optional): Figure title.  Defaults to
            ``"LOCATOR - TRAINING"``.

    Returns:
        matplotlib.figure.Figure: The assembled figure.
    """

    assert len(matrix_list1) == 3
    assert len(matrix_list2) == 3
    assert len(matrix_list3) == 3

    fig = plt.figure(figsize=(8, 8))
    labels = (("X (km)", "Y (km)"),
              ("X (km)", "Z (km)"),
              ("Y (km)", "Z (km)"))
    titles = (("Labels"),
              ("Predictions"),
              ("Delta"))
    axes = ((xgr, ygr),
            (xgr, zgr),
            (ygr, zgr))

    # Plot matrix_list1, matrix_list2, and matrix_list3 in a 3x3 grid
    for i in range(3):
        # First set (matrix_list1) on row 1, column 1
        ax = fig.add_subplot(3, 3, i * 3 + 1)  # i*3 + 1 -> plots in the first column
        im = ax.pcolormesh(axes[i][0], axes[i][1], matrix_list1[i].T,
                           cmap='viridis', vmin=0.0, vmax=1.0,
                           edgecolors='none', rasterized=True)  # shading='gouraud',
        ax.set_xlabel(labels[i][0])
        ax.set_ylabel(labels[i][1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.07)
        if i == 0:
            ax.set_title(titles[0])
        if i in (1, 2):
            ax.invert_yaxis()

        # Second set (matrix_list2) on row 1, column 2
        ax = fig.add_subplot(3, 3, i * 3 + 2)  # i*3 + 2 -> plots in the second column
        im = ax.pcolormesh(axes[i][0], axes[i][1], matrix_list2[i].T,
                           cmap='viridis', vmin=0.0, vmax=1.0,
                           edgecolors='none', rasterized=True)  # shading='gouraud',)
        ax.set_xlabel(labels[i][0])
        ax.set_ylabel(labels[i][1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.07)
        if i == 0:
            ax.set_title(titles[1])
        if i in (1, 2):
            ax.invert_yaxis()

        # Third set (matrix_list3) on row 1, column 3
        ax = fig.add_subplot(3, 3, i * 3 + 3)  # i*3 + 3 -> plots in the third column
        im = ax.pcolormesh(axes[i][0], axes[i][1], matrix_list3[i].T,
                           cmap='gray', vmin=0.0, vmax=1.0,
                           edgecolors='none', rasterized=True)  # shading='gouraud',)
        ax.set_xlabel(labels[i][0])
        ax.set_ylabel(labels[i][1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.07)
        if i == 0:
            ax.set_title(titles[2])
        if i in (1, 2):
            ax.invert_yaxis()

    if figtitle:
        fig.suptitle(figtitle, fontsize=16)

    return fig


def plot_images_with_inputs(
                tsX, tsR, xgr, ygr, zgr,
                matrix_list1, matrix_list2, matrix_list3,
                reference_locations=[],
                figtitle="LOCATOR - TRAINING"):
    assert len(matrix_list1) == 3
    assert len(matrix_list2) == 3
    assert len(matrix_list3) == 3

    reference_locations_colors = ("darkred", "white", "darkblue")

    fig = plt.figure(figsize=(36 / 2.54, 24 / 2.54))  # 1cm = 2.54 inch.
    gs = fig.add_gridspec(6, 9, hspace=0.8, wspace=1)

    labels = (("X (km)", "Y (km)"),
              ("X (km)", "Z (km)"),
              ("Y (km)", "Z (km)"))
    titles = (("Labels"),
              ("Predictions"),
              ("Delta"))
    axes = ((xgr, ygr),
            (xgr, zgr),
            (ygr, zgr))
    ref_loc_idx = [(0, 1),
                   (0, 2),
                   (1, 2)]

    # Plot time series (rectangles) on rows 0-2, spanning 1 row and 3 columns each
    for i in range(6):
        ax = fig.add_subplot(gs[i, :3])  # Spans 1 row and 3 columns

        ts_y = tsX[i, 0, :]

        ts_r_z = tsR[i, 0, :]
        ts_r_z /= np.max(ts_r_z)

        ts_r_ne = tsR[i, 1, :] + tsR[i, 2, :]
        ts_r_ne /= np.max(ts_r_ne)

        ax.plot(ts_r_ne, label='NE', color="darkgray", alpha=0.8)
        ax.plot(ts_r_z, label='Z', color="black", alpha=0.8)
        ax.plot(ts_y, label='Y', ls="-", color="darkred", alpha=0.8)
        ax.set_ylim([-0.2, 1.2])

        if i == 0:
            ax.set_title("Input Waveforms")

    # Plot matrix_list1, matrix_list2, and matrix_list3 in a 3x3 grid
    preds_max = []
    for i in range(3):
        # First set (matrix_list1) on row 1, column 1
        axl = fig.add_subplot(gs[i*2:i*2+2, 3:5])
        im = axl.pcolormesh(axes[i][0], axes[i][1], matrix_list1[i].T,
                            cmap='viridis', vmin=0.0, vmax=1.0,
                            edgecolors='none', rasterized=True)  # shading='gouraud',)
        axl.set_xlabel(labels[i][0])
        axl.set_ylabel(labels[i][1])
        axl.spines['top'].set_visible(False)
        axl.spines['right'].set_visible(False)
        if i == 0:
            axl.set_title(titles[0])
        if i in (1, 2):
            axl.invert_yaxis()
        if i == 2:
            fig.colorbar(im, ax=axl, orientation='horizontal')  # fraction=0.05, pad=0.1)

        # Second set (matrix_list2) on row 1, column 2
        axp = fig.add_subplot(gs[i*2:i*2+2, 5:7])
        preds_max.append(np.nanmax(matrix_list2[i]))
        im = axp.pcolormesh(axes[i][0], axes[i][1], matrix_list2[i].T,
                            cmap='viridis', vmin=0.0, vmax=1.0,
                            edgecolors='none', rasterized=True)  # shading='gouraud',)
        axp.set_xlabel(labels[i][0])
        axp.set_ylabel(labels[i][1])
        axp.spines['top'].set_visible(False)
        axp.spines['right'].set_visible(False)
        if i == 0:
            axp.set_title(titles[1])
        if i in (1, 2):
            axp.invert_yaxis()
        if i == 2:
            fig.colorbar(im, ax=axp, orientation='horizontal')  # fraction=0.05, pad=0.2)

        # Third set (matrix_list3) on row 1, column 3
        axd = fig.add_subplot(gs[i*2:i*2+2, 7:9])
        im = axd.pcolormesh(axes[i][0], axes[i][1], matrix_list3[i].T,
                            #cmap='gray', vmin=0.0, vmax=1.0,
                            cmap='viridis', vmin=0.0, vmax=1.0,
                            edgecolors='none', rasterized=True)  # shading='gouraud',)
        axd.set_xlabel(labels[i][0])
        axd.set_ylabel(labels[i][1])
        axd.spines['top'].set_visible(False)
        axd.spines['right'].set_visible(False)
        if i == 0:
            axd.set_title(titles[2])
        if i in (1, 2):
            axd.invert_yaxis()
        if i == 2:
            fig.colorbar(im, ax=axd, orientation='horizontal')  # fraction=0.05, pad=0.3)

        if reference_locations:
            for _ii, loc in enumerate(reference_locations):
                for _ax in (axl, axp, axd):
                    _ax.scatter(
                        loc[ref_loc_idx[i][0]], loc[ref_loc_idx[i][1]],
                        facecolor=reference_locations_colors[_ii],
                        edgecolor='black',
                        marker='o',
                        s=20)

    # Add title if provided
    if figtitle:
        fig.suptitle(figtitle+(
                " [%.02f - %.02f - %.02f]" % (preds_max[0], preds_max[1], preds_max[2])
            ), fontsize=16)

    return fig


def plot_heimdall_flow(
            window_x, window_y, window_r, window_pred,
            image_xy, image_xz, image_yz, grid, verdict,
            stations=[], reference_locations=[],
            store_dir="HeimdallResults"):
    """Create a per-window diagnostic figure for inference snapshots.

    The layout combines six stations on the left and the
    three PDF projections on the right.

    Args:
        window_x (np.ndarray): CF stack ``(win, sta, cf_chan, npts)``.
        window_y (np.ndarray): Labels with the same trailing dimensions.
        window_r (np.ndarray): Real waveform tensor.
        window_pred (np.ndarray): Model predictions.
        image_xy (np.ndarray): XY PDF slices ``(win, x, y)``.
        image_xz (np.ndarray): XZ PDF slices.
        image_yz (np.ndarray): YZ PDF slices.
        grid (tuple[np.ndarray, np.ndarray, np.ndarray]): The locator grid
            ``(x, y, z)`` in metres.
        verdict (np.ndarray): Model confidence per window (0–1).
        stations (Sequence[Sequence[float]], optional): Station ENU
            coordinates to overlay.  Defaults to empty.
        reference_locations (Sequence[Sequence[float]] | None): Reference
            source locations in grid coordinates.  Defaults to ``None``.
        store_dir (str | Path, optional): Folder for the PNG/PDF output.
            Created if necessary.  Pass ``None`` to skip saving.

    Returns:
        list[matplotlib.figure.Figure]: One figure per window.
    """

    # Inputs msut be NUMPY array already
    fig_list = []
    labels = (("X (km)", "Y (km)"),
              ("X (km)", "Z (km)"),
              ("Y (km)", "Z (km)"))
    grid_list = [(grid[0], grid[1]),
                 (grid[0], grid[2]),
                 (grid[1], grid[2])]
    image_list = [image_xy, image_xz, image_yz]
    reference_locations_colors = ["darkred", "white"]

    for ii in tqdm(range(window_x.shape[0])):
        # Create the figure
        # fig = plt.figure(figsize=(8, 8))
        fig = plt.figure(figsize=(7, 7))

        # Set up the grid
        gs = GridSpec(6, 5, figure=fig)

        image_list = [image_xy[ii], image_xz[ii], image_yz[ii]]
        _x, _y, _r, _p = window_x[ii], window_y[ii], window_r[ii], window_pred[ii]
        _verdict = verdict[ii]

        _ref_loc = reference_locations[ii] if len(reference_locations) > 0 else []

        condition = np.any(_y >= 0.95, axis=-1)
        _tot_picks = np.sum(np.any(condition, axis=1))

        # Plot the timeseries on the left (rectangular panels) in columns 1-3
        for xx in range(6):
            ax = fig.add_subplot(gs[xx, 0:3])
            ax.plot(_r[xx][1] + _r[xx][2], color="darkgray", alpha=0.6, label="real_NE")
            ax.plot(_r[xx][0], color="black", alpha=0.6, label="real_Z")
            lf_ax2 = ax.twinx()
            for _cfs in range(_x[xx].shape[0]):
                lf_ax2.plot(_x[xx][_cfs], alpha=0.7, color=f"C{_cfs}", label=f"CF_{_cfs}")
            lf_ax2.plot(_y[xx][0], alpha=0.7, label="Y", color="darkred", ls="--")
            lf_ax2.plot(_p[xx][0], alpha=0.7, label="prediction", color="purple")
            lf_ax2.set_ylim([-0.2, 1.2])
            # # ax.set_ylabel(sorted_indices_ylabel[idx])  # station name
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            lf_ax2.spines['top'].set_visible(False)
            lf_ax2.spines['right'].set_visible(False)
            if xx != 5:
                ax.spines['bottom'].set_visible(False)
                lf_ax2.spines['right'].set_visible(False)

            if xx == 0:
                ax.legend(loc='upper left')
                # lf_ax2.legend(loc='upper right')
                ax.set_title("Total Stations with signal: %d" % _tot_picks)

        # Plot the images on the right (square panels) in rows 1, 3, and 5, spanning columns 4 and 5
        image_axis = []
        for yy in range(3):
            img_ax = fig.add_subplot(gs[2*yy:2*yy+2, 3:5])  # Each square spans two rows and two columns
            image_axis.append(img_ax)

            # img_ax.imshow(random_images[i], cmap='Blues')  # You can choose another colormap
            _ = img_ax.pcolormesh(grid_list[yy][0] / 1000.0,
                                  grid_list[yy][1] / 1000.0,
                                  image_list[yy].T,
                                  cmap='viridis', vmin=0.0, vmax=1.0,
                                  edgecolors='none', rasterized=True)  # shading='gouraud',)
            img_ax.spines['top'].set_visible(False)
            img_ax.spines['right'].set_visible(False)

            img_ax.set_xlabel(labels[yy][0])
            img_ax.set_ylabel(labels[yy][1])
            xticks = img_ax.get_xticks()
            yticks = img_ax.get_yticks()
            # img_ax.set_xticklabels([f'{x:.1f}' for x in xticks])  # Format xticks in km
            # img_ax.set_yticklabels([f'{y:.1f}' for y in yticks])  # Format yticks in km
            # Set the ticks explicitly before setting tick labels
            img_ax.set_xticks(xticks)
            img_ax.set_yticks(yticks)

            if yy == 0:
                img_ax.set_title(" XY / XZ / YZ - %s (%.2f)" % (
                                 "EVENT" if _verdict > 0.5 else "NOISE",
                                 _verdict))
                if stations:
                    for _stat in stations:
                        image_axis[0].scatter(
                            _stat[0] / 1000.0, _stat[1] / 1000.0,
                            facecolor="white",
                            edgecolor='black',
                            marker='^', alpha=0.5,
                            s=20)

            if yy in (1, 2):
                img_ax.invert_yaxis()

            if yy == 2:
                imt = img_ax.collections[0]
                _ = fig.colorbar(imt, ax=img_ax, orientation='horizontal')

        if len(_ref_loc) > 0 and _tot_picks > 0:
            gridlon, gridlat, griddep = _ref_loc
            image_axis[0].scatter(
                    gridlon / 1000.0, gridlat / 1000.0,
                    facecolor=reference_locations_colors[0],
                    edgecolor='black',
                    marker='o',
                    s=20)
            image_axis[1].scatter(
                    gridlon / 1000.0, griddep / 1000.0,
                    facecolor=reference_locations_colors[0],
                    edgecolor='black',
                    marker='o',
                    s=20)
            image_axis[2].scatter(
                    gridlat / 1000.0, griddep / 1000.0,
                    facecolor=reference_locations_colors[0],
                    edgecolor='black',
                    marker='o',
                    s=20)

        # Adjust layout for better spacing
        plt.tight_layout()

        if store_dir:
            Path(store_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(str(Path(store_dir) / ("Results_%03d.png" % ii)), dpi=310)
            fig.savefig(str(Path(store_dir) / ("Results_%03d.pdf" % ii)), dpi=310)
        #
        fig_list.append(fig)
        plt.close()
    #
    return fig_list


def plot_heimdall_flow_continuous(
            window_x, window_y, window_r, window_pred,
            image_xy, image_xz, image_yz, grid, verdict,
            heim_gnn, heim_grid, main_timeseries,
            start_date="", delta_t=None,
            threshold_pdf=0.3,
            order=None, plot_stations=False,
            reference_locations_1=[],
            reference_locations_2=[],
            center_of_mass=[],
            window_length=501,
            sliding=25,
            suptitle="",
            show_cfs=True,
            store_dir="HeimdallResults"):
    """Generate a scrolling diagnostic plot for continuous inference runs.

    Each figure covers one sliding window and contains:

    * Six selected station waveforms (left)
    * PDF projections (right)
    * Full-length reference waveform at the bottom with the processed
      window highlighted

    Args:
        window_x, window_y, window_r, window_pred: Same meaning as in
            :pyfunc:`plot_heimdall_flow`, but for the full continuous run.
        image_xy, image_xz, image_yz: PDF slices per window.
        grid (tuple[np.ndarray, np.ndarray, np.ndarray]): The locator grid.
        verdict (np.ndarray): Model confidence for every window.
        heim_gnn (dict): Full GNN object as returned by training.
        heim_grid (HeimdallGrid): Grid converter for station coordinates.
        main_timeseries (tuple[np.ndarray, np.ndarray]): Tuple
            ``(Rplot, Yplot)`` containing long un-windowed waveforms for
            the reference station.
        start_date (str | obspy.UTCDateTime, optional): Absolute time stamp
            of the first main-timeseries sample.  Enables human-readable
            x-axis.  Defaults to empty string.
        delta_t (float | None): Sample interval for *main_timeseries* in
            seconds.  Required when *start_date* is supplied.
        threshold_pdf (float, optional): Minimum PDF value that counts as a
            detection in the consistency check.  Defaults to ``0.3``.
        order (dict | None): Explicit plot order ``{station_name: rank}``.
        plot_stations (bool, optional): Overlay station markers on the XY
            plane.  Defaults to ``False``.
        reference_locations_1 (Sequence | None): First set of reference
            locations (e.g. catalogue).  Defaults to ``None``.
        reference_locations_2 (Sequence | None): Second set of reference
            locations (e.g. previous model).  Defaults to ``None``.
        center_of_mass (Sequence | None): Pre-computed PDF centre of mass
            per window.  Defaults to ``None``.
        window_length (int, optional): Window length in samples.
        sliding (int, optional): Sliding step in samples.
        suptitle (str, optional): Figure super-title.
        show_cfs (bool, optional): Plot CF channels next to the real
            waveforms.  Defaults to ``True``.
        store_dir (str | Path, optional): Folder for PNG/PDF output.  Use
            ``None`` to disable saving.

    Returns:
        list[matplotlib.figure.Figure]: Figures for all processed windows.
    """

    def __normalize__(inarr):
        asd = inarr / np.max(inarr)
        return asd

    # Inputs must be NUMPY array already
    fig_list = []
    labels = (("X (km)", "Y (km)"),
              ("X (km)", "Z (km)"),
              ("Y (km)", "Z (km)"))
    grid_list = [(grid[0], grid[1]),
                 (grid[0], grid[2]),
                 (grid[1], grid[2])]
    image_list = [image_xy, image_xz, image_yz]
    reference_locations_colors = ["white", "darkred"]

    # ================================================================
    (Rplot, Yplot) = main_timeseries

    # Check STARTDATE
    if start_date and delta_t:
        if isinstance(start_date, str):
            plt_start_date = UTC(start_date)
        elif isinstance(start_date, UTC):
            plt_start_date = start_date
        else:
            raise ValueError("start_date must be either string or UTCDateTime")
        assert isinstance(delta_t, (int, float))
        assert delta_t is not None
        # --> Create array time
        time_delta = np.arange(Rplot.shape[-1]) * delta_t
        time_stamps = [plt_start_date + t for t in time_delta]
        # --> Convert to matplotlib datetime format
        time_stamps = [t.datetime for t in time_stamps]
    else:
        plt_start_date, delta_t, time_stamps = None, None, np.arange(
                                                            Rplot.shape[-1])

    # --------------------------- Stations
    stations_order = heim_gnn['stations_order'].item()
    stations_latlon = [(vv[1], vv[0]) for kk, vv in
                       heim_gnn['stations_coordinate'].item().items()]
    stations_xyz = heim_grid.grid.convert_geo_list(stations_latlon)

    if order:
        # order = order or {name: i for i, name in enumerate(stations_order)}
        closest_waveform_indices = [stations_order[name] for name in
                                    sorted(stations_order, key=order.get)]
        closest_stations = [name for name in
                            sorted(stations_order, key=order.get)]

    else:
        # ==============  Find the 6 closest station, and their names
        argmax_result = np.argmax(Yplot >= 0.97, axis=2)
        flat_result = [val[0] for val in argmax_result]
        non_zero_values = [(val, idx) for idx, val in enumerate(flat_result) if val != 0]
        non_zero_values_sorted = sorted(non_zero_values, key=lambda x: x[0])
        closest_waveform_indices = [idx for _, idx in non_zero_values_sorted[:7]]

        index_to_station = {v: k for k, v in stations_order.items()}
        closest_stations = [index_to_station[idx] for idx in closest_waveform_indices]

    # # -----------------------------------  Main Trace
    ts_r_z = __normalize__(Rplot[closest_waveform_indices[0], 0, :])
    ts_r_ne = __normalize__((Rplot[closest_waveform_indices[0], 1, :] +
                             Rplot[closest_waveform_indices[0], 2, :]))
    ts_r_y = __normalize__(Yplot[closest_waveform_indices[0], 0, :])
    start_val, end_val = np.min(ts_r_ne), np.max(ts_r_ne)
    plot_verdict_list = []

    # ========================================  For all windows in event
    for ii in range(window_x.shape[0]):
        # Create the figure with an increased figure size to accommodate the new timeseries
        fig = plt.figure(figsize=(7, 8))

        # Set up the grid (7 rows instead of 6)
        gs = GridSpec(7, 5, figure=fig)

        _start_idx = ii*sliding
        # _sliding_day = sliding / (24 * 3600)
        # _start_idx = (plt_start_date + ii*_sliding_day).datetime

        image_list = [image_xy[ii], image_xz[ii], image_yz[ii]]
        _x, _y, _r, _p = window_x[ii], window_y[ii], window_r[ii], window_pred[ii]

        _verdict = verdict[ii]
        _all_above_threshold = all(x > threshold_pdf for x in
                                   (np.max(image_list[0]),
                                    np.max(image_list[1]),
                                    np.max(image_list[2])))
        # if _verdict > 0.5 and _all_above_threshold:
        if _verdict > 0.5:
            plot_verdict_list.append((
                    _start_idx+window_length,      # X
                    0.75  # np.abs(end_val-start_val)*0.9  # Y
            ))

        _ref_loc1 = reference_locations_1[ii] if len(reference_locations_1) > 0 else []
        _ref_loc2 = reference_locations_2[ii] if len(reference_locations_2) > 0 else []

        # ========================================  MAIN  TIMESERIES
        # Add a new timeseries plot at the bottom, spanning all 5 columns
        ax_bottom = fig.add_subplot(gs[6, 0:5])
        ax_bottom.plot(time_stamps, ts_r_ne, label='NE', color="darkgray", alpha=0.8)
        ax_bottom.plot(time_stamps, ts_r_z, label='Z', color="black", alpha=0.8)
        ax_bottom.plot(time_stamps, ts_r_y, label='Y', ls="--", color="purple", alpha=0.8)
        # ax4.set_title('Time Series')
        ax_bottom.set_xlabel('Time (idx)')
        ax_bottom.set_ylabel(closest_stations[0])
        ax_bottom.set_ylim((-1.2, 1.2))
        ax_bottom.spines['top'].set_visible(False)
        ax_bottom.spines['right'].set_visible(False)
        #
        rect = Rectangle((_start_idx, start_val),  # low.left corner
                         width=window_length, height=np.abs(end_val-start_val),
                         linewidth=1, edgecolor='teal', facecolor='none')
        ax_bottom.add_patch(rect)
        if plot_verdict_list:
            ax_bottom.scatter(
                        [_pv[0] for _pv in plot_verdict_list],
                        [_pv[1] for _pv in plot_verdict_list],
                        facecolor="teal",
                        marker="x",
                        s=18)
        if plt_start_date:
            ax_bottom.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6))
            ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        # ========================================  TIMESERIES
        # Plot the timeseries on the left (rectangular panels) in columns 1-3
        for _plotidx, (xx, sxx) in enumerate(zip(closest_waveform_indices[1:7],
                                                 closest_stations[1:7])):
            ax = fig.add_subplot(gs[_plotidx, 0:3])
            ax.plot(__normalize__(_r[xx][1] + _r[xx][2]),
                    color="darkgray", alpha=0.6, label="real_NE")
            ax.plot(__normalize__(_r[xx][0]),
                    color="black", alpha=0.6, label="real_Z")
            ax.set_ylabel(sxx)  # station name
            ax.set_ylim((-1.2, 1.2))
            lf_ax2 = ax.twinx()

            # SED_Sept2024 --> comment
            if show_cfs:
                for _cfs in range(_x[xx].shape[0]):
                    lf_ax2.plot(_x[xx][_cfs], alpha=0.7, color=f"C{_cfs}", label=f"CF_{_cfs}")
            #
            lf_ax2.plot(_y[xx][0], alpha=0.7, label="Y", color="darkred", ls="--")

            # PREDICTIONS
            lf_ax2.plot(_p[xx][0], alpha=0.7, label="prediction", color="purple")
            if _p[xx].shape[0] > 1:
                # picker_type
                lf_ax2.plot(_p[xx][1], alpha=0.7, label="prediction_P", color="darkblue")
                lf_ax2.plot(_p[xx][2], alpha=0.7, label="prediction_S", color="darkred")
            lf_ax2.set_ylim([-0.2, 1.2])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            lf_ax2.spines['top'].set_visible(False)
            lf_ax2.spines['right'].set_visible(False)
            if xx != 5:
                ax.spines['bottom'].set_visible(False)
                lf_ax2.spines['right'].set_visible(False)

            # if xx == 0:
            #     ax.legend(loc='upper left')
            #     ax.set_title("Total Stations with signal: %d" % _tot_picks)

        # ========================================  IMAGES
        # Plot the images on the right (square panels) in rows 1, 3, and 5, spanning columns 4 and 5
        image_axis = []
        image_max_coord = []
        for yy in range(3):
            img_ax = fig.add_subplot(gs[2*yy:2*yy+2, 3:5])  # Each square spans two rows and two columns
            image_axis.append(img_ax)

            image_max_coord.append(extract_max_info(grid_list[yy][0],
                                                    grid_list[yy][1],
                                                    image_list[yy].T))
            _ = img_ax.pcolormesh(grid_list[yy][0] / 1000.0,
                                  grid_list[yy][1] / 1000.0,
                                  image_list[yy].T,
                                  cmap='viridis', vmin=0.0, vmax=1.0,
                                  edgecolors='none', rasterized=True)
            img_ax.spines['top'].set_visible(False)
            img_ax.spines['right'].set_visible(False)

            img_ax.set_xlabel(labels[yy][0])
            img_ax.set_ylabel(labels[yy][1])
            xticks = img_ax.get_xticks()
            yticks = img_ax.get_yticks()
            img_ax.set_xticklabels([f'{x:.1f}' for x in xticks])  # Format xticks in km
            img_ax.set_yticklabels([f'{y:.1f}' for y in yticks])  # Format yticks in km

            if yy == 0:
                # img_ax.set_title(" Grid Planes (%.2f)" % _verdict)
                if plot_stations:
                    for _stat in stations_xyz:
                        image_axis[0].scatter(
                            _stat[0] / 1000.0, _stat[1] / 1000.0,
                            facecolor="white",
                            edgecolor='black',
                            marker='^', alpha=0.5,
                            s=20)

            if yy in (1, 2):
                img_ax.invert_yaxis()

        image_axis[0].set_title("%.2f - %.2f - %.2f" %
                                (np.max(image_list[0]),
                                 np.max(image_list[1]),
                                 np.max(image_list[2])))

        # =========================================  PDF MAX
        image_axis[0].scatter(
                image_max_coord[0][0] / 1000.0, image_max_coord[0][1] / 1000.0,
                facecolor='darkred',
                marker='x',
                s=18)
        image_axis[1].scatter(
                image_max_coord[1][0] / 1000.0, image_max_coord[1][1] / 1000.0,
                facecolor='darkred',
                marker='x',
                s=18)
        image_axis[2].scatter(
                image_max_coord[2][0] / 1000.0, image_max_coord[2][1] / 1000.0,
                facecolor='darkred',
                marker='x',
                s=18)

        # =========================================  Reference LOCATION
        if len(_ref_loc1) > 0:
            gridlon, gridlat, griddep = _ref_loc1
            image_axis[0].scatter(
                    gridlon / 1000.0, gridlat / 1000.0,
                    facecolor=reference_locations_colors[0],
                    edgecolor='black',
                    marker='o',
                    s=20)
            image_axis[1].scatter(
                    gridlon / 1000.0, griddep / 1000.0,
                    facecolor=reference_locations_colors[0],
                    edgecolor='black',
                    marker='o',
                    s=20)
            image_axis[2].scatter(
                    gridlat / 1000.0, griddep / 1000.0,
                    facecolor=reference_locations_colors[0],
                    edgecolor='black',
                    marker='o',
                    s=20)

        if len(_ref_loc2) > 0:
            gridlon, gridlat, griddep = _ref_loc2
            image_axis[0].scatter(
                    gridlon / 1000.0, gridlat / 1000.0,
                    facecolor=reference_locations_colors[1],
                    edgecolor='black',
                    marker='o',
                    s=20)
            image_axis[1].scatter(
                    gridlon / 1000.0, griddep / 1000.0,
                    facecolor=reference_locations_colors[1],
                    edgecolor='black',
                    marker='o',
                    s=20)
            image_axis[2].scatter(
                    gridlat / 1000.0, griddep / 1000.0,
                    facecolor=reference_locations_colors[1],
                    edgecolor='black',
                    marker='o',
                    s=20)

        # --------> Adjust layout for better spacing
        if suptitle:
            fig.suptitle(suptitle, fontweight='bold')
        plt.tight_layout()

        if store_dir:
            Path(store_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(str(Path(store_dir) / ("Results_%03d.png" % ii)), dpi=310)
            fig.savefig(str(Path(store_dir) / ("Results_%03d.pdf" % ii)), dpi=310)
        #
        fig_list.append(fig)
        plt.close()
    #
    return fig_list


def extract_max_info(xgr, ygr, matrix):
    """Locate the maximum of a 2-D PDF slice.

    Args:
        xgr (np.ndarray): X grid corresponding to *matrix* columns.
        ygr (np.ndarray): Y grid corresponding to *matrix* rows.
        matrix (np.ndarray): 2-D array to be analysed.

    Returns:
        tuple: ``(x_max, y_max, value_max)`` where *x_max* and *y_max* are
        the grid coordinates at which *matrix* is maximal.
    """
    max_value = np.max(matrix)
    y, x = np.unravel_index(np.argmax(matrix), matrix.shape)
    return (xgr[x], ygr[y], max_value)
