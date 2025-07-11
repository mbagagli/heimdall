"""
This module provides functionality for handling seismic data and graph-based analyses using geospatial and network properties.
It includes tools for station activity querying, node and edge creation for graph analysis, and utilities for plotting seismic networks.

Functions:
    __get_active_stations__: Retrieves active stations from an ObsPy inventory object within specified time constraints.
    __get_all_stations__: Retrieves all stations from an ObsPy inventory object.
    define_nodes_from_inventory: Defines graph nodes from a seismic inventory based on various filtering and grouping criteria.
    __extract_unique_connections__: Identifies unique connections in a tensor of point pairs.
    __connect_single_element_clusters__: Connects single-element clusters to their nearest neighbors to ensure graph connectivity.
    __connect_clusters__: Establishes connections between sparsely connected nodes in a graph.
    __connect_contour_edges__: Connects nodes along the perimeter of their convex hull.
    define_edges: Configures edges between nodes in a graph based on spatial relationships and clustering.
    weight_edges_by_distance: Computes weights for graph edges based on geographical distance.
    plot_graph: Visualizes a graph of seismic stations and their connections.
"""

import sys
import obspy
import numpy as np
from pathlib import Path
#
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull
#
import torch
from torch_geometric.nn import radius_graph
from torch_geometric.nn import knn_graph
#
from heimdall import utils as GUT
from heimdall import custom_logger as CL

logger = CL.init_logger(Path(sys.argv[0]).name, lvl="INFO")


# ===================================================================
# ===================================================================  NODES
# ===================================================================

def __get_active_stations__(inv, time=None, starttime=None, endtime=None):
    """
    Retrieves active seismic stations from an inventory within optional specified time frames.

    Args:
        inv (obspy.core.inventory.inventory.Inventory): The inventory from which to retrieve station data.
        time (obspy.core.utcdatetime.UTCDateTime, optional): Specific time at which stations should be active.
        starttime (obspy.core.utcdatetime.UTCDateTime, optional): Start time for the activity window.
        endtime (obspy.core.utcdatetime.UTCDateTime, optional): End time for the activity window.

    Returns:
        tuple: Contains tensors of station coordinates and dictionaries with station metadata.
    """
    assert isinstance(inv, obspy.core.inventory.inventory.Inventory)
    if time:
        assert isinstance(time, obspy.core.utcdatetime.UTCDateTime)
    if starttime:
        assert isinstance(starttime, obspy.core.utcdatetime.UTCDateTime)
    if endtime:
        assert isinstance(endtime, obspy.core.utcdatetime.UTCDateTime)
    #
    net_codes = []
    out_nodes = []  # lon / lat / elev
    to_remove = []
    sta_dict = {}
    sta_coord = {}
    sta_idx = 0
    out_inv = inv.copy()
    for _x, _net in enumerate(inv):
        net_codes.append(_net.code)
        for _y, _sta in enumerate(_net):
            if _sta.is_active(time=time, starttime=starttime, endtime=endtime):
                out_nodes.append(
                    (_sta.longitude, _sta.latitude, _sta.elevation)
                )
                #
                sta_dict[_net.code+"."+_sta.code] = sta_idx
                sta_coord[_net.code+"."+_sta.code] = (_sta.longitude,
                                                      _sta.latitude,
                                                      _sta.elevation)
                sta_idx += 1
            else:
                to_remove.append((_net.code, _sta.code))
    #
    for (_net, _sta) in to_remove:
        out_inv = out_inv.remove(network=_net, station=_sta)
    #
    nodes_tensor = torch.tensor(out_nodes)
    return (nodes_tensor, sta_dict, sta_coord, out_inv)


def __get_all_stations__(inv):
    """
    Retrieves all seismic stations from an inventory.

    Args:
        inv (obspy.core.inventory.inventory.Inventory): The inventory from which to retrieve station data.

    Returns:
        tuple: Contains tensors of station coordinates and dictionaries with station metadata.
    """
    assert isinstance(inv, obspy.core.inventory.inventory.Inventory)
    #
    net_codes = []
    out_nodes = []  # lon / lat / elev
    sta_dict = {}
    sta_coord = {}
    sta_idx = 0
    for _x, _net in enumerate(inv):
        net_codes.append(_net.code)
        for _y, _sta in enumerate(_net):
            out_nodes.append(
                (_sta.longitude, _sta.latitude, _sta.elevation)
            )
            #
            sta_dict[_net.code+"."+_sta.code] = sta_idx
            sta_coord[_net.code+"."+_sta.code] = (_sta.longitude, _sta.latitude, _sta.elevation)
            sta_idx += 1
    nodes_tensor = torch.tensor(out_nodes)
    return (nodes_tensor, sta_dict, sta_coord, inv)


def define_nodes_from_inventory(obs_inv, relative=True, time_query={},
                                networks_query=[]):
    """
    Defines nodes from seismic inventory data for graph-based analyses, filtering and grouping based on provided criteria.

    Args:
        obs_inv (str, Path, obspy.core.inventory.inventory.Inventory): The seismic inventory to process.
        relative (bool): Whether to calculate node positions relative to a central point.
        time_query (dict): Criteria to filter stations based on their activity times.
        networks_query (list): Specific network codes to filter the inventory.

    Returns:
        tuple: Contains tensors of node coordinates and dictionaries with node metadata.
    """
    logger.info("Reading INVENTORY:  %s" % str(obs_inv))
    if isinstance(obs_inv, (str, Path)):
        obs_inv = obspy.read_inventory(str(obs_inv))

    assert isinstance(obs_inv, obspy.core.inventory.inventory.Inventory)
    work_inv = obs_inv.copy()

    # Network query
    if networks_query:
        assert isinstance(networks_query, (list, tuple))
        logger.debug("Filtering by network: %s" % networks_query)
        _net_in = set([net.code for net in work_inv])
        _net_remove = _net_in - set(networks_query)
        for _net in _net_remove:
            work_inv = work_inv.remove(network=_net)

    # Time Query
    if time_query:
        # Select based on dates
        logger.debug("Importing TIME-BASED STATIONS")
        (nodes, stat_map_dict, stat_coord_dict, opinventory) = __get_active_stations__(
                                                        work_inv, **time_query)
    else:
        # Get All
        logger.debug("Importing ALL STATIONS")
        (nodes, stat_map_dict, stat_coord_dict, opinventory) = __get_all_stations__(work_inv)

    # Relative
    if relative:
        logger.debug("Relative Method not yet implemented ... returning standard coordinates")
    return (nodes, stat_map_dict, stat_coord_dict, opinventory)


# ===================================================================
# ===================================================================  EDGES
# ===================================================================


def __extract_unique_connections__(pt_tensor):
    """
    Extracts unique connections from a tensor of point pairs, ensuring each connection is listed only once.

    Args:
        pt_tensor (torch.Tensor): A tensor of point pairs.

    Returns:
        torch.Tensor: A tensor containing unique point pairs.
    """
    # Sort the tensor along the second dimension
    sorted_tensor, _ = torch.sort(pt_tensor, dim=1)

    # Find unique rows
    unique_rows, _ = torch.unique(sorted_tensor, dim=0, return_inverse=True)

    # Optionally, sort the result to maintain the original order
    _, original_order = torch.sort(torch.arange(unique_rows.shape[0]))
    unique_rows = unique_rows[original_order]

    # unique_rows now contains the unique rows
    return unique_rows


def __connect_single_element_clusters__(nodes, edges,
                                        max_adj_clusters=2,
                                        max_edge_per_cluster=3):
    """
    Connects single-element clusters within a node set to ensure graph connectivity.

    Args:
        nodes (torch.Tensor): Tensor of node features.
        edges (torch.Tensor): Tensor of existing edges.
        max_adj_clusters (int): Maximum adjacent clusters to connect.
        max_edge_per_cluster (int): Maximum edges per cluster to create.

    Returns:
        torch.Tensor: Updated tensor of edges including new connections.
    """
    from scipy.spatial import distance_matrix

    # Identify unique clusters
    unique_clusters = nodes[:, 3].unique().tolist()

    # Identify single-element clusters
    single_element_clusters = [c for c in unique_clusters if (nodes[:, 3] == c).sum() == 1]

    # For each single-element cluster, find closest nodes in other clusters
    for cluster in single_element_clusters:
        single_node_index = (nodes[:, 3] == cluster).nonzero(as_tuple=True)[0][0]
        single_node = nodes[single_node_index, :2]

        # Create a distance matrix to find closest nodes in other clusters based on 2D
        distances = distance_matrix(single_node.unsqueeze(0).numpy(), nodes[:, :2].numpy())

        # We'll sort clusters based on the distance of their closest node to our single node
        sorted_clusters = sorted(unique_clusters, key=lambda x: distances[0, nodes[:, 3] == x].min())

        # Connect with min to max clusters
        connected_clusters = 0
        for adj_cluster in sorted_clusters:
            if adj_cluster == cluster:
                continue  # Skip self

            # Nodes in the adjacent cluster
            adj_nodes_indices = (nodes[:, 3] == adj_cluster).nonzero(as_tuple=True)[0]
            adj_nodes = nodes[adj_nodes_indices, :2]

            # Sort them based on distance to our single-node
            adj_nodes_indices = adj_nodes_indices[torch.argsort(torch.norm(adj_nodes - single_node.unsqueeze(0), dim=1))]

            # Connect with the closest nodes (up to max_edge_per_cluster) in this cluster
            for i in range(min(len(adj_nodes_indices), max_edge_per_cluster)):
                # Add edge from single-node to adj_node
                edge = torch.tensor([
                    single_node_index,
                    adj_nodes_indices[i]
                ]).unsqueeze(1)  # Shape it as (2, 1)
                edges = torch.cat((edges, edge), dim=1)

            connected_clusters += 1
            if connected_clusters >= max_adj_clusters:
                break

    return edges


def __connect_clusters__(nodes, edges, method="nearest", k=1):
    """
    Connects sparsely connected nodes in a graph based on the nearest or k-nearest neighbors.

    Args:
        nodes (torch.Tensor): Tensor of node features.
        edges (torch.Tensor): Tensor of existing edges.
        method (str): Method to connect nodes, either 'nearest' or 'k_nearest'.
        k (int): Number of neighbors to connect for 'k_nearest' method.

    Returns:
        torch.Tensor: Updated tensor of edges including new connections.
    """

    node_features = nodes[:, :2]

    # Identify spurious dots (nodes without any edges)
    all_nodes = set(range(node_features.size(0)))
    connected_nodes = set(edges.view(-1).tolist())  # Flatten the edges tensor
    spurious_dots = all_nodes - connected_nodes

    new_edges = []
    for dot in spurious_dots:
        distances = torch.norm(node_features - node_features[dot], dim=1)
        # Mask out the spurious dot itself
        distances[dot] = float('inf')

        if method == "nearest":
            neighbor = torch.argmin(distances).item()
            new_edges.append([dot, neighbor])
        elif method == "k_nearest":
            _, neighbors = distances.topk(k, largest=False)
            for neighbor in neighbors:
                new_edges.append([dot, neighbor.item()])

    # Convert to tensor and concatenate with the original edges
    new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()  # Transpose to match shape (2, num_edges)
    combined_edges = torch.cat([edges, new_edges_tensor], dim=1)

    return combined_edges


def __connect_contour_edges__(nodes, edges):
    """
    Connects nodes along the contour of their convex hull to form perimeter edges.

    Args:
        nodes (torch.Tensor): Tensor of node coordinates.
        edges (torch.Tensor): Tensor of existing edges.

    Returns:
        torch.Tensor: Updated tensor of edges forming the convex hull perimeter.
    """

    # Get the convex hull of the points
    hull = ConvexHull(nodes[:, :2])

    # hull.vertices will give you the perimeter indices in order
    perimeter_indices = hull.vertices.tolist()

    # Create connections using these indices
    connections = []
    for i in range(len(perimeter_indices)):
        start = perimeter_indices[i]
        # Connect to the next in line or to the first one if it's the last index
        end = perimeter_indices[(i+1) % len(perimeter_indices)]
        connections.append((start, end))

    # Convert connections to a PyTorch tensor
    edges = torch.tensor(connections).t()
    return edges


def define_edges(nodes,
                 do_clustering=True,
                 clustering_args={
                        "mode": "DBSCAN",
                        "radius": 0.05,
                        "min_samples": 1
                    },
                 base_connect_type="KNN",
                 base_connect_value=5,
                 add_self_loops=False):
    """
    Defines edges for a graph representing seismic stations based on spatial relationships and optional clustering.

    Args:
        nodes (torch.Tensor): Tensor of node coordinates.
        do_clustering (bool): Whether to apply clustering to define edges.
        clustering_args (dict): Arguments to control the clustering method.

    Returns:
        tuple: Contains tensors of edge pairs and cluster labels.
    """
    logger.info("Creating NODE-EDGES")
    X = nodes[:, 0:2]

    logger.info(f"Using {base_connect_type} linking with value {base_connect_value}")
    if add_self_loops:
        logger.warning("Adding self-loops !!!")

    if base_connect_type.lower() in ("knn", "nn"):
        if add_self_loops:
            edges = knn_graph(X, base_connect_value+1, loop=add_self_loops)
        else:
            edges = knn_graph(X, base_connect_value)

    elif base_connect_type.lower() in ("radius", "rad", "circle"):
        if add_self_loops:
            edges = radius_graph(X, base_connect_value+1, loop=add_self_loops)
        else:
            edges = radius_graph(X, base_connect_value)

    edges = edges.t()  # for pytorch geometric

    # --- DBSCAN
    if do_clustering:
        logger.warning(f"Creating NODE clustering:  {clustering_args['mode']}")
        if clustering_args["mode"].lower() == "dbscan":
            clusters = DBSCAN(eps=clustering_args["radius"],
                              min_samples=clustering_args["min_samples"]).fit_predict(X)
        elif clustering_args["mode"].lower() == "gaussian":
            # --- GaussianMixture
            clusters = GaussianMixture(n_components=5).fit_predict(X)
        else:
            logger.error("Unkown clustering method: %s" %
                         clustering_args["mode"])
            sys.exit()
        #
        pytorch_vector = torch.from_numpy(clusters)
        pytorch_vector = pytorch_vector.view(-1, 1)
        nodes = torch.cat((nodes, pytorch_vector), dim=1)

        # --- Edge CUSTOM
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                if clusters[i] == clusters[j] and clusters[i] != -1:
                    new_edge = torch.tensor([i, j])
                    edges = torch.cat((edges, new_edge.unsqueeze(0)), dim=0)
    else:
        clusters = None

    # edges = __connect_single_element_clusters__(nodes, edges)  # MB
    # edges = __connect_clusters__(nodes, edges, method="nearest")
    # edges = __connect_clusters__(nodes, edges, method="k_nearest", k=3)
    # edges = __connect_contour_edges__(nodes, edges)

    # # ---> Finalize edge vector!
    # edges = __extract_unique_connections__(edges)
    edges = GUT.__numpy_array_to_pytorch_tensor__(
                    edges, dtype="float32")

    return (edges, clusters, base_connect_type.lower(), base_connect_value)


def weight_edges_by_distance(nodes, edges, scale_distances=False):
    """
    Calculates weights for edges based on geographical distance between nodes.
    Args:
        nodes (torch.Tensor): Tensor of node coordinates including longitude, latitude, and elevation.
        edges (torch.Tensor): Tensor of edges to weight.

    Returns:
        torch.Tensor: Tensor of weights corresponding to the geographical distances between nodes.
    """

    def __normalize_weigths_zeroone__(inarr, X=0, Y=1):
        try:
            assert isinstance(inarr, np.ndarray)
        except AssertionError:
            inarr = np.array(inarr)
        # Normalize the vector between 0 and 1
        min_val = np.min(inarr)
        max_val = np.max(inarr)
        normalized_vector = (inarr - min_val) / (max_val - min_val)

        # Normalize the vector to the range [X, Y]
        normalized_vector = X + (Y - X) * (inarr - min_val) / (max_val - min_val)
        return normalized_vector

    def __normalize_weigths_max__(weight_tensor):
        return weight_tensor / np.max(np.abs(weight_tensor))

    def __normalize_weigths_std__(weight_tensor):
        return weight_tensor / np.std(weight_tensor)

    def __calc_interstation_distance__(coord_one, coord_two):
        (sta1_lon, sta1_lat, sta1_elev) = coord_one
        (sta2_lon, sta2_lat, sta2_elev) = coord_two
        # ==== If elevation in meters --> convert to KM!  (default)
        sta1_elev = sta1_elev / 1000.0
        sta2_elev = sta2_elev / 1000.0
        # ================================
        epi_deg = obspy.geodetics.locations2degrees(
                sta1_lat, sta1_lon, sta2_lat, sta2_lon)
        epi_km = obspy.geodetics.degrees2kilometers(epi_deg)
        dist_km = np.sqrt((epi_km**2) + np.abs(sta2_elev-sta1_elev)**2)
        return dist_km
    #
    assert edges.shape[1] == 2
    weight_tensor = []
    for (_node1, _node2) in edges:
        _weight = __calc_interstation_distance__(
            np.array(nodes[_node1]), np.array(nodes[_node2])
            )
        weight_tensor.append(_weight)

    # # -------------------------------------  v0.2.9 and higher
    # # If we want to provide minor importance to distant station, we need
    # # to give higher values to the smaller edge lengths (station distance)
    # # Therefore, we use the inverse AND THEN calculate the normalization.
    weight_tensor = [1.0 / w if w > 0 else 1.0 for w in weight_tensor]  # In case of self loops
    # # -------------------------------------

    if scale_distances and scale_distances.lower() in ("std", "dev"):
        weight_tensor = __normalize_weigths_std__(weight_tensor)
    elif scale_distances and scale_distances.lower() in ("maximum", "max"):
        weight_tensor = __normalize_weigths_max__(weight_tensor)
    elif scale_distances and scale_distances.lower() in ("zeroone", "01"):
        weight_tensor = __normalize_weigths_zeroone__(weight_tensor)
    #
    weight_tensor = torch.tensor(weight_tensor)
    return weight_tensor
