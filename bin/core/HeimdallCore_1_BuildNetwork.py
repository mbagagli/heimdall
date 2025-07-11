#!/Users/matteo/miniconda3/envs/msca_pt/bin/python

"""
First script to build and visualize the GRAPH NEURAL NETWORK

At the moment only STATIC graph are allowed, not the dynamic...

It will store a *npz file named: `heim_gnn.npz` containing the following:
 - NODES:  torch Tensor
 - EDGES: Torch tensor
 - WEIGHTS: torch Tensor (edges attributes)
 - STATION_ORDER: dictionary ({'net.stat': idx, ...})

An also:
 - `node_station_map_unordered.txt` containing ASCII text of all
                                    stations` coordinates
"""

import sys
import numpy as np
from obspy import UTCDateTime as UTC
#
import torch
#
import heimdall
from heimdall import io as gio
from heimdall import graph as gnn
from heimdall import plot as gpl
from heimdall import custom_logger as CL
#
from pathlib import Path


logger = CL.init_logger(Path(sys.argv[0]).name, lvl="INFO",
                        log_file="BuildNetwork.log")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


def build_heimdall_gnn(confs):
    proj_start_date = UTC(confs.PROJ_START_DATE)
    proj_end_date = UTC(confs.PROJ_END_DATE)

    _start_title = "%04d-%02d-%02d" % (
            proj_start_date.year,
            proj_start_date.month,
            proj_start_date.day,
        )
    _end_title = "%04d-%02d-%02d" % (
            proj_end_date.year,
            proj_end_date.month,
            proj_end_date.day,
        )
    logger.info("Working with:  %s - %s" % (_start_title, _end_title))

    # 1. ---------------------------------------  Define Graphs
    (NODES, stations_dict, stations_coord_dict, inv) = gnn.define_nodes_from_inventory(
                                    confs.INVENTORY_PATH,
                                    time_query={
                                            # 'time': PROJ_START_DATE,
                                            'starttime': proj_start_date,
                                            'endtime': proj_end_date,
                                        },
                                    networks_query=confs.NETWORKS,
                                )

    (EDGES, clusters, base_group_type, base_group_value) = gnn.define_edges(
                                NODES,
                                do_clustering=False,
                                clustering_args={
                                    "mode": "DBSCAN",
                                    "radius": 0.03,
                                    "min_samples": 1
                                },
                                base_connect_type=confs.BASE_CONNECT_TYPE,
                                base_connect_value=confs.BASE_CONNECT_VALUE,
                                add_self_loops=confs.SELF_LOOPS)

    WEIGHTS = gnn.weight_edges_by_distance(NODES, EDGES,
                                           scale_distances=confs.SCALE_DISTANCES)

    # 2. ---------------------------------------  Plot
    if confs.PLOT_GRAPH_ARCH:
        logger.info("Plotting")
        _ = gpl.plot_graph(
                NODES, EDGES,
                clusters=clusters,
                fig_title="Graph Network\n%s / %s" % (_start_title, _end_title),
                store="./Heimdall_GNN__%s_%s__.pdf" % (
                                _start_title, _end_title),
                limits=confs.PLOT_BOUNDARIES,
                show=True)

    # 3. ---------------------------------------  Store
    logger.info("Storing")
    np.savez('heim_gnn.npz',
             nodes=NODES,
             edges=EDGES.T,  # must be with SHAPE (2, N)
             base_group_type=base_group_type,
             base_group_value=base_group_value,
             weights=WEIGHTS,
             networks=confs.NETWORKS,
             stations_order=stations_dict,
             stations_coordinate=stations_coord_dict,
             tag=confs.GNN_TAG,
             version=heimdall.__version__)

    with open("node_station_map_unordered.txt", "w") as OUT:
        OUT.write("#LON LAT ELE_mt NAME NET\n")
        for _net in inv:
            for _sta in _net:
                OUT.write("%f  %f  %8.2f  %6s  %2s\n" % (
                        _sta.longitude, _sta.latitude, _sta.elevation,
                        _sta.code, _net.code
                    ))

    logger.warning("NODES:  %3d - EDGES:  %3d" % (NODES.shape[0], EDGES.shape[0]))
    logger.info("DONE")


if __name__ == "__main__":
    try:
        assert Path(sys.argv[1]).exists()
    except:
        logger.error("Configuration file-path non existent!")
        sys.exit()
    CONFIG = gio.read_configuration_file(sys.argv[1], check_version=True)
    build_heimdall_gnn(CONFIG.BUILD_GNN)
