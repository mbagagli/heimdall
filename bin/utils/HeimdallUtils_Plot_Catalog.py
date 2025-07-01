#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read_inventory
from matplotlib.lines import Line2D


plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'axes.edgecolor': 'gray',
    'axes.linewidth': 0.8,
})


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Plot 2D and 3D views of HEIMDALL seismic catalog.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-catalog", "-c", type=str,
                        help="Path to the CSV catalog file")
    parser.add_argument("-inventory", "-i", type=str,
                        help="Path to the station inventory XML file")
    parser.add_argument("--plotstations", action="store_true",
                        help="Plot stations on the map")
    parser.add_argument("--starttime", type=str, default="1970-01-01 00:00:00",
                        help="Start time for filtering events")
    parser.add_argument("--endtime", type=str, default="2100-01-01 00:00:00",
                        help="End time for filtering events")
    parser.add_argument("--depthmax", type=int, default=15000,
                        help="Maximum depth in meters for plotting")
    parser.add_argument("--region", type=float, nargs=4,
                        metavar=('LON_MIN', 'LON_MAX', 'LAT_MIN', 'LAT_MAX'),
                        help="Region bounding box: lon_min lon_max lat_min lat_max")
    return parser.parse_args()


def main():
    args = parse_cli()
    inv = read_inventory(args.inventory)
    inv = inv.remove(network="VI")
    stations = []
    for network in inv:
        for station in network:
            stations.append({
                "latitude": station.latitude,
                "longitude": station.longitude,
                "elevation": station.elevation * -1
            })
    stations_df = pd.DataFrame(stations)

    print("Reading: ", args.catalog)
    df = pd.read_csv(args.catalog)
    df['ORIGIN_TIME'] = pd.to_datetime(df['ORIGIN_TIME'])
    df.sort_values('ORIGIN_TIME', inplace=True)

    # =============== FILTERING ===============
    # Filter by time
    if args.starttime:
        df = df[df['ORIGIN_TIME'] >= pd.to_datetime(args.starttime)]
    if args.endtime:
        df = df[df['ORIGIN_TIME'] <= pd.to_datetime(args.endtime)]

    # Filter by depth
    df = df[df['DEPTH(km)'] <= args.depthmax * 1e-3]

    # Region boundaries
    if args.region:
        lon_min, lon_max, lat_min, lat_max = args.region
    else:
        lon_min, lon_max = df['LONGITUDE'].min()-0.05, df['LONGITUDE'].max()+0.05
        lat_min, lat_max = df['LATITUDE'].min()-0.05, df['LATITUDE'].max()+0.05

    boundaries = np.array([
        [lon_min, lon_max],
        [lat_min, lat_max],
        [0., args.depthmax]
    ])

    # =============== MAGNITUDE SIZE ==========
    magmin = df["MAGNITUDE"].min()
    magshift = -magmin + 0.1
    magscale = 6
    df["MAGPLOTSIZES"] = magscale * (3 ** (df["MAGNITUDE"] + magshift))
    df = df[df['DEPTH(km)'] <= args.depthmax * 1e-3]

    mag_1_size = magscale * (3 ** (1.0 + magshift))
    mag_2_size = magscale * (3 ** (2.0 + magshift))

    # =============== 3D FIGURE ===============
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    if args.plotstations and not stations_df.empty:
        ax.scatter(
            stations_df["longitude"],
            stations_df["latitude"],
            stations_df["elevation"],
            c="darkorange", alpha=0.7, s=60, marker="^",
            label="Stations", edgecolor='black', linewidths=0.5)

    ax.scatter(
        df['LONGITUDE'], df['LATITUDE'], df['DEPTH(km)'] * 1e3,
        s=df["MAGPLOTSIZES"], c='teal', alpha=0.8, marker='o',
        edgecolor='black', linewidths=0.5, label='Events')

    ax.set_xlim(boundaries[0, :])
    ax.set_ylim(boundaries[1, :])
    ax.set_zlim(boundaries[2, :])
    ax.invert_zaxis()
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_zlabel("Depth (m)")
    ax.tick_params(axis='both', which='major', labelsize=8)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Mag 1.0',
               markerfacecolor='teal', markeredgecolor='black',
               markersize=np.sqrt(mag_1_size), linewidth=0.1),
        Line2D([0], [0], marker='o', color='w', label='Mag 2.0',
               markerfacecolor='teal', markeredgecolor='black',
               markersize=np.sqrt(mag_2_size), linewidth=0.1),
        Line2D([0], [0], marker='^', color='w', label='Stations',
               markerfacecolor='darkorange', markeredgecolor='black',
               markersize=6, linewidth=0)
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              frameon=True, facecolor='white', edgecolor='gray', fontsize=8)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    plt.savefig('HeimdallCatalog_3D.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('HeimdallCatalog_3D.pdf', dpi=300, bbox_inches='tight', facecolor='white')

    # =============== 2D FIGURE ===============
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    ax.scatter(
        df['LONGITUDE'], df['LATITUDE'],
        s=df["MAGPLOTSIZES"], c='teal', alpha=0.8, marker='o',
        edgecolor='black', linewidths=0.5, label='Events')

    if args.plotstations and not stations_df.empty:
        ax.scatter(
            stations_df["longitude"],
            stations_df["latitude"],
            c="darkorange", alpha=0.7, s=60, marker="^",
            label="Stations", edgecolor='black', linewidths=0.5)

    ax.set_xlim(boundaries[0, :])
    ax.set_ylim(boundaries[1, :])
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.tick_params(axis='both', which='major', labelsize=8)

    ax.legend(handles=legend_elements, loc='upper left',
              frameon=True, facecolor='white', edgecolor='gray', fontsize=8)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig('HeimdallCatalog_2D.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('HeimdallCatalog_2D.pdf', dpi=300, bbox_inches='tight', facecolor='white')

    print("DONE!")


if __name__ == "__main__":
    main()
