import math
import numpy as np

"""
This module provides a class for converting geographical coordinates to Cartesian coordinates
using the WGS84 ellipsoid model. It supports conversion between latitude, longitude, and elevation
(in degrees and kilometers) to east, north, and up coordinates (in meters).

Classes:
    Coordinates: Converts geographic coordinates (latitude, longitude, elevation) to Cartesian
    coordinates (east, north, up) and vice versa, based on a defined reference point using the WGS84 ellipsoid.

Usage:
    Instantiate the Coordinates class with a reference geographical point (latitude, longitude, elevation).
    Convert any point from geographic to Cartesian coordinates or from Cartesian back to geographic coordinates.
    Example:
        region = latlon2cart.Coordinates(lat_ref, lon_ref, ele_ref)
        E, N, U = region.geo2cart(lat_i, lon_i, ele_i)
        lat_i, lon_i, ele_i = region.cart2geo(E, N, U)

Author:
    Francesco Grigoli
    Department of Earth Sciences
    University of Pisa
    Italy
"""


class Coordinates(object):
    """
    A class for converting between geographical coordinates (Latitude, Longitude, Elevation)
    and Cartesian coordinates (East, North, Up) based on a specified reference point using the
    WGS84 ellipsoid model.

    Attributes:
        lat0 (float): The reference latitude in degrees.
        lon0 (float): The reference longitude in degrees.
        ele0 (float): The reference elevation in kilometers.

    Methods:
        geo2cart(lat, lon, ele=0, geo2enu=True): Converts geographical coordinates to Cartesian coordinates.
        cart2geo(E, N, U): Converts Cartesian coordinates back to geographical coordinates.
    """

    def __init__(self, lat0, lon0, ele0=0):
        """
        Initializes the Coordinates object with a reference geographical point.

        Args:
            lat0 (float): Reference latitude in degrees.
            lon0 (float): Reference longitude in degrees.
            ele0 (float, optional): Reference elevation in kilometers. Defaults to 0.
        """

        # ==================================  DO NOT CHANGE
        self._deg2rad = np.pi / 180.0
        self._rad2deg = 180.0 / np.pi
        self._km2m = 1000
        self._semi_major_axis = 6378137.000000
        self._semi_minor_axis = 6356752.314245
        self._eccentricity_squared = 1 - (self._semi_minor_axis /
                                          self._semi_major_axis)**2
        # ======================================================
        X0, Y0, Z0 = self.geo2cart(lat0, lon0, ele0, geo2enu=False)
        self.lat0 = lat0
        self.lon0 = lon0
        self.ele0 = ele0
        self.X0 = X0
        self.Y0 = Y0
        self.Z0 = Z0

    def __conv2enu(self, X, Y, Z):
        lon0rad = self.lon0*self._deg2rad
        lat0rad = self.lat0*self._deg2rad

        DX = X-self.X0
        DY = Y-self.Y0
        DZ = Z-self.Z0

        E = DY*np.cos(lon0rad)-DX*np.sin(lon0rad)
        N = DZ*np.cos(lat0rad)-DY*np.sin(lat0rad) * \
            np.sin(lon0rad)-DX*np.sin(lat0rad)*np.cos(lon0rad)
        U = DZ*np.sin(lat0rad)+DY*np.cos(lat0rad) * \
            np.sin(lon0rad)+DX*np.cos(lat0rad)*np.cos(lon0rad)
        return E, N, U

    def __enu2geo(self, E, N, U):

        lon0rad = self.lon0*self._deg2rad
        lat0rad = self.lat0*self._deg2rad

        X = (U*np.cos(lat0rad)*np.cos(lon0rad)-E*np.sin(lon0rad) -
             N*np.sin(lat0rad)*np.cos(lon0rad))+self.X0
        Y = (E*np.cos(lon0rad)-N*np.sin(lat0rad)*np.sin(lon0rad) +
             U*np.cos(lat0rad)*np.sin(lon0rad))+self.Y0
        Z = (N*np.cos(lat0rad)+U*np.sin(lat0rad))+self.Z0

        return X, Y, Z

    def geo2cart(self, lat, lon, ele=0, geo2enu=True):
        """
        Converts geographical coordinates (latitude, longitude, elevation) to Cartesian coordinates
        (east, north, up).

        Args:
            lat (float): Latitude in degrees to convert.
            lon (float): Longitude in degrees to convert.
            ele (float, optional): Elevation in kilometers to convert.
                Defaults to 0.
            geo2enu (bool, optional): If True, convert output to local
                                east, north, up coordinates. If False,
                                returns global X, Y, Z coordinates.
                                Defaults to True.

        Returns:
            tuple: A tuple (E, N, U) of east, north, and up coordinates
                in meters if geo2enu is True. Otherwise, a tuple
                (X, Y, Z) of global Cartesian coordinates.
        """

        lat = lat*self._deg2rad
        lon = lon*self._deg2rad
        ele = ele*self._km2m

        N = self._semi_major_axis / np.sqrt(1-self._eccentricity_squared*(np.sin(lat)**2))

        X = (N+ele)*np.cos(lat)*np.cos(lon)
        Y = (N+ele)*np.cos(lat)*np.sin(lon)
        Z = ((1-self._eccentricity_squared)*N+ele)*np.sin(lat)

        if geo2enu:
            E, N, U = self.__conv2enu(X, Y, Z)
            return E, N, U
        else:
            return X, Y, Z

    def cart2geo(self, E, N, U):
        """
        Converts Cartesian coordinates (east, north, up) back to
        geographical coordinates (latitude, longitude, elevation).

        Args:
            E (float): East coordinate in meters.
            N (float): North coordinate in meters.
            U (float): Up coordinate in meters.

        Returns:
            tuple: A tuple (lat, lon, ele) where lat and lon are in
                degrees, and ele is in kilometers.
        """
        X, Y, Z = self.__enu2geo(E, N, U)
        e = (self._semi_major_axis**2-self._semi_minor_axis**2)/self._semi_minor_axis**2
        p = np.sqrt(X**2+Y**2)
        F = 54*(self._semi_minor_axis**2)*Z**2
        G = p**2+(1-self._eccentricity_squared)*Z**2-self._eccentricity_squared * \
            (self._semi_major_axis**2-self._semi_minor_axis**2)
        c = ((self._eccentricity_squared**2)*F*p**2)/G**3
        s = np.cbrt(1+c+np.sqrt(c**2+2*c))
        k = s+1+(1/s)
        P = F/(3*k**2*G**2)
        Q = np.sqrt(1+2*(self._eccentricity_squared**2)*P)
        r0_1 = -((P*self._eccentricity_squared*p)/(1+Q))
        r0_2 = 0.5*self._semi_major_axis**2*(1+(1/Q))
        r0_3 = (P*(1-self._eccentricity_squared)*Z**2)/(Q*(1+Q))
        r0_4 = 0.5*P*p**2
        r0 = r0_1+np.sqrt(r0_2-r0_3-r0_4)
        U = np.sqrt((p-self._eccentricity_squared*r0)**2+Z**2)
        V = np.sqrt((p-self._eccentricity_squared*r0) **
                    2+(1-self._eccentricity_squared)*Z**2)
        z0 = (Z*self._semi_minor_axis**2)/(V*self._semi_major_axis)

        lat = np.arctan((Z+e*z0)/p)*self._rad2deg
        lon = np.arctan2(Y, X)*self._rad2deg
        ele = (U*(1-(self._semi_minor_axis**2)/(V*self._semi_major_axis)))/self._km2m

        return lat, lon, ele


def haversine_distance(lon1, lat1, lon2, lat2):
    """Compute the great-circle distance between two geographic points.

    The calculation is performed with the haversine formula and assumes
    a spherical Earth with radius 6 371 km.

    Args:
        lon1 (float): Longitude of the first point (decimal degrees,
            east positive).
        lat1 (float): Latitude of the first point (decimal degrees,
            north positive).
        lon2 (float): Longitude of the second point (decimal degrees).
        lat2 (float): Latitude of the second point (decimal degrees).

    Returns:
        float: Distance between the two points in kilometers.
    """
    # Convert latitude and longitude from degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    # Radius of Earth in kilometers. Use 6371 for kilometers or 3956 for miles
    r = 6371
    return c * r


def sort_stations_by_distance(lon0, lat0, stations_dict):
    """Order stations by epicentral distance.

    The function computes the haversine distance from a reference
    location to every station in *stations_dict* and returns a new
    dictionary mapping each station code to its ordinal rank
    (0 = closest).

    Args:
        lon0 (float): Reference longitude (decimal degrees).
        lat0 (float): Reference latitude (decimal degrees).
        stations_dict (dict): Mapping ``station_id -> (lon, lat, *extra)``,
            where only the first two elements of the tuple (lon, lat) are
            used for the distance calculation.

    Returns:
        dict: ``{station_id: rank}`` where *rank* is an int starting
        at 0 for the nearest station and increasing with distance.

    Examples:
        >>> stations = {"NET.STA1": (-21.5, 64.0),
        ...             "NET.STA2": (-22.0, 63.9)}
        >>> sort_stations_by_distance(-21.8, 64.1, stations)
        {'NET.STA1': 0, 'NET.STA2': 1}
    """
    # Create a dictionary with the station key and its calculated distance to the point (lon0, lat0)
    distances = {key: haversine_distance(lon0, lat0, *coords[:2]) for key, coords in stations_dict.items()}
    # Sort the dictionary by distance and create a new dictionary with the same structure but sorted keys
    sorted_keys = sorted(distances, key=distances.get)
    # sorted_stations_dict = {key: stations_dict[key] for key in sorted_keys}
    sorted_stations_dict = {key: xx for xx, key in enumerate(sorted_keys)}
    return sorted_stations_dict


# # =========== For testing --> import module
# if __name__ == '__main__':
#     latref = 36.117
#     lonref = -117.8536
#     eleref = 0.
#     test = Coordinates(latref, lonref, eleref)
#     #
#     lat1 = 35.536
#     lon1 = -118.1445
#     ele1 = 5.  # --> possible problem, returning 4618, instead of 5000
#     E, N, U = test.geo2cart(lat1, lon1, ele1)
#     lat2, lon2, ele2 = test.cart2geo(E, N, U)
#     print(U)
#     print('Lat %8.5f %8.5f, Lon %8.5f %8.5f, Ele %8.5f %8.5f' %
#           (lat1, lat2, lon1, lon2, ele1, ele2))

# # =========== For testing --> IPYTHON
# from heimdall import coordinates as DE
# latref = 36.117
# lonref = -117.8536
# eleref = 0.
# test = DE.Coordinates(latref, lonref, eleref)
# #
# lat1 = 35.536
# lon1 = -118.1445
# ele1 = 5.  # --> possible problem, returning 4618, instead of 5000
# E, N, U = test.geo2cart(lat1, lon1, ele1)
# lat2, lon2, ele2 = test.cart2geo(E, N, U)
# print(U)
# print('Lat %8.5f %8.5f, Lon %8.5f %8.5f, Ele %8.5f %8.5f' %
#       (lat1, lat2, lon1, lon2, ele1, ele2))


# # =========== For testing --> EPICENTRAL DIST
# stations_coordinates = {
#     '2C.BIT06': (-21.26694, 64.04884, 403.3),
#     '2C.BLK22': (-21.47562, 64.04066, 294.8),
#     # add more stations as per your actual data...
# }

# # Define your point of interest
# lon0, lat0 = (-21.26694, 64.04884)  # Example point

# # Get sorted dictionary
# sorted_stations = sort_stations_by_distance(lon0, lat0, stations_coordinates)
# print(sorted_stations)
