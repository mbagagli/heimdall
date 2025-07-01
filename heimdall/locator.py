"""
Grid, source-PDF, and event-location helpers for the HEIMDALL toolkit.

This module lets you:

- Generate a Cartesian grid anchored to a geographic origin.
- Convert between geographic (lon, lat, depth) and local Cartesian
  (E, N, U) coordinates.
- Build synthetic 3-D probability-density functions (PDFs) for seismic
  sources with optional noise.
- Project a 3-D PDF onto the XY, XZ, and YZ planes for 2-D localisation
  targets.
- Locate an event and compute its origin time from a location PDF and
  station picks.

Classes:
    HeimdallGrid
        Builds the grid and offers coordinate conversions.
    HeimdallLocator
        Creates PDFs, 2-D projections, and helper plots.
    HeimdallEvent
        Extracts source coordinates and origin time from detections.

Example:
    >>> bounds = [10.0, 11.0, 44.0, 45.0, 0.0, 20.0]  # lon_min, lon_max,
    ...                                                # lat_min, lat_max,
    ...                                                # dep_min_km, dep_max_km
    >>> locator = HeimdallLocator(bounds,
    ...                            spacing_x=1.0,
    ...                            spacing_y=1.0,
    ...                            spacing_z=2.0)
    >>> pdf, local_coord = locator.create_source((10.5, 44.5, 5.0))
    >>> xy, xz, yz = locator.__make_label_projection__(pdf)

Dependencies:
    numpy
    scipy
    plotly  (via heimdall.plot)
    heimdall
"""

import sys
import numpy as np
from pathlib import Path
from scipy.stats import multivariate_normal
#
from heimdall import plot as GPLT
from heimdall import coordinates as GCRD
from heimdall import custom_logger as CL

logger = CL.init_logger(Path(sys.argv[0]).name, lvl="INFO")


class HeimdallGrid(object):
    def __init__(self, boundaries, dep_in_km=True,
                 reference_lon=None, reference_lat=None,
                 __default_reference__="lowest",  # or 'centre'
                 **kwargs):
        """Create a Cartesian grid referenced to a geographic anchor.

        Args:
            boundaries (list[float] | tuple[float, ...]): Sequence
                ``[lon_min, lon_max, lat_min, lat_max, dep_min, dep_max]``.
                Longitudes and latitudes are in decimal degrees; depths
                are in **km** when *dep_in_km* is ``True``.
            dep_in_km (bool, optional): If ``True`` (default) depth limits
                are interpreted as kilometres and converted to metres
                internally.
            reference_lon (float | None, optional): Longitude of the grid
                origin (0 m E, 0 m N).  If ``None`` the origin is chosen
                according to *__default_reference__*.
            reference_lat (float | None, optional): Latitude of the grid
                origin.  ``None`` behaves like *reference_lon*.
            __default_reference__ (str, optional): Either ``"centre"`` or
                ``"lowest"`` – choose the grid midpoint or the SW corner as
                default origin.  Defaults to ``"lowest"``.
            **kwargs: Ignored; accepted for forward compatibility.

        Attributes:
            reference_lon (float): Longitude of grid origin (dec.deg).
            reference_lat (float): Latitude of grid origin (dec.deg).
            converter (heimdall.coordinates.Coordinates): Helper for
                geographic <-> Cartesian conversion.
        """

        (self.lon_min, self.lon_max,
         self.lat_min, self.lat_max,
         self.dep_min, self.dep_max) = boundaries
        self.reference_lon = reference_lon
        self.reference_lat = reference_lat
        #
        if not reference_lon:
            if __default_reference__.lower() in ('centre', 'center'):
                self.reference_lon = self.lon_min + (self.lon_max-self.lon_min) / 2.0
            elif __default_reference__.lower() in ('lowest', 'corner'):
                self.reference_lon = self.lon_min

        if not reference_lat:
            if __default_reference__.lower() in ('centre', 'center'):
                self.reference_lat = self.lat_min + (self.lat_max-self.lat_min) / 2.0
            elif __default_reference__.lower() in ('lowest', 'corner'):
                self.reference_lat = self.lat_min

        if dep_in_km:
            self.dep_min *= 10**3
            self.dep_max *= 10**3
        #
        self.converter = GCRD.Coordinates(
                            self.reference_lat, self.reference_lon, ele0=0)
        logger.info("HeimdallGrid created - Ref.LON: %.4f  Ref.LAT: %.4f"
                    % (self.reference_lon, self.reference_lat))

    def get_grid_origin(self):
        """Return the geographic origin of the grid.

        Returns:
            tuple[float, float]: ``(reference_lon, reference_lat)`` in
            decimal degrees.
        """
        return (self.reference_lon, self.reference_lat)

    def convert_geo_dict(self, inarr):
        """Convert a *dict* of geographic tuples to Cartesian coordinates.

        Args:
            inarr (dict): Mapping ``key -> (lat, lon, *extra)``.  Extra values
                are passed through unchanged.

        Returns:
            dict: Same keys; values are Cartesian ``(E, N, U)`` metres for
            the first two elements, followed by the untouched extra fields.
        """
        assert isinstance(inarr, dict)
        return {kk: self.converter.geo2cart(*de[:2])
                for kk, de in inarr.items()}

    def convert_geo_list(self, inarr):
        """Convert a list/tuple of geographic points to Cartesian metres.

        Args:
            inarr (list | tuple): Iterable of ``(lat, lon, *extra)`` items.

        Returns:
            list[tuple[float, float, float]]: ``(E, N, U)`` metres for each
            input point, preserving additional elements.
        """
        assert isinstance(inarr, (tuple, list))
        return [self.converter.geo2cart(*de[:2]) for de in inarr]

    def convert_cart_list(self, inarr):
        """Convert Cartesian points back to geographic coordinates.

        Args:
            inarr (list | tuple): Iterable of ``(E, N, U)`` metre tuples.

        Returns:
            list[tuple[float, float, float]]: ``(lat, lon, ele)`` for each
            point (latitude/longitude in degrees, elevation in metres).
        """
        assert isinstance(inarr, (tuple, list))
        return [self.converter.cart2geo(*de) for de in inarr]

    def create_grid(self, spacing_x=2.5, spacing_y=2.5, spacing_z=2.5):
        """Generate regularly spaced X, Y, Z grid vectors.

        Args:
            spacing_x (float, optional): Easting step (metres). Default ``2.5``.
            spacing_y (float, optional): Northing step (metres). Default ``2.5``.
            spacing_z (float, optional): Depth step (metres). Default ``2.5``.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: 1-D arrays
            ``(x_grid, y_grid, z_grid)`` covering the full bounding box.
        """
        [(lon_min_km, lat_min_km, _), (lon_max_km, lat_max_km, _)] = (
                                        self.convert_geo_list(
                                            [(self.lat_min, self.lon_min),
                                             (self.lat_max, self.lon_max)]
                                        ))

        x_grid = np.arange(lon_min_km, lon_max_km+spacing_x, spacing_x)
        y_grid = np.arange(lat_min_km, lat_max_km+spacing_y, spacing_y)
        z_grid = np.arange(self.dep_min, self.dep_max+spacing_z, spacing_z)

        # -----------------------------------------------------------
        # Log a warning if the rounded edge values exceed the original grid boundaries
        if x_grid[-1] > lon_max_km or y_grid[-1] > lat_max_km or z_grid[-1] > self.dep_max:
            logger.warning("Rounded edge values exceed the original grid boundaries.")

        return (x_grid, y_grid, z_grid)


class HeimdallLocator(object):
    def __init__(self, grid_limits, spacing_x=1.0, spacing_y=1.0, spacing_z=1.0,
                 reference_point_lonlat=None):
        """Instantiate a locator with its working grid.

        Args:
            grid_limits (sequence): See :pyclass:`HeimdallGrid` *boundaries*.
            spacing_x (float, optional): Grid spacing in **km** east. Default 1.
            spacing_y (float, optional): Grid spacing in **km** north. Default 1.
            spacing_z (float, optional): Grid spacing in **km** depth. Default 1.
            reference_point_lonlat (tuple[float, float] | None, optional):
                Longitude/latitude to anchor the grid. If ``None`` the anchor
                is chosen automatically (centre of bounds).

        Attributes:
            grid (HeimdallGrid): Underlying geographic grid helper.
            grid_x (np.ndarray): X-axis sample locations (metres).
            grid_y (np.ndarray): Y-axis sample locations (metres).
            grid_z (np.ndarray): Z-axis sample locations (metres).
        """
        self.MT = 10**3
        self.KM = 10**-3
        #
        if isinstance(reference_point_lonlat, (list, tuple, np.ndarray)):
            self.grid = HeimdallGrid(grid_limits,
                                     reference_lon=reference_point_lonlat[0],
                                     reference_lat=reference_point_lonlat[1],
                                     )
        else:
            # It will automatically calculate the centre
            self.grid = HeimdallGrid(grid_limits)
        #
        self.spacing_x = spacing_x
        self.spacing_y = spacing_y
        self.spacing_z = spacing_z
        #
        (self.grid_x, self.grid_y, self.grid_z) = (
                    self.grid.create_grid(
                        self.spacing_x*self.MT,
                        self.spacing_y*self.MT,
                        self.spacing_z*self.MT,
                    ))

    @classmethod
    def __make_label__(cls, lon, lat, dep,
                       x_grid, y_grid, z_grid,
                       source_error=None,  # km
                       source_noise=None,
                       noise_max=None):
        """
        Creates a 3D probability density function (PDF) label with an optional
        random background noise.

        This method generates a 3D label representing a probability density
        function (PDF) in a grid defined by `x_grid`, `y_grid`, and `z_grid`. The
        method allows for adding random background noise with a specified maximum
        value and a point source error distribution with an isotropic covariance.
        The resulting label matrix is normalized to a range between 0 and 1.

        Args:
            lon (float): Longitude of the point source.
            lat (float): Latitude of the point source.
            dep (float): Depth of the point source.
            x_grid (numpy.ndarray): 1D array representing the x-coordinates of the
                grid.
            y_grid (numpy.ndarray): 1D array representing the y-coordinates of the
                grid.
            z_grid (numpy.ndarray): 1D array representing the z-coordinates of the
                grid.
            source_error (float, optional): Standard deviation (in grid units)
                of the isotropic covariance for the point source. If None or 0,
                no point source is added. Defaults to None.
            noise_max (float, optional): Maximum value for random background noise.
                If None or 0, no noise is added. Defaults to None.

        Returns:
            numpy.ndarray: A 3D array representing the normalized probability
            density function (PDF) with optional background noise and point source
            added, with values ranging from 0 to 1.
        """

        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

        # 1. INITIALIZE
        # Add random background noise if std_background is provided ...
        if noise_max and noise_max > 0.0:
            label_pdf = np.random.normal(loc=0.0,
                                         scale=1.0,
                                         size=X.shape)
            label_pdf = (label_pdf - label_pdf.min()) / (label_pdf.max() - label_pdf.min())
            label_pdf *= noise_max

        # ... else, Initialize the final label_pdf with zeros
        else:
            label_pdf = np.zeros(X.shape)

        # 2. ADD SOURCE
        if source_error and source_error > 0:
            # Create the point source PDF if source_error
            covariance_matrix = np.diag([source_error**2] * 3)  # isotropic covariance
            pdf = multivariate_normal(mean=(lon, lat, dep),
                                      cov=covariance_matrix)
            grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
            label_pdf_point = pdf.pdf(grid_points).reshape(X.shape)

            # Normalize between 0 and 1
            label_pdf_point /= label_pdf_point.max()

            # Add the point source PDF to the final label
            label_pdf = np.maximum(label_pdf, label_pdf_point)

            # 3. ADD NOISE CONFINED TO SOURCE ERROR RADIUS
            if source_noise and source_noise > 0.0:
                # Calculate Euclidean distance from the source point to each grid point
                distances = np.sqrt((X - lon) ** 2 + (Y - lat) ** 2 + (Z - dep) ** 2)

                # Create a mask where distances are within the source_error radius
                within_radius = distances <= source_error

                # Generate noise only within this confined region
                confined_noise = np.random.normal(loc=0.0, scale=1.0, size=X.shape)
                confined_noise = (confined_noise - confined_noise.min()) / (
                                  confined_noise.max() - confined_noise.min())
                confined_noise *= source_noise

                # Apply the noise only where distances are within the source_error radius
                label_pdf[within_radius] = np.maximum(
                                                label_pdf[within_radius],
                                                confined_noise[within_radius])
        return label_pdf

    @classmethod
    def __make_label_projection__(cls, input_pdf, normalize=True):
        """
        Projects the 3D PDF onto the XY, XZ, and YZ planes
        and normalizes the values between 0 and 1.

        Args:
            input_pdf (numpy.ndarray): The 3D array representing the PDF.
            normalize (bool, optional): Whether to normalize the projections
                between 0 and 1. Defaults to True.

        Returns:
            tuple: Three 2D numpy arrays representing the XY, XZ, and YZ
            projections.
        """
        # Project the 3D PDF onto each plane
        xy_projection = np.max(input_pdf, axis=2)  # Collapse depth (DP) for XY plane
        xz_projection = np.max(input_pdf, axis=1)  # Collapse latitude (LT) for XZ plane
        yz_projection = np.max(input_pdf, axis=0)  # Collapse longitude (LN) for YZ plane

        if normalize:
            # Normalize each projection between 0 and 1
            xy_projection /= np.max(xy_projection) if np.max(xy_projection) != 0 else 1
            xz_projection /= np.max(xz_projection) if np.max(xz_projection) != 0 else 1
            yz_projection /= np.max(yz_projection) if np.max(yz_projection) != 0 else 1

        return (xy_projection, xz_projection, yz_projection)

    def create_source(self, source_coord, std_err_km=1.5, noise_max=None,
                      dep_in_km=True, stations_coordinates=None, plot=False):
        """Build a 3-D PDF for a single seismic source.

        Args:
            source_coord (tuple[float, float, float]): ``(lon, lat, dep)``
                longitude & latitude in degrees, depth in **km** unless
                *dep_in_km* is ``False``.
            std_err_km (float, optional): 1-σ radius of the Gaussian source
                (kilometres). Default ``1.5``.
            noise_max (float | None, optional): Max amplitude of uniform noise
                (0–1).  ``None`` or ``0`` disables background noise.
            dep_in_km (bool, optional): When ``True`` (default) *dep* and
                *std_err_km* are interpreted in km and converted to metres.
            stations_coordinates (dict | None, optional): Not used at present.
            plot (bool, optional): If ``True`` produce a Plotly 3-D figure.

        Returns:
            tuple: ``(pdf_3d, source_coord_local)`` where *pdf_3d* is the
            normalised 3-D array and *source_coord_local* is the point in
            grid metres.
        """
        (eqlon, eqlat, eqdep) = source_coord
        if dep_in_km:
            eqdep *= self.MT
        #
        logger.info("Creating PDF for source: %.4f  %.4f  %.1f" % (
                    eqlon, eqlat, eqdep))

        [(eqlon_ongrid, eqlat_ongrid, _), ] = self.grid.convert_geo_list(
                                                      [(eqlat, eqlon),])
        label_pdf = self.__make_label__(
                                eqlon_ongrid, eqlat_ongrid, eqdep,
                                self.grid_x,
                                self.grid_y,
                                self.grid_z,
                                #
                                source_error=std_err_km*self.MT,
                                noise_max=noise_max)

        if plot:
            fig = GPLT.plot_source_pdf(self.grid_x,
                                       self.grid_y,
                                       self.grid_z,
                                       label_pdf,
                                       isosurface_values=[0.9, ],
                                       title="Event: %.3f / %.3f / %.1f (%.1f / %.1f)" % (
                                            eqlon, eqlat, eqdep,
                                            eqlon_ongrid, eqlat_ongrid))
            fig.write_image(
                    "Source_%.3f_%.3f_%.1f.png" % (eqlon, eqlat, eqdep))
            fig.show()

        return (label_pdf, (eqlon_ongrid, eqlat_ongrid, eqdep))

    def create_source_images(
                    self, source_coord, std_err_km=1.5, noise_max=None,
                    source_noise=None, dep_in_km=True, stations_coordinates=None,
                    plot=False, store_file=None, additional_title_figure=""):
        """Create 3-D PDF plus its *XY*, *XZ*, *YZ* projections.

        Args:
            source_coord (tuple[float, float, float]): Geographic source
                coordinates in degrees / km.
            std_err_km (float, optional): Gaussian 1-sigma radius. Default 1.5 km.
            noise_max (float | None, optional): Background noise amplitude.
            source_noise (float | None, optional): Confined noise amplitude
                within *std_err_km*.
            dep_in_km (bool, optional): Treat *dep* as km when ``True``.
            stations_coordinates (dict | None, optional): Reserved for future
                plotting features.
            plot (bool, optional): Plot the three images when ``True``.
            store_file (str | None, optional): PNG path for saved figure.
                Auto-generated when ``None``.
            additional_title_figure (str, optional): Extra text appended to
                the plot title.

        Returns:
            tuple: ``(pdf_3d, xy_img, xz_img, yz_img, source_coord_local)``
            where *xy_img*, *xz_img*, *yz_img* have shapes matching the
            corresponding grid planes.
        """
        (eqlon, eqlat, eqdep) = source_coord
        if dep_in_km:
            eqdep *= self.MT
        #
        logger.info("Creating PDF for source: %.4f  %.4f  %.1f" % (
                    eqlon, eqlat, eqdep))

        [(eqlon_ongrid, eqlat_ongrid, _), ] = self.grid.convert_geo_list(
                                                      [(eqlat, eqlon),])
        label_pdf = self.__make_label__(
                                eqlon_ongrid, eqlat_ongrid, eqdep,
                                self.grid_x,
                                self.grid_y,
                                self.grid_z,
                                #
                                source_error=std_err_km*self.MT,
                                source_noise=source_noise,
                                noise_max=noise_max)
        #
        (xy_image, xz_image, yz_image) = self.__make_label_projection__(
                                label_pdf,
                                # If no point-source, the noise will be normalized
                                # to one, vanishing the effect of MAX_noise before.
                                # --> KEEP IT FALSE
                                normalize=False)

        if plot:
            (fig, _) = GPLT.plot_source_pdf_images_simple(
                                self.grid_x,
                                self.grid_y,
                                self.grid_z,
                                xy_image, xz_image, yz_image,
                                reference_locations=[(eqlon_ongrid,
                                                      eqlat_ongrid,
                                                      eqdep),],
                                figtitle=(
                                    "Event: %.3f / %.3f / %.1f (%.1f / %.1f)" +
                                    additional_title_figure) % (
                                    eqlon, eqlat, eqdep,
                                    eqlon_ongrid, eqlat_ongrid))
            if not store_file:
                store_file = "Source_%.3f_%.3f_%.1f.png" % (eqlon, eqlat, eqdep)
            fig.savefig(store_file,
                        dpi=310, bbox_inches='tight', facecolor='w')
        #
        return (label_pdf, xy_image, xz_image, yz_image,
                (eqlon_ongrid, eqlat_ongrid, eqdep))

    def get_grid(self):
        """Return the locator’s X, Y, Z grid vectors."""
        return (self.grid_x, self.grid_y, self.grid_z)

    def get_grid_reference(self):
        """Return longitude/latitude of the grid origin."""
        return self.grid.get_grid_origin()

    @classmethod
    def sort_stations_by_distance(cls, lon0, lat0, stations_dict):
        """Sort station keys by great-circle distance to a reference point.

        Args:
            lon0 (float): Reference longitude (degrees).
            lat0 (float): Reference latitude (degrees).
            stations_dict (dict): ``{"STA": (lon, lat, ele), ...}``.

        Returns:
            list[str]: Station names ordered nearest → farthest.
        """
        return GCRD.sort_stations_by_distance(lon0, lat0, stations_dict)


class HeimdallEvent(object):
    """Container for all information related to a single detected event.

    !!!  UNDER DEVELOPMENT  !!!

    The class bundles station detections, a location PDF, grid utilities,
    and timing information.  It can derive the most likely epicentre and
    origin time directly from the PDF.

    Notes:
        Depths are converted between metres and kilometres internally
        using :pyattr:`KM` (= 1e-3) and :pyattr:`MT` (= 1e3) factors.
    """

    def __init__(self, stat_detects, loc_pdf, locator, gnn,
                 reference_utc, df, half_space_velocity=6.00):
        """
        Initializes a HeimdallEvent instance.

        Args:
            stat_detects (list): List of station detections.
            loc_pdf (np.ndarray): Location probability density function.
            locator (object): Locator object containing grid information.
            gnn (dict): Graph Neural Network object containing station information.
            reference_utc (datetime): Reference UTC time for the event.
            df (float): Frequency sampling interval.
            half_space_velocity (float, optional): Velocity of half-space (km/s). Default is 6.00.
        """
        self.MT = 10**3
        self.KM = 10**-3
        #
        self.stat_detects = stat_detects
        self.loc_pdf = loc_pdf
        self.locator = locator
        self.gnn = gnn
        self.reference_utc = reference_utc
        self.df = df
        self.half_space_velocity = half_space_velocity
        #
        self.location = []

    @classmethod
    def locate_only_from_pdf(cls, loc_pdf, locator):
        """
        Stand-alone function to locate an event based only on the PDF matrix and locator object.

        Args:
            loc_pdf (np.ndarray): Location probability density function.
            locator (object): Locator object containing grid information.

        Returns:
            list: Event location details.
        """
        # Find the maximum location in the PDF
        max_index = np.unravel_index(np.argmax(loc_pdf), loc_pdf.shape)
        xm = locator.grid_x[max_index[0]]
        ym = locator.grid_y[max_index[1]]
        zm = locator.grid_z[max_index[2]]
        source_coord_local = (xm, ym, zm)

        # Convert local coordinates to geographic coordinates
        (_source_coord,) = locator.grid.convert_cart_list([source_coord_local])

        # Construct the location list
        source_coord = (_source_coord[1], _source_coord[0],
                        source_coord_local[2] * 10**-3)  # Convert to KM
        return [*source_coord, *source_coord_local]

    def __get_earliest_arrival_station__(self):
        """
        Finds the earliest arrival station and its coordinates.

        Returns:
            tuple: A tuple containing the station coordinates (tuple), station name (str),
                   and arrival index (int).
        """
        min_value = float('inf')
        min_tuple = None
        for item in self.stat_detects:
            for sub_item in item[1]:
                if sub_item[0] < min_value:
                    min_value = sub_item[0]
                    min_tuple = item
        (_station_idx, _arrival_idx) = min_tuple
        station_name = list(self.gnn["stations_order"].item().keys())[_station_idx]
        station_coordinate = self.gnn["stations_coordinate"].item()[station_name]
        return (station_coordinate, station_name, _arrival_idx)

    def __extract_location__(self):
        """
        Extracts the location of the event based on the maximum value in the location PDF.

        Returns:
            tuple: A tuple containing the coordinates (tuple), the maximum index (tuple),
                   and the maximum value (float).
        """
        max_index = np.unravel_index(np.argmax(self.loc_pdf), self.loc_pdf.shape)
        max_value = self.loc_pdf[max_index]
        #
        xm = self.locator.grid_x[max_index[0]]
        ym = self.locator.grid_y[max_index[1]]
        zm = self.locator.grid_z[max_index[2]]
        return ((xm, ym, zm), max_index, max_value)

    def locate_event(self):
        """
        Locates the event by determining the source and station coordinates, and calculating
        the event's origin time.

        Updates:
            self.location (list): List containing event coordinates in different formats
                                  and origin time.
        """
        # ---------  Get Earliest arrival
        (station_coord, stat_name, arrival_time_idx) = self.__get_earliest_arrival_station__()

        # --------- STATION
        (_station_coord_local, ) = self.locator.grid.convert_geo_list([(
                                                        station_coord[1],
                                                        station_coord[0])
                                                    ])
        station_coord_local = (
            _station_coord_local[0], _station_coord_local[1], -station_coord[2]
        )

        # --------- EVENT
        (source_coord_local, _, _) = self.__extract_location__()
        (_source_coord, ) = self.locator.grid.convert_cart_list([
                                source_coord_local
            ])

        source_coord = (_source_coord[1], _source_coord[0],
                        source_coord_local[2]*self.KM)

        # --------- Calculate the Euclidean distance
        distance = np.linalg.norm(
                        np.array(source_coord_local) -
                        np.array(station_coord_local))
        time = distance*self.KM / self.half_space_velocity  # seconds
        source_ot = (self.reference_utc + arrival_time_idx[0][0]/self.df) - time

        # Create location --> LON, LAT, DEP, X_MT, Y_MT, Z_MT, OT
        self.location = [*source_coord, *source_coord_local, source_ot]

    def get_results(self):
        """Return the final location results.

        Returns:
            list: The cached *self.location* list.  When empty, a warning
            is logged and an empty list is returned instead.
        """
        if self.location:
            return self.location
        else:
            logger.warning("No LOCATION found")
