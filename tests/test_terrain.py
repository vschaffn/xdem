from __future__ import annotations

import os
import re
import tempfile
import warnings

import geoutils as gu
import numpy as np
import pytest
import richdem as rd
from geoutils.raster import RasterType

import xdem
from xdem._typing import MArrayf, NDArrayf

xdem.examples.download_longyearbyen_examples()

PLOT = True


def run_gdaldem(filepath: str, processing: str, options: str | None = None) -> MArrayf:
    """Run GDAL's DEMProcessing and return the read numpy array."""
    # Rasterio strongly recommends against importing gdal along rio, so this is done here instead.
    from osgeo import gdal

    gdal.UseExceptions()

    # Converting string into gdal processing options here to avoid import gdal outside this function:
    # Riley or Wilson for Terrain Ruggedness, and Zevenberg or Horn for slope, aspect and hillshade
    gdal_option_conversion = {
        "Riley": gdal.DEMProcessingOptions(alg="Riley"),
        "Wilson": gdal.DEMProcessingOptions(alg="Wilson"),
        "Zevenberg": gdal.DEMProcessingOptions(alg="ZevenbergenThorne"),
        "Horn": gdal.DEMProcessingOptions(alg="Horn"),
        "hillshade_Zevenberg": gdal.DEMProcessingOptions(azimuth=315, altitude=45, alg="ZevenbergenThorne"),
        "hillshade_Horn": gdal.DEMProcessingOptions(azimuth=315, altitude=45, alg="Horn"),
    }

    if options is None:
        gdal_option = gdal.DEMProcessingOptions(options=None)
    else:
        gdal_option = gdal_option_conversion[options]

    temp_dir = tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir.name, "output.tif")
    gdal.DEMProcessing(
        destName=temp_path,
        srcDS=filepath,
        processing=processing,
        options=gdal_option,
    )

    data = gu.Raster(temp_path).data
    temp_dir.cleanup()
    return data


class TestTerrainAttribute:
    filepath = xdem.examples.get_path("longyearbyen_ref_dem")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Parse metadata")
        dem = xdem.DEM(filepath, silent=True)

    @pytest.mark.parametrize(
        "attribute",
        [
            "slope_Horn",
            "aspect_Horn",
            "hillshade_Horn",
            "slope_Zevenberg",
            "aspect_Zevenberg",
            "hillshade_Zevenberg",
            "tri_Riley",
            "tri_Wilson",
            "tpi",
            "roughness",
        ],
    )  # type: ignore
    def test_attribute_functions_against_gdaldem(self, attribute: str) -> None:
        """
        Test that all attribute functions give the same results as those of GDALDEM within a small tolerance.

        :param attribute: The attribute to test (e.g. 'slope')
        """

        functions = {
            "slope_Horn": lambda dem: xdem.terrain.slope(dem.data, resolution=dem.res, degrees=True),
            "aspect_Horn": lambda dem: xdem.terrain.aspect(dem.data, degrees=True),
            "hillshade_Horn": lambda dem: xdem.terrain.hillshade(dem.data, resolution=dem.res),
            "slope_Zevenberg": lambda dem: xdem.terrain.slope(
                dem.data, resolution=dem.res, method="ZevenbergThorne", degrees=True
            ),
            "aspect_Zevenberg": lambda dem: xdem.terrain.aspect(dem.data, method="ZevenbergThorne", degrees=True),
            "hillshade_Zevenberg": lambda dem: xdem.terrain.hillshade(
                dem.data, resolution=dem.res, method="ZevenbergThorne"
            ),
            "tri_Riley": lambda dem: xdem.terrain.terrain_ruggedness_index(dem.data, method="Riley"),
            "tri_Wilson": lambda dem: xdem.terrain.terrain_ruggedness_index(dem.data, method="Wilson"),
            "tpi": lambda dem: xdem.terrain.topographic_position_index(dem.data),
            "roughness": lambda dem: xdem.terrain.roughness(dem.data),
        }

        # Writing dictionary options here to avoid importing gdal outside the dedicated function
        gdal_processing_attr_option = {
            "slope_Horn": ("slope", "Horn"),
            "aspect_Horn": ("aspect", "Horn"),
            "hillshade_Horn": ("hillshade", "hillshade_Horn"),
            "slope_Zevenberg": ("slope", "Zevenberg"),
            "aspect_Zevenberg": ("aspect", "Zevenberg"),
            "hillshade_Zevenberg": ("hillshade", "hillshade_Zevenberg"),
            "tri_Riley": ("TRI", "Riley"),
            "tri_Wilson": ("TRI", "Wilson"),
            "tpi": ("TPI", None),
            "roughness": ("Roughness", None),
        }

        # Copy the DEM to ensure that the inter-test state is unchanged, and because the mask will be modified.
        dem = self.dem.copy()

        # Derive the attribute using both GDAL and xdem
        attr_xdem = functions[attribute](dem).squeeze()
        attr_gdal = run_gdaldem(
            self.filepath,
            processing=gdal_processing_attr_option[attribute][0],
            options=gdal_processing_attr_option[attribute][1],
        )

        # For hillshade, we round into an integer to match GDAL's output
        if attribute in ["hillshade_Horn", "hillshade_Zevenberg"]:
            with warnings.catch_warnings():
                # Normal that a warning would be raised here, so we catch it
                warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
                attr_xdem = attr_xdem.astype("int").astype("float32")

        # We compute the difference and keep only valid values
        diff = (attr_xdem - attr_gdal).filled(np.nan)
        diff_valid = diff[np.isfinite(diff)]

        try:
            # Difference between xdem and GDAL attribute
            # Mean of attribute values to get an order of magnitude of the attribute unit
            magn = np.nanmean(np.abs(attr_xdem))

            # Check that the attributes are similar within a tolerance of a thousandth of the magnitude
            # For instance, slopes have an average magnitude of around 30 deg, so the tolerance is 0.030 deg
            if attribute in ["hillshade_Horn", "hillshade_Zevenberg"]:
                # For hillshade, check 0 or 1 difference due to integer rounding
                assert np.all(np.logical_or(diff_valid == 0.0, np.abs(diff_valid) == 1.0))

            elif attribute in ["aspect_Horn", "aspect_Zevenberg"]:
                # For aspect, check the tolerance within a 360 degree modulo due to the circularity of the variable
                diff_valid = np.mod(np.abs(diff_valid), 360)
                assert np.all(np.minimum(diff_valid, np.abs(360 - diff_valid)) < 10 ** (-3) * magn)
            else:
                # All attributes other than hillshade and aspect are non-circular floats, so we check within a tolerance
                assert np.all(np.abs(diff_valid < 10 ** (-3) * magn))

        except Exception as exception:

            if PLOT:
                import matplotlib.pyplot as plt

                # Plotting the xdem and GDAL attributes for comparison (plotting "diff" can also help debug)
                plt.subplot(121)
                plt.imshow(attr_gdal.squeeze())
                plt.colorbar()
                plt.subplot(122)
                plt.imshow(attr_xdem.squeeze())
                plt.colorbar()
                plt.show()

            raise exception

        # Introduce some nans
        rng = np.random.default_rng(42)
        dem.data.mask = np.zeros_like(dem.data, dtype=bool)
        dem.data.mask.ravel()[rng.choice(dem.data.size, 50000, replace=False)] = True

        # Validate that this doesn't raise weird warnings after introducing nans.
        functions[attribute](dem)

    @pytest.mark.parametrize(
        "attribute",
        ["slope_Horn", "aspect_Horn", "hillshade_Horn", "curvature", "profile_curvature", "planform_curvature"],
    )  # type: ignore
    def test_attribute_functions_against_richdem(self, attribute: str) -> None:
        """
        Test that all attribute functions give the same results as those of RichDEM within a small tolerance.

        :param attribute: The attribute to test (e.g. 'slope')
        """

        def _raster_to_rda(rst: RasterType) -> rd.rdarray:
            """
            Get georeferenced richDEM array from geoutils.Raster
            :param rst: DEM as raster
            :return: DEM
            """
            arr = rst.data.filled(rst.nodata).squeeze()
            rda = rd.rdarray(arr, no_data=rst.nodata)
            rda.geotransform = rst.transform.to_gdal()

            return rda

        def _get_terrainattr_richdem(rst: RasterType, attribute: str = "slope_radians") -> NDArrayf:
            """
            Derive terrain attribute for DEM opened with rasterio. One of "slope_degrees", "slope_percentage", "aspect",
            "profile_curvature", "planform_curvature", "curvature" and others (see RichDEM documentation).
            :param rst: DEM as raster
            :param attribute: RichDEM terrain attribute
            :return:
            """
            rda = _raster_to_rda(rst)
            terrattr = rd.TerrainAttribute(rda, attrib=attribute)
            terrattr[terrattr == terrattr.no_data] = np.nan

            return np.array(terrattr)

        def get_terrain_attribute_richdem(
                dem: NDArrayf | MArrayf | RasterType,
                attribute: str | list[str],
                degrees: bool = True,
                hillshade_altitude: float = 45.0,
                hillshade_azimuth: float = 315.0,
                hillshade_z_factor: float = 1.0,
        ) -> NDArrayf | list[NDArrayf] | RasterType | list[RasterType]:
            """
            Derive one or multiple terrain attributes from a DEM using Richdem.

            Attributes:
            * 'slope': The slope in degrees or radians (degs: 0=flat, 90=vertical). Default method: "Horn".
            * 'aspect': The slope aspect in degrees or radians (degs: 0=N, 90=E, 180=S, 270=W).
            * 'hillshade': The shaded slope in relation to its aspect.
            * 'curvature': The second derivative of elevation (the rate of slope change per pixel), multiplied by 100.
            * 'planform_curvature': The curvature perpendicular to the direction of the slope, multiplied by 100.
            * 'profile_curvature': The curvature parallel to the direction of the slope, multiplied by 100.

            :param dem: The DEM to analyze.
            :param attribute: The terrain attribute(s) to calculate.
            :param degrees: Convert radians to degrees?
            :param hillshade_altitude: The shading altitude in degrees (0-90°). 90° is straight from above.
            :param hillshade_azimuth: The shading azimuth in degrees (0-360°) going clockwise, starting from north.
            :param hillshade_z_factor: Vertical exaggeration factor.

            :raises ValueError: If the inputs are poorly formatted or are invalid.

            :returns: One or multiple arrays of the requested attribute(s)
            """

            # Validate and format the inputs
            if isinstance(attribute, str):
                attribute = [attribute]

            if not isinstance(dem, gu.Raster):
                # Here, maybe we could pass the geotransform based on the resolution, and add a "default" projection as
                # this is mandated but likely not used by the rdarray format of RichDEM...
                # For now, not supported
                raise ValueError("To derive RichDEM attributes, the DEM passed must be a Raster object")

            if (hillshade_azimuth < 0.0) or (hillshade_azimuth > 360.0):
                raise ValueError(
                    f"Azimuth must be a value between 0 and 360 degrees (given value: {hillshade_azimuth})")
            if (hillshade_altitude < 0.0) or (hillshade_altitude > 90):
                raise ValueError("Altitude must be a value between 0 and 90 degrees (given value: {altitude})")
            if (hillshade_z_factor < 0.0) or not np.isfinite(hillshade_z_factor):
                raise ValueError(f"z_factor must be a non-negative finite value (given value: {hillshade_z_factor})")

            # Initialize the terrain_attributes dictionary, which will be filled with the requested values.
            terrain_attributes: dict[str, NDArrayf] = {}

            # Check which products should be made to optimize the processing
            make_aspect = any(attr in attribute for attr in ["aspect", "hillshade"])
            make_slope = any(
                attr in attribute
                for attr in
                ["slope", "hillshade", "planform_curvature", "aspect", "profile_curvature", "maximum_curvature"]
            )
            make_hillshade = "hillshade" in attribute
            make_curvature = "curvature" in attribute
            make_planform_curvature = "planform_curvature" in attribute or "maximum_curvature" in attribute
            make_profile_curvature = "profile_curvature" in attribute or "maximum_curvature" in attribute

            if make_slope:
                terrain_attributes["slope"] = _get_terrainattr_richdem(dem, attribute="slope_radians")

            if make_aspect:
                # The aspect of RichDEM is returned in degrees, we convert to radians to match the others
                terrain_attributes["aspect"] = np.deg2rad(_get_terrainattr_richdem(dem, attribute="aspect"))
                # For flat slopes, RichDEM returns a 90° aspect by default, while GDAL return a 180° aspect
                # We stay consistent with GDAL
                slope_tmp = _get_terrainattr_richdem(dem, attribute="slope_radians")
                terrain_attributes["aspect"][slope_tmp == 0] = np.pi

            if make_hillshade:
                # If a different z-factor was given, slopemap with exaggerated gradients.
                if hillshade_z_factor != 1.0:
                    slopemap = np.arctan(np.tan(terrain_attributes["slope"]) * hillshade_z_factor)
                else:
                    slopemap = terrain_attributes["slope"]

                azimuth_rad = np.deg2rad(360 - hillshade_azimuth)
                altitude_rad = np.deg2rad(hillshade_altitude)

                # The operation below yielded the closest hillshade to GDAL (multiplying by 255 did not work)
                # As 0 is generally no data for this uint8, we add 1 and then 0.5 for the rounding to occur between 1 and 255
                terrain_attributes["hillshade"] = np.clip(
                    1.5
                    + 254
                    * (
                            np.sin(altitude_rad) * np.cos(slopemap)
                            + np.cos(altitude_rad) * np.sin(slopemap) * np.sin(
                        azimuth_rad - terrain_attributes["aspect"])
                    ),
                    0,
                    255,
                ).astype("float32")

            if make_curvature:
                terrain_attributes["curvature"] = _get_terrainattr_richdem(dem, attribute="curvature")

            if make_planform_curvature:
                terrain_attributes["planform_curvature"] = _get_terrainattr_richdem(dem, attribute="planform_curvature")

            if make_profile_curvature:
                terrain_attributes["profile_curvature"] = _get_terrainattr_richdem(dem, attribute="profile_curvature")

            # Convert the unit if wanted.
            if degrees:
                for attr in ["slope", "aspect"]:
                    if attr not in terrain_attributes:
                        continue
                    terrain_attributes[attr] = np.rad2deg(terrain_attributes[attr])

            output_attributes = [terrain_attributes[key].reshape(dem.shape) for key in attribute]

            if isinstance(dem, gu.Raster):
                output_attributes = [
                    gu.Raster.from_array(attr, transform=dem.transform, crs=dem.crs, nodata=-99999)
                    for attr in output_attributes
                ]

            return output_attributes if len(output_attributes) > 1 else output_attributes[0]

        # Functions for xdem-implemented methods
        functions_xdem = {
            "slope_Horn": lambda dem: xdem.terrain.slope(dem, resolution=dem.res, degrees=True),
            "aspect_Horn": lambda dem: xdem.terrain.aspect(dem.data, degrees=True),
            "hillshade_Horn": lambda dem: xdem.terrain.hillshade(dem.data, resolution=dem.res),
            "curvature": lambda dem: xdem.terrain.curvature(dem.data, resolution=dem.res),
            "profile_curvature": lambda dem: xdem.terrain.profile_curvature(dem.data, resolution=dem.res),
            "planform_curvature": lambda dem: xdem.terrain.planform_curvature(dem.data, resolution=dem.res),
        }

        # Functions for RichDEM wrapper methods
        functions_richdem = {
            "slope_Horn": lambda dem: get_terrain_attribute_richdem(dem, attribute="slope", degrees=True),
            "aspect_Horn": lambda dem: get_terrain_attribute_richdem(dem, attribute="aspect", degrees=True),
            "hillshade_Horn": lambda dem: get_terrain_attribute_richdem(dem, attribute="hillshade"),
            "curvature": lambda dem: get_terrain_attribute_richdem(dem, attribute="curvature"),
            "profile_curvature": lambda dem: get_terrain_attribute_richdem(dem, attribute="profile_curvature"),
            "planform_curvature": lambda dem: get_terrain_attribute_richdem(dem, attribute="planform_curvature", degrees=True),
        }

        # Copy the DEM to ensure that the inter-test state is unchanged, and because the mask will be modified.
        dem = self.dem.copy()

        # Derive the attribute using both RichDEM and xdem
        attr_xdem = gu.raster.get_array_and_mask(functions_xdem[attribute](dem))[0].squeeze()
        attr_richdem = gu.raster.get_array_and_mask(functions_richdem[attribute](dem))[0].squeeze()

        # We compute the difference and keep only valid values
        diff = attr_xdem - attr_richdem
        diff_valid = diff[np.isfinite(diff)]

        try:
            # Difference between xdem and RichDEM attribute
            # Mean of attribute values to get an order of magnitude of the attribute unit
            magn = np.nanmean(np.abs(attr_xdem))

            # Check that the attributes are similar within a tolerance of a thousandth of the magnitude
            # For instance, slopes have an average magnitude of around 30 deg, so the tolerance is 0.030 deg
            if attribute in ["aspect_Horn"]:
                # For aspect, check the tolerance within a 360 degree modulo due to the circularity of the variable
                diff_valid = np.mod(np.abs(diff_valid), 360)
                assert np.all(np.minimum(diff_valid, np.abs(360 - diff_valid)) < 10 ** (-3) * magn)

            else:
                # All attributes other than aspect are non-circular floats, so we check within a tolerance
                # Here hillshade is not rounded as integer by our calculation, so no need to differentiate as with GDAL
                assert np.all(np.abs(diff_valid < 10 ** (-3) * magn))

        except Exception as exception:

            if PLOT:
                import matplotlib.pyplot as plt

                # Plotting the xdem and RichDEM attributes for comparison (plotting "diff" can also help debug)
                plt.subplot(221)
                plt.imshow(attr_richdem)
                plt.colorbar()
                plt.subplot(222)
                plt.imshow(attr_xdem)
                plt.colorbar()
                plt.subplot(223)
                plt.imshow(diff)
                plt.colorbar()
                plt.show()

            raise exception

        # Introduce some nans
        rng = np.random.default_rng(42)
        dem.data.mask = np.zeros_like(dem.data, dtype=bool)
        dem.data.mask.ravel()[rng.choice(dem.data.size, 50000, replace=False)] = True

        # Validate that this doesn't raise weird warnings after introducing nans and that mask is preserved
        output = functions_richdem[attribute](dem)
        assert np.all(dem.data.mask == output.data.mask)

    def test_hillshade_errors(self) -> None:
        """Validate that the hillshade function raises appropriate errors."""
        # Try giving the hillshade invalid arguments.

        with pytest.raises(ValueError, match="Azimuth must be a value between 0 and 360"):
            xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, azimuth=361)

        with pytest.raises(ValueError, match="Altitude must be a value between 0 and 90"):
            xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, altitude=91)

        with pytest.raises(ValueError, match="z_factor must be a non-negative finite value"):
            xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, z_factor=np.inf)

    def test_hillshade(self) -> None:
        """Test hillshade-specific settings."""

        zfactor_1 = xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, z_factor=1.0)
        zfactor_10 = xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, z_factor=10.0)

        # A higher z-factor should be more variable than a low one.
        assert np.nanstd(zfactor_1) < np.nanstd(zfactor_10)

        low_altitude = xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, altitude=10)
        high_altitude = xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, altitude=80)

        # A low altitude should be darker than a high altitude.
        assert np.nanmean(low_altitude) < np.nanmean(high_altitude)

    @pytest.mark.parametrize(
        "name", ["curvature", "planform_curvature", "profile_curvature", "maximum_curvature"]
    )  # type: ignore
    def test_curvatures(self, name: str) -> None:
        """Test the curvature functions"""

        # Copy the DEM to ensure that the inter-test state is unchanged, and because the mask will be modified.
        dem = self.dem.copy()

        # Derive curvature without any gaps
        curvature = xdem.terrain.get_terrain_attribute(
            dem.data, attribute=name, resolution=dem.res, edge_method="nearest"
        )

        # Validate that the array has the same shape as the input and that all values are finite.
        assert curvature.shape == dem.data.shape
        try:
            assert np.all(np.isfinite(curvature))
        except Exception:
            import matplotlib.pyplot as plt

            plt.imshow(curvature.squeeze())
            plt.show()

        with pytest.raises(ValueError, match="Quadric surface fit requires the same X and Y resolution."):
            xdem.terrain.get_terrain_attribute(dem.data, attribute=name, resolution=(1.0, 2.0))

        # Introduce some nans
        rng = np.random.default_rng(42)
        dem.data.mask = np.zeros_like(dem.data, dtype=bool)
        dem.data.mask.ravel()[rng.choice(dem.data.size, 50000, replace=False)] = True
        # Validate that this doesn't raise weird warnings after introducing nans.
        xdem.terrain.get_terrain_attribute(dem.data, attribute=name, resolution=dem.res)

    def test_get_terrain_attribute(self) -> None:
        """Test the get_terrain_attribute function by itself."""

        # Validate that giving only one terrain attribute only returns that, and not a list of len() == 1
        slope = xdem.terrain.get_terrain_attribute(self.dem.data, "slope", resolution=self.dem.res)
        assert isinstance(slope, np.ndarray)

        # Create three products at the same time
        slope2, _, hillshade = xdem.terrain.get_terrain_attribute(
            self.dem.data, ["slope", "aspect", "hillshade"], resolution=self.dem.res
        )

        # Create a hillshade using its own function
        hillshade2 = xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res)

        # Validate that the "batch-created" hillshades and slopes are the same as the "single-created"
        assert np.array_equal(hillshade, hillshade2, equal_nan=True)
        assert np.array_equal(slope, slope2, equal_nan=True)

        # A slope map with a lower resolution (higher value) should have gentler slopes.
        slope_lowres = xdem.terrain.get_terrain_attribute(self.dem.data, "slope", resolution=self.dem.res[0] * 2)
        assert np.nanmean(slope) > np.nanmean(slope_lowres)

    def test_get_terrain_attribute_errors(self) -> None:
        """Test the get_terrain_attribute function raises appropriate errors."""

        # Below, re.escape() is needed to match expressions that have special characters (e.g., parenthesis, bracket)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Slope method 'DoesNotExist' is not supported. Must be one of: " "['Horn', 'ZevenbergThorne']"
            ),
        ):
            xdem.terrain.slope(self.dem.data, method="DoesNotExist")

        with pytest.raises(
            ValueError,
            match=re.escape("TRI method 'DoesNotExist' is not supported. Must be one of: " "['Riley', 'Wilson']"),
        ):
            xdem.terrain.terrain_ruggedness_index(self.dem.data, method="DoesNotExist")

    def test_raster_argument(self) -> None:

        slope, aspect = xdem.terrain.get_terrain_attribute(self.dem, attribute=["slope", "aspect"])

        assert slope != aspect

        assert type(slope) == type(aspect)
        assert all(isinstance(r, gu.Raster) for r in (aspect, slope, self.dem))

        assert slope.transform == self.dem.transform == aspect.transform
        assert slope.crs == self.dem.crs == aspect.crs

    def test_rugosity_jenness(self) -> None:
        """
        Test the rugosity with the same example as in Jenness (2004),
        https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2.
        """

        # Derive rugosity from the function
        dem = np.array([[190, 170, 155], [183, 165, 145], [175, 160, 122]], dtype="float32")

        # Derive rugosity
        rugosity = xdem.terrain.rugosity(dem, resolution=100.0)

        # Rugosity of Jenness (2004) example
        r = 10280.48 / 10000.0

        assert rugosity[1, 1] == pytest.approx(r, rel=10 ** (-4))

    # Loop for various elevation differences with the center
    @pytest.mark.parametrize("dh", np.linspace(0.01, 100, 10))  # type: ignore
    # Loop for different resolutions
    @pytest.mark.parametrize("resolution", np.linspace(0.01, 100, 10))  # type: ignore
    def test_rugosity_simple_cases(self, dh: float, resolution: float) -> None:
        """Test the rugosity calculation for simple cases."""

        # We here check the value for a fully symmetric case: the rugosity calculation can be simplified because all
        # eight triangles have the same surface area, see Jenness (2004).

        # Derive rugosity from the function
        dem = np.array([[1, 1, 1], [1, 1 + dh, 1], [1, 1, 1]], dtype="float32")

        rugosity = xdem.terrain.rugosity(dem, resolution=resolution)

        # Half surface length between the center and a corner cell (in 3D: accounting for elevation changes)
        side1 = np.sqrt(2 * resolution**2 + dh**2) / 2.0
        # Half surface length between the center and a side cell (in 3D: accounting for elevation changes)
        side2 = np.sqrt(resolution**2 + dh**2) / 2.0
        # Half surface length between the corner and side cell (no elevation changes on this side)
        side3 = resolution / 2.0

        # Formula for area A of one triangle
        s = (side1 + side2 + side3) / 2.0
        A = np.sqrt(s * (s - side1) * (s - side2) * (s - side3))

        # We sum the area of the eight triangles, and divide by the planimetric area (resolution squared)
        r = 8 * A / (resolution**2)

        # Check rugosity value is valid
        assert r == pytest.approx(rugosity[1, 1], rel=10 ** (-6))

    def test_get_quadric_coefficients(self) -> None:
        """Test the outputs and exceptions of the get_quadric_coefficients() function."""

        dem = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype="float32")

        coefficients = xdem.terrain.get_quadric_coefficients(
            dem, resolution=1.0, edge_method="nearest", make_rugosity=True
        )

        # Check all coefficients are finite with an edge method
        assert np.all(np.isfinite(coefficients))

        # The 4th to last coefficient is the dem itself (could maybe be removed in the future as it is duplication..)
        assert np.array_equal(coefficients[-4, :, :], dem)

        # The middle pixel (index 1, 1) should be concave in the x-direction
        assert coefficients[3, 1, 1] < 0

        # The middle pixel (index 1, 1) should be concave in the y-direction
        assert coefficients[4, 1, 1] < 0

        with pytest.raises(ValueError, match="Invalid input array shape"):
            xdem.terrain.get_quadric_coefficients(dem.reshape((1, 1, -1)), 1.0)

        # Validate that when using the edge_method="none", only the one non-edge value is kept.
        coefs = xdem.terrain.get_quadric_coefficients(dem, resolution=1.0, edge_method="none")
        assert np.count_nonzero(np.isfinite(coefs[0, :, :])) == 1
        # When using edge wrapping, all coefficients should be finite.
        coefs = xdem.terrain.get_quadric_coefficients(dem, resolution=1.0, edge_method="wrap")
        assert np.count_nonzero(np.isfinite(coefs[0, :, :])) == 9
