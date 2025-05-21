# %% [markdown]
# # Reading Map through STAC

# %%
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import boto3
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rioxarray  # noqa: F401
import stackstac
import xarray
from dotenv import load_dotenv
from loguru import logger
from pystac import Item
from pystac_client import Client
from rasterio.enums import ColorInterp, Resampling
from rasterio.session import AWSSession
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, shape
from tqdm import tqdm

RGBA_TABLE = {
    10: [0, 100, 0, 255],
    20: [255, 187, 34, 255],
    30: [255, 255, 76, 255],
    40: [240, 150, 255, 255],
    50: [250, 0, 0, 255],
    60: [180, 180, 180, 255],
    70: [240, 240, 240, 255],
    80: [0, 100, 200, 255],
    90: [0, 150, 160, 255],
    100: [250, 230, 160, 255],
    110: [0, 207, 117, 255],
    254: [0, 0, 0, 255],
    255: [0, 0, 0, 0],
}


def get_s3_session(profile="lcfm"):
    # Authentication
    try:
        b3 = boto3.Session(profile_name="test")
    except Exception:
        # Read s3 keys from .env file
        load_dotenv()
        b3 = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    if profile == "gaf":
        endpoint_url = "lcfm-datahub.gaf.de"
    else:
        endpoint_url = "s3.waw3-1.cloudferro.com"
    # Create a session with the specified endpoint URL
    aws_session = AWSSession(session=b3, endpoint_url=endpoint_url)
    # Create gdal env based on this
    gdal_env = stackstac.DEFAULT_GDAL_ENV.updated(
        always={
            "session": aws_session,
        }
    )
    return gdal_env


def create_gdf_utm(gdf_pm: gpd.GeoDataFrame, crs_utm: int) -> gpd.GeoDataFrame:
    """
    Create a new GeoDataFrame from the input one and reproject it to the specified UTM CRS.
    """
    # Get bounds
    bounds = gdf_pm.to_crs(crs_utm).total_bounds.round()
    logger.debug(f"Bounds: {bounds}")

    # Create a new GeoDataFrame with the specified UTM CRS
    polygon = Polygon.from_bounds(*bounds)
    gdf = gpd.GeoDataFrame(index=[0], crs=crs_utm, geometry=[polygon])

    # Check span
    span_x = bounds[2] - bounds[0]
    span_y = bounds[3] - bounds[1]
    logger.debug(f"Size [m]: {span_x}, {span_y}")
    if span_x != 100 or span_y != 100:
        raise ValueError(
            f"Size of the bounding box is not 100m x 100m, but {span_x}m x {span_y}m"
        )
    return gdf


def get_stac_items(
    geometry_latlon: Polygon, product: str, version: str, year: int
) -> List[Item]:
    # Collection parameters
    collection = f"LCFM_{product}_{version}"
    # Define the date range for the search
    start_date = datetime(year, 1, 1).isoformat() + "Z"
    end_date = datetime(year, 12, 12).isoformat() + "Z"

    # STAC query
    # Connect to the STAC API
    stac_api_url = "https://www.stac.lcfm.dataspace.copernicus.eu/"
    catalog = Client.open(stac_api_url)
    # Fetch items
    search = catalog.search(
        collections=[collection],
        datetime=f"{start_date}/{end_date}",
        intersects=geometry_latlon,
    )
    items = list(search.items())

    # Check if items are found
    if items:
        logger.debug(f"Found {len(items)} items in the collection:")
        for item in items:
            logger.debug(f"- {item.id}")
    else:
        logger.debug("No items found in the collection.")
        raise ValueError(f"No items found in the collection {collection}.")
    return items


def get_map_asset(
    gdf: gpd.GeoDataFrame,
    product: str,
    resolution: int,
    nodata: int,
    version: str,
    year: int,
    profile: str = "lcfm",
) -> xarray.DataArray:
    crs_utm = int(str(gdf.crs).split(":")[-1])
    geometry_latlon = gdf.to_crs("EPSG:4326").geometry.iloc[0]
    bounds_utm = gdf.iloc[0].geometry.bounds
    span_x = bounds_utm[2] - bounds_utm[0]
    span_y = bounds_utm[3] - bounds_utm[1]

    # Get STAC items
    items = get_stac_items(geometry_latlon, product, version, year)

    # Select item from the list with matching crs
    item = None
    for i in items:
        if i.properties["proj:epsg"] == crs_utm:
            item = i
            break
    if item:
        logger.debug(f"Selected {item.id} with matching CRS.")
        source_crs = crs_utm
        source_bounds = bounds_utm
    else:
        # TODO: refine the selection of the item
        item = min(
            items,
            key=lambda x: shape(x.geometry).centroid.distance(geometry_latlon.centroid),
        )
        source_crs = item.properties["proj:epsg"]
        # Reproject
        source_bounds = gdf.to_crs(source_crs).iloc[0].geometry.bounds
        raise NotImplementedError(
            f"No items match the requested CRS {crs_utm}. Reprojection implemented, but not checked."
        )

    # Read MAP asset
    assets = ["map"]
    if profile == "lcfm":
        for asset_key in assets:
            item.assets[asset_key].href = item.assets[asset_key].extra_fields[
                "alternate"
            ]["local"]["href"]
    # Now you can use stackstac with the modified STAC item
    gdal_env = get_s3_session(profile=profile)
    map = stackstac.stack(
        [item],
        assets=assets,
        bounds=source_bounds,
        resolution=resolution,
        epsg=source_crs,
        fill_value=np.uint8(nodata),
        dtype=np.uint8,
        rescale=False,
        gdal_env=gdal_env,
    ).isel(time=0)
    # Fix the coordinates
    map.coords["x"] = map.coords["x"] + map.transform[0] / 2
    map.coords["y"] = map.coords["y"] + map.transform[4] / 2
    # Example to check the DataArray information
    # map

    # Reproject if needed
    if source_crs != crs_utm:
        transform = from_bounds(
            *bounds_utm, (span_x / resolution), (span_y / resolution)
        )
        map = map.rio.reproject(
            f"EPSG:{crs_utm}",
            resampling=Resampling.nearest,
            transform=transform,
        )
    # Extract the bands
    map_asset = map.sel(band="map")
    return map_asset


def create_plot(asset: xarray.DataArray):
    # sort by key and build hex list & class values
    class_values, hex_colors = zip(
        *[
            (k, mcolors.to_hex(np.array(v) / 255.0))
            for k, v in sorted(RGBA_TABLE.items())
            if k not in (254, 255)  # skip background/no‐data if you like
        ]
    )
    cmap = mcolors.ListedColormap(hex_colors, name="lc")
    # Build boundaries (one extra above highest)
    class_bounds = list(class_values) + [
        class_values[-1] + (class_values[-1] - class_values[-2])
    ]
    norm = mcolors.BoundaryNorm(class_bounds, cmap.N)

    # Plot the map
    plt.figure(figsize=(10, 10))
    plt.imshow(asset, cmap=cmap, norm=norm)
    plt.title("Land‐cover classification")
    plt.axis("off")
    plt.show()


def run_gdalinfo(output_file, bounds, resolution, results_dir, filename):
    result = subprocess.run(
        ["gdalinfo", output_file, "-json"], capture_output=True, text=True
    )
    # Parse JSON output into a Python dictionary
    if result.returncode == 0:
        gdalinfo = json.loads(result.stdout)
        # Extract key information
        transform = gdalinfo.get("geoTransform", [])
        if transform:
            logger.debug(f"Pixel size (X): {transform[1]} meters")
            logger.debug(f"Pixel size (Y): {abs(transform[5])} meters")
            logger.debug(f"Upper left X: {transform[0]}")
            logger.debug(f"Upper left Y: {transform[3]}")
        # Write to results/gdalinfo directory
        gdalinfo_dir = os.path.join(results_dir, "gdalinfo")
        os.makedirs(gdalinfo_dir, exist_ok=True)
        gdalinfo_file = os.path.join(gdalinfo_dir, f"{filename}.json")
        with open(gdalinfo_file, "w") as f:
            json.dump(gdalinfo, f, indent=4)
    else:
        raise RuntimeError(f"gdalinfo command failed with error: {result.stderr}")
    # Perform checks
    # Check the pixel size
    if transform[1] != resolution or abs(transform[5]) != resolution:
        raise ValueError(
            f"Pixel size is not {resolution}m, but {transform[1]}m x {abs(transform[5])}m"
        )
    # Check the bounds
    if transform[0] != bounds[0] or transform[3] != bounds[3]:
        raise ValueError(
            f"Upper left corner is not {bounds[0]}, {bounds[3]}, but {transform[0]}, {transform[3]}"
        )


def extract_location(
    gdf_pm: gpd.GeoDataFrame,
    product="LCM-10",
    version="v008",
    year=2020,
    results_dir=Path("./results"),
    profile="lcfm",
):
    loc_id = gdf_pm["id_loc"].iloc[0]

    # Create a new gdf from the input one
    crs_utm = int(gdf_pm["UTM"].iloc[0])
    gdf = create_gdf_utm(gdf_pm, crs_utm)
    bounds_utm = gdf.iloc[0].geometry.bounds

    # Fixed, though these could also be inferred from the STAC collection
    resolution = 10
    nodata = 255

    # Get map asset
    map_asset = get_map_asset(
        gdf, product, resolution, nodata, version, year, profile=profile
    )

    create_plot(map_asset)

    # Write
    filename = f"LCFM_{product}_{version.upper()}_{year}_{loc_id}_MAP"
    output_file = f"{results_dir}/{filename}.tif"
    with rasterio.open(
        output_file,
        "w",
        driver="GTiff",
        height=map_asset.rio.height,
        width=map_asset.rio.width,
        count=1,
        dtype=map_asset.dtype,
        crs=map_asset.rio.crs,
        transform=map_asset.rio.transform(),
        compress="LZW",
        nodata=nodata,
    ) as dst:
        dst.write(map_asset.values, 1)
        dst.set_band_description(1, "MAP")
        dst.colorinterp = [ColorInterp.palette]
        dst.write_colormap(1, RGBA_TABLE)

    # Check: run gdalinfo command
    run_gdalinfo(output_file, bounds_utm, resolution, results_dir, filename)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Extract Map data using STAC API.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--product", type=str, default="LCM-10", help="Product name (default: LCM-10)"
    )
    parser.add_argument(
        "--version", type=str, default="v008", help="Product version (default: v008)"
    )
    parser.add_argument(
        "--year", type=int, default=2020, help="Year to process (default: 2020)"
    )
    parser.add_argument(
        "--shapefile",
        type=str,
        default="100perc_sample_10m_epsg3857_idloc_selection.shp",
        help="Shapefile name (default: 100perc_sample_10m_epsg3857_idloc_selection.shp",
    )
    parser.add_argument(
        "--loc-ids",
        "--loc_ids",
        type=str,
        default=None,
        help="Comma-separated list of location IDs to process (default: all)",
    )
    args = parser.parse_args()

    # Set logging level based on debug flag
    logger.remove()  # Remove default handler
    if args.debug:
        logger.add(sys.stderr, level="DEBUG")
        logger.debug("Debug mode enabled")
    else:
        logger.add(sys.stderr, level="INFO")

    # Dynamically check the current working directory
    cwd = os.getcwd()
    base_dir = os.path.dirname(cwd) if os.path.basename(cwd) == "scripts" else cwd
    shapefile_path = os.path.join(base_dir, "resources", args.shapefile)
    results_dir = os.path.join(base_dir, "results")

    # Load the shapefile
    logger.info(f"Loading shapefile: {shapefile_path}")
    gdf_all = gpd.read_file(shapefile_path)

    # Get location IDs to process
    if args.loc_ids:
        # If specific location IDs provided, split and convert to integers
        loc_ids = [int(loc_id.strip()) for loc_id in args.loc_ids.split(",")]
        # Verify that all requested IDs exist in the shapefile
        all_loc_ids = gdf_all["id_loc"].unique().tolist()
        missing_ids = [loc_id for loc_id in loc_ids if loc_id not in all_loc_ids]
        if missing_ids:
            logger.warning(
                f"The following location IDs do not exist in the shapefile: {missing_ids}"
            )
            loc_ids = [loc_id for loc_id in loc_ids if loc_id in all_loc_ids]
    else:
        # Otherwise, process all location IDs
        loc_ids = gdf_all["id_loc"].unique().tolist()

    logger.info(
        f"Processing {len(loc_ids)} locations with product={args.product}, version={args.version}, year={args.year}"
    )

    # Process each location, keeping track of failed locations
    failed_locations = []
    for loc_id in tqdm(loc_ids, desc="Processing locations"):
        logger.debug(f"Processing location {loc_id}")
        gdf_pm = gdf_all[gdf_all["id_loc"] == loc_id].copy()
        try:
            extract_location(
                gdf_pm,
                product=args.product,
                version=args.version,
                year=args.year,
                results_dir=results_dir,
            )
        except Exception as e:
            failed_locations.append((loc_id, str(e)))

    # Print summary of failed locations
    if failed_locations:
        logger.error(f"{len(failed_locations)} locations failed to process:")
        for loc_id, error in failed_locations:
            logger.error(f"Location {loc_id}: {error}")
    else:
        logger.info("All locations processed successfully!")


if __name__ == "__main__":
    main()
