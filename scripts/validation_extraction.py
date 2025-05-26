from pathlib import Path

import geopandas as gpd
import rasterio
from loguru import logger
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from tqdm.auto import tqdm


def get_block_path(product, version, product_path):
    def wrapped_get_block_path(row):
        tile = row.tile
        block_id = row.block_id
        if product == "LCM-10":
            # LCFM/LCM-10/v008-m10-c84/blocks/16/S/GA/2020/MAP/LCFM_LCM-10_V008-M10-C84_2020_16SGA_100_MAP.tif
            return f"{product_path}/LCFM/LCM-10/{version}/blocks/{tile[:2]}/{tile[2]}/{tile[-2:]}/2020/MAP/LCFM_LCM-10_{version.upper()}_2020_{tile}_{block_id:03d}_MAP.tif"
        elif product == "TCD-10":
            # data/lcfm/TCD-10-raw/data_v2/LSF-ANNUAL_v100/TCD_v01-alpha02-harm/blocks/51/R/TP/2020/TCD-10/LCFM_LSF-ANNUAL_V100_2020_51RTP_026_TCD-10_masked.ti
            return f"{product_path}/{tile[:2]}/{tile[2]}/{tile[-2:]}/2020/TCD-10/LCFM_LSF-ANNUAL_{version.upper()}_2020_{tile}_{block_id:03d}_TCD-10_masked.tif"
        else:
            raise NotImplementedError

    return wrapped_get_block_path


def merge_datasets(input_paths, output_path):
    src_files_to_mosaic = []

    for path in input_paths:
        src = rasterio.open(path)
        src_files_to_mosaic.append(src)

    mosaic, out_transform = merge(src_files_to_mosaic)

    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
        }
    )

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files_to_mosaic:
        src.close()

    return output_path, out_meta


def warp_dataset(input_path, output_path, target_epsg):
    with rasterio.open(input_path) as src:
        vrt_options = {
            "crs": f"EPSG:{target_epsg}",
            "resampling": Resampling.nearest,
        }
        with WarpedVRT(src, **vrt_options) as vrt:
            data = vrt.read()
            out_meta = vrt.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": vrt.height,
                    "width": vrt.width,
                    "transform": vrt.transform,
                    "crs": f"EPSG:{target_epsg}",
                }
            )

            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(data)

    return output_path, out_meta


def extract_patch(input_path, output_path, target_bounds):
    with rasterio.open(input_path) as src:
        window = src.window(*target_bounds)
        data = src.read(window=window)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": window.height,
                "width": window.width,
                "transform": src.window_transform(window),
                "crs": src.crs,
            }
        )
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(data)


def load_blocks(loc_geom, loc_epsg, blocks_grid_path):
    # Load blocks
    blocks = gpd.read_file(blocks_grid_path, mask=loc_geom)

    # check if any block contains the location
    blocks_cont = blocks[blocks.contains(loc_geom)]
    if blocks_cont.empty:  # if not, return all blocks that intersect
        return blocks

    # check if any block is in the same epsg as the location
    blocks_cont_epsg = blocks_cont[blocks_cont.epsg == loc_epsg]
    if blocks_cont_epsg.empty:
        return blocks_cont
    else:
        return blocks_cont_epsg.iloc[[0]]


def process_loc(
    gdf, id_loc, blocks_grid_path, product, version, product_path, output_path
):
    loc_gdf = gdf[gdf["id_loc"] == id_loc]
    loc_geom = loc_gdf.union_all()
    target_epsg = gdf[gdf["id_loc"] == id_loc].iloc[0].UTM
    target_bounds = loc_gdf.to_crs(target_epsg).total_bounds.round()

    blocks = load_blocks(loc_geom, target_epsg, blocks_grid_path)

    if blocks.empty:
        logger.warning(f"No blocks found for location {id_loc}")
        return

    blocks["path"] = blocks.apply(
        get_block_path(product, version, product_path), axis=1
    )
    blocks_epsgs = blocks.epsg.unique()
    # Check if all blocks are in the same EPSG

    out_fn = output_path / f"{id_loc}.tif"

    tmp_folder = output_path / f"tmp_{id_loc}"
    tmp_folder.mkdir(exist_ok=True, parents=True)

    if len(blocks) == 1:
        # only 1 block
        if target_epsg == blocks.iloc[0].epsg:
            # same epsg, just read the target window
            logger.debug(
                f"1 intersecting block - same EPSG: {target_epsg} - extracting patch"
            )
            extract_patch(blocks.iloc[0].path, out_fn, target_bounds)
        else:
            # different epsg, merge the blocks
            logger.debug(
                f"1 intersecting block - different EPSG: {target_epsg} - warping and extracting patch"
            )
            tmp_path = tmp_folder / f"{id_loc}_warped_tmp.tif"
            warp_dataset(blocks.iloc[0].path, tmp_path, target_epsg)
            extract_patch(tmp_path, out_fn, target_bounds)
    else:
        # multiple blocks
        # check for multiple input epsgs
        if len(blocks_epsgs) == 1:
            # only 1 epsg
            if target_epsg == blocks_epsgs[0]:
                # same epsg, just merge the blocks
                logger.debug(
                    f"{len(blocks)} intersecting blocks - same target EPSG: {target_epsg} - merging"
                )

                tmp_path = tmp_folder / f"{id_loc}_merged_tmp.tif"
                merge_datasets(blocks.path.tolist(), tmp_path)
                extract_patch(tmp_path, out_fn, target_bounds)

            else:
                # only 1 epsg but different from target
                logger.debug(
                    f"{len(blocks)} intersecting blocks - different target EPSG: {target_epsg} - merging and warping"
                )
                tmp_path = tmp_folder / f"{id_loc}_merged_tmp.tif"
                merge_datasets(blocks.path.tolist(), tmp_path)

                tmp_path2 = tmp_folder / f"{id_loc}_merged_warped_tmp.tif"
                warp_dataset(tmp_path, tmp_path2, target_epsg)
                extract_patch(tmp_path2, out_fn, target_bounds)

        else:
            # multiple epsgs & multiple blocks. for each epsg, merge the blocks and warp to target

            logger.debug(
                f"{len(blocks)} intersecting blocks - {len(blocks_epsgs)} source EPSGS - merging and warping"
            )

            # this works but it's giving a strange resampling, it does not
            # seem to match well the original data. as this is probably not happening
            # in the real dataset, we raise and ask IGNFI to provide us the location shapefile
            # to test it on a real case.
            raise NotImplementedError(
                f"Multiple EPSGs detected. Please provide a shapefile for id_loc = {id_loc} test this case."
            )

            # merged_epsgs_paths = []
            # for epsg in blocks_epsgs:

            #     tmp_path = tmp_folder / f"{id_loc}_merged_{epsg}_tmp.tif"
            #     merge_datasets(blocks.path.tolist(), tmp_path)
            #     if epsg != target_epsg:
            #         tmp_path2 = tmp_folder / f"{id_loc}_merged_warped_{epsg}_tmp.tif"
            #         warp_dataset(tmp_path, tmp_path2, target_epsg)
            #     else:
            #         tmp_path2 = tmp_path
            #     merged_epsgs_paths.append(tmp_path2)
            # # merge all the merged epsgs
            # tmp_path2 = tmp_folder / f"{id_loc}_merged_all_epsgs_warped_tmp.tif"
            # merge_datasets(merged_epsgs_paths, tmp_path2)
            # extract_patch(tmp_path2, out_fn, target_bounds)

    # clean up
    for path in tmp_folder.glob("*"):
        path.unlink()
    tmp_folder.rmdir()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract patches from LCFM blocks for a given location"
    )
    product_group = parser.add_mutually_exclusive_group(required=True)
    product_group.add_argument(
        "-l",
        "--lcm10-path",
        type=str,
        help="LCFM path to use",
    )
    product_group.add_argument(
        "-t",
        "--tcd10-path",
        type=str,
        help="TCD path to use",
    )

    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="v008-m10-c84",
        help="LCM / TCD version to use",
    )

    parser.add_argument(
        "-b",
        "--blocks-grid-path",
        type=str,
        default="/vitodata/vegteam/auxdata/grid/blocks_global/blocks_global_v12.fgb",
        help="Path to the blocks grid shapefile",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default=".",
        help="Path to the output folder",
    )
    parser.add_argument(
        "input_shapefile",
        type=str,
        help="Path to the input shapefile with locations (id_loc, UTM, geometry) columns",
    )

    args = parser.parse_args()
    version = args.version
    if args.lcm10_path:
        product = "LCM-10"
        product_path = args.lcm10_path
    elif args.tcd10_path:
        product = "TCD-10"
        product_path = args.tcd10_path
    else:
        # Should not tigger as the mutually exclusive group is `required=True`, just to be sure
        raise NotImplementedError("Please provide one of --lcm10_path or --tcd10_path")

    blocks_grid_path = args.blocks_grid_path
    input_shapefile = args.input_shapefile
    output_path = Path(args.output_path)

    logger.info(f"Loading input shapefile {input_shapefile}")
    if input_shapefile.endswith(".zip"):
        input_shapefile = f"zip://{input_shapefile}"

    gdf = gpd.read_file(input_shapefile)
    gdf = gdf.to_crs("EPSG:4326")

    for id_loc in tqdm(gdf.id_loc.unique(), desc="Processing locations"):
        logger.debug(f"Processing location {id_loc}")
        process_loc(
            gdf,
            id_loc,
            blocks_grid_path,
            product,
            version,
            product_path,
            output_path,
        )
