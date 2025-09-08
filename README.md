# LCFM Validation Extracion

Code for the extraction of validation data for the LCM-10 and TCD-10 products of the LCFM project.

## Set-up
To start with, Gdal is assumed to be installed.

To set up the environment for running the notebooks/scripts, follow the instructions below.

Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Sync the virtual environment.

```
uv sync --all-extras
```

Next, activate the environment:

```
source .venv/bin/activate
```

Last, make sure the application can access the designated S3 storage. Either create a profile "lcfm" with the aws cli or create a .env file in this repository with the following contents:
```
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=yyy
```

## Use

### LCM-10
Navigate to the scripts folder and run:
```
python lcm10_stac.py
```

Alternatively, run this locally:
```
python scripts/validation_extraction.py -l /vitodata/vegteam_vol2/products/LCFM/ -b ./resources/ -o ./results resources/100perc_sample_10m_epsg3857_idloc_selection.shp
```

Or with remote data access:
```
python scripts/validation_extraction.py -l /vsis3/lcfm_waw3-1_4b82fdbbe2580bdfc4f595824922507c0d7cae2541c0799982/vito/validation -b ./resources/ -o ./results resources/100perc_sample_10m_epsg3857_idloc_selection.shp
```

For the PROB10, add:
```
--layer PROB10
```

### TCD-10
```
python scripts/validation_extraction.py -t /vsis3/lcfm_waw3-1_4b82fdbbe2580bdfc4f595824922507c0d7cae2541c0799982/gaf/test/TCD-10-raw-masked/LCFM/TCD-10/v100/blocks -b ./resources/ --blocks-grid-file blocks_tropics_v12.fgb -v v100 -o ./results resources/100perc_sample_10m_epsg3857_idloc_selection.shp
```

### STAC-based workflow

#### Limitations:
- Reprojection case not checked --> this will raise an error instead
- Limited to CloudFerro's S3; GAF's does not work --> can't ping lcfm-datahub.gaf.de, whereas this is possible for s3.waw3-1.cloudferro.com, so likely a firewall issue at GAF's side

#### Prerequisites:
Extractions require some actions from producer's side to work:
1. Tile to UTM tiles
    - VITO: Run [tiler](https://github.com/VITO-RS-Vegetation/veg-workflows/blob/main/workflows/products/LCFM/LCM-10/tiling.py) with the --tiling utm argument
    - Note: these are full S2 tiles. It's possible to support "reduced" S2 tiles when reprojecting client-side (see "Limitations" above).
2. Upload to S3
    - Command: ```aws s3 sync --profile profile_name source target```
    - Note: exact path does not matter as long as it can be accessed and the STAC items refer to the correct path
3. Create a STAC collection for UTM tiles
    - VITO: run [STAC builder](https://github.com/VitoTAP/stac-catalog-builder/tree/lcfm/configs-datasets/lcfm/LCM-10_utm)
4. Distribute S3 keys to users