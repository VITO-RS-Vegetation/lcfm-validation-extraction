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
Navigate to the scripts folder and run:
```
python lcm10_stac.py
```

### Limitations:
- Reprojection case not checked --> this will raise an error instead
- Limited to CloudFerro's S3; GAF's does not work --> can't ping lcfm-datahub.gaf.de, whereas this is possible for s3.waw3-1.cloudferro.com, so likely a firewall issue at GAF's side

### Prerequisites:
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