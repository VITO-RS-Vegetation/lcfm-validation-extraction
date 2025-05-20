# LCFM Validation Extracion

Code for the extraction of validation data for the LCM-10 and TCD-10 products of the LCFM project.

## Set-up
To start with, Gdal is assumed to be installed.

To set up the environment for running the notebooks/scripts, follow the instructions below.

Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Synv the virtual environment.

```
uv sync --all-extras
```

Next, activate the environment:

```
source .venv/bin/activate
```

## Use
Navigate to the notebooks folder and run:
```
python lcm10_stac.py
```

### Limitations:
- Reprojection case not checked --> this will raise an error instead
- Needs a STAC collection, which preferably contains UTM tiles (cfr. above)
- Limited to CloudFerro's S3; GAF's does not work --> can't ping lcfm-datahub.gaf.de, whereas this is possible for s3.waw3-1.cloudferro.com, so likely a firewall issue at GAF's side

### Prerequisites:
Extractions require some actions from producer's side to work:
1. Tile to UTM tiles
    - VITO: Run [tiler](https://github.com/VITO-RS-Vegetation/veg-workflows/blob/main/workflows/products/LCFM/LCM-10/tiling.py) with the --tiling utm argument
2. Upload to S3 (exact path does not matter as long as it can be accessed and the STAC items refer to the correct path)
3. Create a STAC collection for UTM tiles
    - VITO: run [STAC builder](https://github.com/VitoTAP/stac-catalog-builder/tree/lcfm/configs-datasets/lcfm/LCM-10_utm)
4. Distribute keys to S3 to users