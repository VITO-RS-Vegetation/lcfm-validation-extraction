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
- Shapefile is hardcoded
- Print statements
- Reprojection case not checked --> TODO: raise an error instead
- Needs a STAC collection, which preferably contains UTM tiles (cfr. above)
- Limited to CloudFerro's S3; GAF's does not work --> can't ping lcfm-datahub.gaf.de, whereas this is possible for s3.waw3-1.cloudferro.com, so likely a firewall issue at GAF's side