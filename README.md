# Description

Template repository

# LCFM and Sen4LDN Production

Code and utilities for the production of products using OpenEO for the LCFM and Sen4LDN projects.

## Set-up
Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

You need access to the VITO artifactory (check with sops) and UV credentials configured with your Terrascope user and password in order to install our packages on the private index. This is needed for the dependencies.
Add these to the `~/.user_aliases` file (or add `source ~/Private/path/to/uv.env` where `uv.env` contains the variables below.)

```
export UV_PUBLISH_USERNAME=user
export UV_PUBLISH_PASSWORD=pass
export UV_INDEX_VITO_ARTIFACTORY_USERNAME=user
export UV_INDEX_VITO_ARTIFACTORY_PASSWORD=pass
```

Synv the virtual environment.

```
uv sync --all-extras
```

Next, activate the environment:

```
source .venv/bin/activate
```

Lastly, configure the pre-commit hook to ensure your code meets our quality standards:

```
uv run pre-commit install
```
This will install the hook defined in the `.pre-commit-config.yaml` file in the _.git/hooks/pre-commit_ file.

## Pytorch dependencies
Pytorch dependencies on default come with GPU cuda support, with a big overhead on package size.
If you only need the GPU support (you are not working on HPC), then have a look in the relevant part on documentation:
[Confluence Documentation - Install CPU-only dependencies for torch](https://confluence.vito.be/display/GEOM/Documentation+%7C+UV+Python+package+manager#Documentation|UVPythonpackagemanager-InstallCPUonlydependenciesfortorch)

## GDAL dependencies
To keep minimal this template, I haven't include GDAL two step installation in the template.
You can find that in the gdal specific template: [gdal-uv-template](https://github.com/VITO-RS-Vegetation/gdal-uv)

## Define your prefered python version
The simpliest way to define your python version is to create a `.python-version` file and add the python version e.g. 3.11.
The version you are requiring should be included in the `pyproject.toml` `requires-python` field.

## Strip out notebooks output
Apply the following git filter to avoid commiting notebook output:
```bash
git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'  
```
If you already tracking some notebooks you will also need to renormalize git tracking
```bash
git add --renormalize .
```

## CI/CD
The current template repo contains a basic github action under .github/actions/setup-env/action.yml which is meant to be used from all the vegetation github workflows

## Support
Are you member of LCF team and facing an issue? Reach the vegdevs in our [teams channel](https://teams.microsoft.com/l/channel/19%3A03615b1a60c94e59afd30847b20566d9%40thread.tacv2/Tech%20Stack?groupId=bcba1a22-7d87-4bfe-8baf-d42830c644c3&tenantId=9e2777ed-8237-4ab9-9278-2c144d6f6da3)

## Version 2 of action
just to unblock git push
