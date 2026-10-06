import importlib.util
from pathlib import Path
from types import SimpleNamespace

script_path = (
    Path(__file__).resolve().parents[1] / "scripts" / "validation_extraction.py"
)
script_spec = importlib.util.spec_from_file_location(
    "validation_extraction", script_path
)
if script_spec is None or script_spec.loader is None:
    raise ImportError(f"Unable to load {script_path}")
validation_extraction = importlib.util.module_from_spec(script_spec)
script_spec.loader.exec_module(validation_extraction)
get_file_path = validation_extraction.get_file_path


def test_tcpc_utm_tile_path_uses_tiles_root():
    path = get_file_path(
        "TCPC-10",
        "v021",
        "/vsis3/lcfm_waw3-1_4b82fdbbe2580bdfc4f595824922507c0d7cae2541c0799982/gaf/products/TCPC-10/v021/tiles_utm/",
        year=2023,
        is_tiles=True,
    )(SimpleNamespace(tile="57PZN", block_id=0))

    assert path == (
        "/vsis3/lcfm_waw3-1_4b82fdbbe2580bdfc4f595824922507c0d7cae2541c0799982/gaf/products/TCPC-10/v021/tiles_utm/"
        "57/P/ZN/2023/LCFM_TCPC-10_V021_2023_57PZN_CHANGE.tif"
    )


def test_tcpc_block_path_uses_change_filename():
    path = get_file_path(
        "TCPC-10",
        "v021",
        "/vsis3/bucket/gaf/products/TCPC-10/v021/blocks",
        year=2023,
        is_tiles=False,
    )(SimpleNamespace(tile="57PZN", block_id=0))

    assert path.endswith(
        "/57/P/ZN/2023/TCPC-10/LCFM_TCPC-10_V021_2023_57PZN_CHANGE.tif"
    )
