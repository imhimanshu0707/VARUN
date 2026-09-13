from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio


def read_raw_band(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with rasterio.open(path) as src:
        if src.count not in (1, 2):
            raise ValueError(
                f"{path.name}: expected 1 data band and optional mask band"
            )

        data = src.read(1).astype(np.float32)
        valid = np.isfinite(data) & (data > 0)

        if src.count == 2:
            data_mask = src.read(2)
            mask_values = np.unique(data_mask)

            if not np.all(np.isin(mask_values, [0, 1])):
                raise ValueError(
                    f"{path.name}: band 2 is not a binary data mask"
                )

            valid &= data_mask > 0

        metadata = {
            "profile": src.profile.copy(),
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
        }

    return data, valid, metadata


def to_filled_db(
    linear: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, float]:
    if not valid.any():
        raise ValueError("Raster contains no valid positive pixels")

    result = np.full(linear.shape, np.nan, dtype=np.float32)
    result[valid] = (
        10.0 * np.log10(linear[valid])
    ).astype(np.float32)

    fill_db = float(np.percentile(result[valid], 2))
    result[~valid] = fill_db

    return result, fill_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vh", required=True)
    parser.add_argument("--vv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--acquisition-time", required=True)
    args = parser.parse_args()

    vh_path = Path(args.vh).resolve()
    vv_path = Path(args.vv).resolve()
    output_path = Path(args.output).resolve()

    vh_linear, vh_valid, vh_meta = read_raw_band(vh_path)
    vv_linear, vv_valid, vv_meta = read_raw_band(vv_path)

    if (
        vh_meta["crs"] != vv_meta["crs"]
        or vh_meta["transform"] != vv_meta["transform"]
        or vh_meta["width"] != vv_meta["width"]
        or vh_meta["height"] != vv_meta["height"]
    ):
        raise ValueError("VH and VV rasters are not spatially aligned")

    if vh_meta["crs"] is None:
        raise ValueError("Input rasters do not have a CRS")

    common_valid = vh_valid & vv_valid

    vh_db, vh_fill = to_filled_db(vh_linear, common_valid)
    vv_db, vv_fill = to_filled_db(vv_linear, common_valid)

    profile = vh_meta["profile"]
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(
    driver="GTiff",
    count=2,
    dtype="float32",
    nodata=None,
    compress="deflate",
    predictor=3,
    tiled=True,
    blockxsize=256,
    blockysize=256,
    BIGTIFF="IF_SAFER",
)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        # Current training-data audit indicates VH first, VV second.
        dst.write(vh_db, 1)
        dst.write(vv_db, 2)
        dst.write_mask(
        common_valid.astype(np.uint8) * 255
        )
        dst.set_band_description(1, "VH_dB_gamma0")
        dst.set_band_description(2, "VV_dB_gamma0")

        dst.update_tags(
            acquisition_time_utc=args.acquisition_time,
            source="Copernicus Browser",
            source_product="Sentinel-1 IW GRD",
            polarization_order="VH,VV",
            calibration="gamma0",
            scale="decibel",
            preprocessing="linear_gamma0_to_db",
            common_valid_pixels=str(int(common_valid.sum())),
            invalid_pixels=str(int((~common_valid).sum())),
            vh_invalid_fill_db=str(vh_fill),
            vv_invalid_fill_db=str(vv_fill),
        )

    print(f"Created: {output_path}")
    print(f"Common valid pixels: {int(common_valid.sum())}")
    print(f"Invalid/fill pixels: {int((~common_valid).sum())}")
    print(f"VH fill: {vh_fill:.6f} dB")
    print(f"VV fill: {vv_fill:.6f} dB")
    print("Band order: VH, VV")


if __name__ == "__main__":
    main()