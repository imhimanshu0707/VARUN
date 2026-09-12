import json
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import CRS, Transformer
from rasterio.features import shapes
from shapely import make_valid
from shapely.geometry import (
    GeometryCollection,
    MultiPolygon,
    Polygon,
    mapping,
    shape,
)
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union


TARGET_CRS = "EPSG:4326"
CONTRACT_VERSION = "phase1-to-phase2-v1"


def _polygonal_geometry(geometry):
    """
    Keep only Polygon/MultiPolygon content after geometry repair.
    """

    geometry = make_valid(geometry)

    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry

    if isinstance(geometry, GeometryCollection):
        polygons = []

        for item in geometry.geoms:
            if isinstance(item, Polygon):
                polygons.append(item)
            elif isinstance(item, MultiPolygon):
                polygons.extend(item.geoms)

        if polygons:
            return unary_union(polygons)

    return None


def _local_utm_crs(
    longitude: float,
    latitude: float,
) -> CRS:
    """
    Select a local UTM CRS for metric area/perimeter calculation.
    """

    zone = int((longitude + 180.0) // 6.0) + 1
    zone = max(1, min(zone, 60))

    epsg = (
        32600 + zone
        if latitude >= 0
        else 32700 + zone
    )

    return CRS.from_epsg(epsg)


def build_spill_geometry(
    mask: np.ndarray,
    transform: Any,
    source_crs: Any,
    probability_map: np.ndarray | None = None,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Polygonize the cleaned binary mask and calculate geographic
    and metric properties.

    Returned geometry always uses EPSG:4326 with coordinate order:
    [longitude, latitude].
    """

    binary_mask = np.asarray(mask, dtype=np.uint8)

    if binary_mask.ndim != 2:
        raise ValueError(
            f"Mask must be 2D, got shape {binary_mask.shape}"
        )

    if source_crs is None:
        raise ValueError(
            "Source CRS is required for geographic polygonization"
        )

    if transform is None:
        raise ValueError(
            "Affine transform is required for polygonization"
        )

    unique_values = set(
        np.unique(binary_mask).tolist()
    )

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            "Mask must contain only 0 and 1"
        )

    source_polygons = []

    for geometry_mapping, value in shapes(
        binary_mask,
        mask=binary_mask == 1,
        transform=transform,
        connectivity=8,
    ):
        if int(value) != 1:
            continue

        polygon = _polygonal_geometry(
            shape(geometry_mapping)
        )

        if polygon is not None and not polygon.is_empty:
            source_polygons.append(polygon)

    if not source_polygons:
        return {
            "feature": None,
            "geometry": None,
            "confidence": None,
            "centroid": None,
            "bounding_box": None,
            "area_km2": 0.0,
            "perimeter_km": 0.0,
        }

    source_geometry = _polygonal_geometry(
        unary_union(source_polygons)
    )

    if source_geometry is None or source_geometry.is_empty:
        raise ValueError(
            "Polygonization produced no valid polygon geometry"
        )

    to_wgs84 = Transformer.from_crs(
        CRS.from_user_input(source_crs),
        CRS.from_epsg(4326),
        always_xy=True,
    )

    wgs84_geometry = _polygonal_geometry(
        transform_geometry(
            to_wgs84.transform,
            source_geometry,
        )
    )

    if wgs84_geometry is None or wgs84_geometry.is_empty:
        raise ValueError(
            "Unable to transform spill geometry to EPSG:4326"
        )

    centroid = wgs84_geometry.centroid
    west, south, east, north = wgs84_geometry.bounds

    metric_crs = _local_utm_crs(
        longitude=float(centroid.x),
        latitude=float(centroid.y),
    )

    to_metric = Transformer.from_crs(
        CRS.from_epsg(4326),
        metric_crs,
        always_xy=True,
    )

    metric_geometry = transform_geometry(
        to_metric.transform,
        wgs84_geometry,
    )

    area_km2 = float(metric_geometry.area / 1_000_000.0)
    perimeter_km = float(metric_geometry.length / 1_000.0)

    confidence = None

    if probability_map is not None:
        probability = np.asarray(
            probability_map,
            dtype=np.float32,
        )

        if probability.shape != binary_mask.shape:
            raise ValueError(
                "Probability map shape must match mask shape"
            )

        oil_probabilities = probability[
            binary_mask == 1
        ]

        if oil_probabilities.size:
            confidence = float(
                np.mean(oil_probabilities)
            )

    feature_properties = {
        "contract_version": CONTRACT_VERSION,
        "confidence": confidence,
        "area_km2": area_km2,
        "perimeter_km": perimeter_km,
        "crs": TARGET_CRS,
    }

    if properties:
        feature_properties.update(properties)

    feature = {
        "type": "Feature",
        "properties": feature_properties,
        "geometry": mapping(wgs84_geometry),
    }

    return {
        "feature": feature,
        "geometry": feature["geometry"],
        "confidence": confidence,
        "centroid": {
            "longitude": float(centroid.x),
            "latitude": float(centroid.y),
        },
        "bounding_box": {
            "west": float(west),
            "south": float(south),
            "east": float(east),
            "north": float(north),
        },
        "area_km2": area_km2,
        "perimeter_km": perimeter_km,
        "metric_crs": metric_crs.to_string(),
    }


def write_spill_geojson(
    feature: dict[str, Any],
    output_path: str | Path,
) -> str:
    if feature.get("type") != "Feature":
        raise ValueError(
            "GeoJSON output must be a Feature"
        )

    geometry = feature.get("geometry")

    if not geometry or geometry.get("type") not in {
        "Polygon",
        "MultiPolygon",
    }:
        raise ValueError(
            "GeoJSON geometry must be Polygon or MultiPolygon"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open(
        "w",
        encoding="utf-8",
    ) as destination:
        json.dump(
            feature,
            destination,
            ensure_ascii=False,
            indent=2,
        )
        destination.write("\n")

    return str(output)