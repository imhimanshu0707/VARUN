"""Complete Phase 2 pipeline runner for VARUN drift simulation."""


import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from shapely.geometry import MultiPoint, mapping, shape
from shapely.ops import unary_union

import numpy as np
import xarray as xr

from app.artifacts import (
    create_particle_positions_geojson,
    create_trajectory_geojson,
    write_json_manifest,
    write_netcdf_dataset,
)
from app.config import Phase2Settings
from app.forcing import audit_forcing_file, create_combined_readers
from app.forcing.models import ForcingConfig
from app.hindcast import HindcastRunner
from app.origin import calculate_origin_density, generate_contours, get_density_bounds
from app.reconstruction import ReconstructionRunner
from app.seeding import SeedingConfig
from app.simulation.engine import DriftSimulationEngine
from app.utils import (
    ensure_output_dir,
    generate_run_id,
    get_logger,
    normalize_simulation_time,
    to_iso_utc,
)
from app.forecast import ForecastRunner

logger = get_logger(__name__)


class Phase2CompleteRunner:
    """Complete Phase 2 runner with all required artifacts and validation."""

    def __init__(self, settings: Optional[Phase2Settings] = None):
        """Initialize runner."""
        self.settings = settings or Phase2Settings()
        self.engine = DriftSimulationEngine()
        self.run_id = None
        self.run_dir = None

    def run(
        self,
        case_id: str,
        scene_id: str,
        observation_time: datetime,
        spill_polygon_geojson: dict,
        forcing_config: ForcingConfig,
        particle_count: int = 1500,
        release_ages_hours: Optional[list[int]] = None,
        forecast_hours: float = 12.0,
    ) -> dict:
        """
        Execute complete Phase 2 workflow.

        Returns:
            Dictionary with execution summary and artifact paths
        """
        self.run_id = generate_run_id()
        self.run_dir = ensure_output_dir(self.settings.output_dir, self.run_id)

        logger.info(f"\n{'='*70}")
        logger.info(f"VARUN Phase 2 Complete Execution")
        logger.info(f"{'='*70}")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Case ID: {case_id}")
        logger.info(f"Scene ID: {scene_id}")
        logger.info(f"Output dir: {self.run_dir}")

        observation_time = normalize_simulation_time(observation_time, "observation_time")
        release_ages_hours = release_ages_hours or self.settings.release_ages_hours

        all_artifacts = []
        validation_checks = {}

        try:
            # ========== Step 1: Forcing Audit ==========
            logger.info(f"\n[1/9] Auditing forcing files...")
            forcing_audit = self._audit_forcing(forcing_config, validation_checks)

            # ========== Step 2: Create Readers ==========
            logger.info(f"[2/9] Creating OpenDrift readers...")
            readers = create_combined_readers(
                current_file=forcing_config.current_file,
                wind_file=forcing_config.wind_file,
                combined_file=forcing_config.combined_file,
            )

            # ========== Step 3: Run Hindcast ==========
            logger.info(f"[3/9] Running backward ensemble...")
            seeding_config = SeedingConfig(
                particle_count=particle_count,
                seed_buffer_km=self.settings.seed_buffer_km,
                random_seed=self.settings.random_seed,
            )

            hindcast_runner = HindcastRunner(self.engine)
            hindcast_result = hindcast_runner.run_hindcast(
                run_id=self.run_id,
                observation_time=observation_time,
                spill_polygon_geojson=spill_polygon_geojson,
                release_ages_hours=release_ages_hours,
                readers=readers,
                seeding_config=seeding_config,
            )

            successful_ages = [
                ra for ra in hindcast_result.release_ages
                if ra.simulation_result is not None
            ]
            validation_checks["hindcast"] = {
                "status": "PASS" if len(successful_ages) > 0 else "FAIL",
                "message": f"{len(successful_ages)}/{len(release_ages_hours)} hindcasts successful",
            }

            # Save hindcast artifacts
            hindcast_dir = self.run_dir / "backward"
            hindcast_dir.mkdir(exist_ok=True)

            for ra in successful_ages:
                try:
                    output_file = hindcast_dir / f"release-{ra.release_age_hours}h.nc"
                    # trajectory_lons/lats are (n_times, n_particles)
                    ds = xr.Dataset(
                        {
                            "lon": (["time", "particle"], ra.simulation_result.trajectory_lons),
                            "lat": (["time", "particle"], ra.simulation_result.trajectory_lats),
                        },
                        coords={
                            "time": ra.simulation_result.trajectory_times,
                            "particle": np.arange(ra.simulation_result.particle_count),
                        },
                    )
                    write_netcdf_dataset(ds, output_file)
                    all_artifacts.append(("backward_trajectory", str(output_file)))
                except Exception as e:
                    logger.warning(f"Could not save hindcast {ra.release_age_hours}h: {e}")


            backward_features = []

            for release_age in successful_ages:
                simulation_result = release_age.simulation_result

                particle_indices = list(
                    range(
                        min(
                            simulation_result.particle_count,
                            200,
                        )
                    )
                )

                trajectory_geojson = create_trajectory_geojson(
                    simulation_result.trajectory_lats,
                    simulation_result.trajectory_lons,
                    simulation_result.trajectory_times,
                    particle_indices=particle_indices,
                )

                for feature in trajectory_geojson["features"]:
                    feature["properties"].update(
                        {
                            "case_id": case_id,
                            "phase2_run_id": self.run_id,
                            "direction": "backward",
                            "release_age_hours": (
                                release_age.release_age_hours
                            ),
                            "detection_time_utc": to_iso_utc(
                                observation_time
                            ),
                            "estimated_release_time_utc": to_iso_utc(
                                observation_time
                                - timedelta(
                                    hours=release_age.release_age_hours
                                )
                            ),
                        }
                    )

                backward_features.extend(
                    trajectory_geojson["features"]
                )

            backward_tracks = {
                "type": "FeatureCollection",
                "features": backward_features,
            }

            backward_tracks_file = (
                self.run_dir / "backward_tracks.geojson"
            )

            with backward_tracks_file.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    backward_tracks,
                    file,
                    indent=2,
                )

            all_artifacts.append(
                (
                    "backward_tracks",
                    str(backward_tracks_file),
                )
            )
            # ========== Step 4: Calculate Origin Density ==========
            logger.info(f"[4/9] Calculating origin density...")
            density_ds = calculate_origin_density(hindcast_result)

            if density_ds is not None:
                density_file = self.run_dir / "origin" / "density.nc"
                density_file.parent.mkdir(exist_ok=True)
                write_netcdf_dataset(density_ds, density_file)
                all_artifacts.append(("origin_density", str(density_file)))
                validation_checks["origin_density"] = {"status": "PASS"}
                logger.info("✓ Origin density calculated")
            else:
                logger.warning("Could not calculate origin density")
                validation_checks["origin_density"] = {"status": "FAIL"}
                density_ds = None

            # ========== Step 5: Generate Origin Contours ==========
            logger.info(f"[5/9] Generating origin contours...")
            contours_50 = None
            contours_75 = None
            contours_90 = None

            if density_ds is not None:
                for level in [0.50, 0.75, 0.90]:
                    contours_gdf = generate_contours(
                        density_ds,
                        levels=[level],
                    )
                    if contours_gdf is not None and len(contours_gdf) > 0:
                        density_percent = int(level * 100)
                        filename = f"origin_{density_percent}.geojson"
                        contours_file = self.run_dir / filename

                        merged_geometry = unary_union(
                            contours_gdf.geometry.tolist()
                        )

                        contour_feature = {
                            "type": "Feature",
                            "properties": {
                                "case_id": case_id,
                                "phase2_run_id": self.run_id,
                                "density_level": density_percent,
                                "contour_level": level,
                                "semantics": "ENSEMBLE_DENSITY_REGION",
                            },
                            "geometry": mapping(merged_geometry),
                        }

                        with contours_file.open(
                            "w",
                            encoding="utf-8",
                        ) as file:
                            json.dump(
                                contour_feature,
                                file,
                                indent=2,
                            )

                        all_artifacts.append(
                            ("origin_contours", str(contours_file))
                        )
                        logger.info(
                            "Generated %.0f%% origin contour",
                            level * 100,
                        )

                validation_checks["origin_contours"] = {"status": "PASS"}
            else:
                validation_checks["origin_contours"] = {"status": "BLOCKED", "reason": "No origin density"}

            # ========== Step 6: Rank Release Ages ==========
            logger.info(f"[6/9] Ranking release ages...")
            (
                release_scores,
                candidate_reconstructions,
                candidate_origins,
            ) = self._rank_release_ages(
                hindcast_result=hindcast_result,
                observation_time=observation_time,
                spill_polygon_geojson=spill_polygon_geojson,
                readers=readers,
                seeding_config=seeding_config,
            )

            # Save release time scores
            scores_csv = (
                self.run_dir / "release_time_scores.csv"
            )

            score_columns = [
                "release_age_hours",
                "release_time_utc",
                "status",
                "reconstruction_score",
                "rank",
                "iou",
                "particle_coverage",
                "centroid_error_km",
            ]

            with scores_csv.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=score_columns,
                )
                writer.writeheader()

                for age in release_ages_hours:
                    score_info = release_scores.get(
                        age,
                        {},
                    )
                    success = (
                        score_info.get("status")
                        == "SUCCESS"
                    )

                    writer.writerow(
                        {
                            "release_age_hours": age,
                            "release_time_utc": (
                                score_info.get(
                                    "release_time",
                                    "",
                                )
                            ),
                            "status": score_info.get(
                                "status",
                                "FAILED",
                            ),
                            "reconstruction_score": (
                                score_info.get("score", "")
                                if success
                                else ""
                            ),
                            "rank": (
                                score_info.get("rank", "")
                                if success
                                else ""
                            ),
                            "iou": (
                                score_info.get("iou", "")
                                if success
                                else ""
                            ),
                            "particle_coverage": (
                                score_info.get(
                                    "particle_coverage",
                                    "",
                                )
                                if success
                                else ""
                            ),
                            "centroid_error_km": (
                                score_info.get(
                                    "centroid_error_km",
                                    "",
                                )
                                if success
                                else ""
                            ),
                        }
                    )
            all_artifacts.append(("release_time_scores", str(scores_csv)))

            best_release_age = self._get_best_release_age(release_scores)
            if best_release_age is None:
                logger.warning("No best release age found")
                validation_checks["release_ranking"] = {"status": "FAIL"}
                return self._create_failed_response("No successful release age")

            validation_checks["release_ranking"] = {"status": "PASS", "best_age": best_release_age}

            selected_density_ds = calculate_origin_density(
                hindcast_result,
                release_age_hours=best_release_age,
            )

            if selected_density_ds is None:
                validation_checks["origin_density"] = {
                    "status": "FAIL",
                    "reason": (
                        "Selected-age origin density "
                        "was not generated"
                    ),
                }
                return self._create_failed_response(
                    "Selected-age origin density unavailable"
                )

            density_ds = selected_density_ds

            selected_density_file = (
                self.run_dir / "origin" / "density.nc"
            )
            write_netcdf_dataset(
                density_ds,
                selected_density_file,
            )

            validation_checks["origin_density"].update(
                {
                    "release_age_hours": best_release_age,
                    "semantics": (
                        "SELECTED_RELEASE_AGE_DENSITY"
                    ),
                }
            )

            selected_contour_count = 0

            for level in [0.50, 0.75, 0.90]:
                contours_gdf = generate_contours(
                    density_ds,
                    levels=[level],
                )

                if (
                    contours_gdf is None
                    or len(contours_gdf) == 0
                ):
                    raise ValueError(
                        "Could not generate selected-age "
                        f"{int(level * 100)}% contour"
                    )

                density_percent = int(level * 100)
                merged_geometry = unary_union(
                    contours_gdf.geometry.tolist()
                )

                contour_feature = {
                    "type": "Feature",
                    "properties": {
                        "case_id": case_id,
                        "phase2_run_id": self.run_id,
                        "density_level": density_percent,
                        "contour_level": level,
                        "release_age_hours": (
                            best_release_age
                        ),
                        "semantics": (
                            "SELECTED_RELEASE_AGE_"
                            "ENSEMBLE_DENSITY_REGION"
                        ),
                    },
                    "geometry": mapping(
                        merged_geometry
                    ),
                }

                contour_file = (
                    self.run_dir
                    / f"origin_{density_percent}.geojson"
                )

                with contour_file.open(
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        contour_feature,
                        file,
                        indent=2,
                    )

                selected_contour_count += 1

            validation_checks["origin_contours"] = {
                "status": (
                    "PASS"
                    if selected_contour_count == 3
                    else "FAIL"
                ),
                "release_age_hours": best_release_age,
                "semantics": (
                    "SELECTED_RELEASE_AGE_CONTOURS"
                ),
            }

            # ========== Step 7: Run Forward Reconstruction ==========
            logger.info(f"[7/9] Running forward reconstruction...")

            best_origin = candidate_origins.get(
                best_release_age
            )
            reconstruction = (
                candidate_reconstructions.get(
                    best_release_age
                )
            )

            if best_origin is None or reconstruction is None:
                validation_checks["reconstruction"] = {
                    "status": "FAIL",
                    "reason": (
                        "Best candidate reconstruction "
                        "was not available"
                    ),
                }
                return self._create_failed_response(
                    "Best candidate reconstruction unavailable"
                )

            best_origin_lat = best_origin["lat"]
            best_origin_lon = best_origin["lon"]

            if reconstruction:
                logger.info("✓ Reconstruction complete")
                recon_file = self.run_dir / "reconstruction" / "trajectory.nc"
                recon_file.parent.mkdir(exist_ok=True)
                ds_recon = xr.Dataset(
                    {
                        "lon": (["time", "particle"], reconstruction.simulation_result.trajectory_lons),
                        "lat": (["time", "particle"], reconstruction.simulation_result.trajectory_lats),
                    },
                    coords={
                        "time": reconstruction.simulation_result.trajectory_times,
                        "particle": np.arange(reconstruction.simulation_result.particle_count),
                    },
                )
                write_netcdf_dataset(ds_recon, recon_file)
                all_artifacts.append(("reconstruction", str(recon_file)))

                # Create canonical forward-reconstruction GeoJSON.
                final_lats, final_lons = (
                    reconstruction.simulation_result.get_final_positions()
                )

                valid_positions = [
                    (float(lon), float(lat))
                    for lat, lon in zip(final_lats, final_lons)
                    if np.isfinite(lat) and np.isfinite(lon)
                ]

                if len(valid_positions) < 3:
                    raise ValueError(
                        "Forward reconstruction produced fewer than "
                        "three valid final positions"
                    )

                reconstructed_geometry = MultiPoint(
                    valid_positions
                ).convex_hull

                forward_reconstruction = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {
                                "case_id": case_id,
                                "phase2_run_id": self.run_id,
                                "scenario": "forward_reconstruction",
                                "release_age_hours": best_release_age,
                                "centroid_error_km": (
                                    reconstruction.metrics
                                    .centroid_distance_km
                                ),
                                "particle_coverage": (
                                    reconstruction.metrics
                                    .particle_coverage
                                ),
                                "consistency_score": (
                                    reconstruction.metrics
                                    .overall_score
                                ),
                            },
                            "geometry": mapping(
                                reconstructed_geometry
                            ),
                        }
                    ],
                }

                recon_geojson_file = (
                    self.run_dir
                    / "forward_reconstruction.geojson"
                )

                with recon_geojson_file.open(
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        forward_reconstruction,
                        file,
                        indent=2,
                    )

                all_artifacts.append(
                    (
                        "forward_reconstruction",
                        str(recon_geojson_file),
                    )
                )

                # Save reconstruction metrics
                metrics_file = self.run_dir / "reconstruction_metrics.json"
                reconstruction.metrics

                metrics_data = {
                    "centroid_distance_km": reconstruction.metrics.centroid_distance_km,
                    "overlap_score": reconstruction.metrics.overlap_score,
                    "particle_coverage": reconstruction.metrics.particle_coverage,
                    "overall_score": reconstruction.metrics.overall_score,
                }
                write_json_manifest(metrics_data, metrics_file)
                all_artifacts.append(("reconstruction_metrics", str(metrics_file)))

                validation_checks["reconstruction"] = {"status": "PASS"}
            else:
                logger.warning("Reconstruction failed")
                validation_checks["reconstruction"] = {"status": "FAIL"}
                reconstruction = None

            # ========== Step 8: Run Forecast ==========
            logger.info(f"[8/9] Running future forecast...")
            forecast_result = None

            if reconstruction:
                forecast_runner = ForecastRunner(self.engine)
                forecast_result = forecast_runner.run_forecast(
                    run_id=self.run_id,
                    reconstruction=reconstruction,
                    forecast_hours=forecast_hours,
                    readers=readers,
                )

                if forecast_result:
                    logger.info("✓ Forecast complete")

                    # Save forecast NetCDF
                    forecast_file = self.run_dir / "forecast" / "forecast.nc"
                    forecast_file.parent.mkdir(exist_ok=True)
                    ds_forecast = xr.Dataset(
                        {
                            "lon": (["time", "particle"], forecast_result.trajectory_lons),
                            "lat": (["time", "particle"], forecast_result.trajectory_lats),
                        },
                        coords={
                            "time": forecast_result.trajectory_times,
                            "particle": np.arange(forecast_result.particle_count),
                        },
                    )
                    write_netcdf_dataset(ds_forecast, forecast_file)
                    all_artifacts.append(("forecast", str(forecast_file)))

                    # Save forecast GeoJSON
                    final_lats, final_lons = forecast_result.get_final_positions()
                    forecast_geojson = create_particle_positions_geojson(
                        final_lats, final_lons,
                        properties={"scenario": "forecast"}
                    )
                    forecast_geojson_file = self.run_dir / "forecast" / "forecast.geojson"
                    with open(forecast_geojson_file, "w") as f:
                        json.dump(forecast_geojson, f)
                    all_artifacts.append(("forecast_geojson", str(forecast_geojson_file)))

                    validation_checks["forecast"] = {"status": "PASS"}
                else:
                    logger.warning("Forecast failed")
                    validation_checks["forecast"] = {"status": "FAIL"}

            # ========== Step 9: Generate Phase 3 Search Window ==========
            logger.info(f"[9/9] Generating Phase 3 search window...")

            if density_ds is not None:
                lat_min, lat_max, lon_min, lon_max = get_density_bounds(density_ds, level=0.9)
            else:

                poly = shape(spill_polygon_geojson)
                bounds = poly.bounds  # (minx, miny, maxx, maxy)
                lon_min, lat_min, lon_max, lat_max = bounds

            # Calculate release and search times
            release_time = observation_time - timedelta(hours=best_release_age)

            best_reconstruction_score = release_scores[
                best_release_age
            ]["score"]
            plausible_score_floor = (
                best_reconstruction_score * 0.80
            )

            plausible_age_values = [
                age
                for age, score_info
                in release_scores.items()
                if (
                    score_info.get("status") == "SUCCESS"
                    and score_info.get("rank", 999) <= 2
                    and score_info.get("score", 0.0)
                    >= plausible_score_floor
                )
            ]

            if not plausible_age_values:
                plausible_age_values = [
                    best_release_age
                ]

            release_window_start = (
                observation_time
                - timedelta(
                    hours=max(plausible_age_values)
                )
            )
            release_window_end = (
                observation_time
                - timedelta(
                    hours=min(plausible_age_values)
                )
            )

            validation_checks["release_ranking"].update(
                {
                    "plausible_ages": (
                        plausible_age_values
                    ),
                    "score_floor_ratio": 0.80,
                }
            )

            observed_geometry = shape(
                spill_polygon_geojson
            )
            observed_centroid = observed_geometry.centroid

            search_region_geometry = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [lon_min, lat_min],
                        [lon_max, lat_min],
                        [lon_max, lat_max],
                        [lon_min, lat_max],
                        [lon_min, lat_min],
                    ]
                ],
            }

            hindcast_corridor_geometry = {
                "type": "LineString",
                "coordinates": [
                    [best_origin_lon, best_origin_lat],
                    [
                        float(observed_centroid.x),
                        float(observed_centroid.y),
                    ],
                ],
            }

            search_window_payload = {
                "contract_version": "phase2-to-phase3-v1",
                "case_id": case_id,
                "phase2_run_id": self.run_id,
                "detection_time_utc": to_iso_utc(
                    observation_time
                ),
                "crs": "EPSG:4326",
                "release_window": {
                    "start_utc": to_iso_utc(
                        release_window_start
                    ),
                    "end_utc": to_iso_utc(
                        release_window_end
                    ),
                    "time_buffer_minutes": 60,
                },
                "search_region": {
                    "geometry": search_region_geometry,
                    "spatial_buffer_m": (
                        self.settings.seed_buffer_km * 1000.0
                    ),
                },
                "hindcast_corridor": {
                    "geometry": hindcast_corridor_geometry,
                    "direction_deg": None,
                    "time_bands": [],
                },
            }

            search_window_file = (
                self.run_dir / "search_window.json"
            )

            with search_window_file.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    search_window_payload,
                    file,
                    indent=2,
                )
            all_artifacts.append(("search_window", str(search_window_file)))

            validation_checks["search_window"] = {"status": "PASS"}
            logger.info("✓ Phase 3 search window generated")

            # ========== Generate Validation Report ==========
            validation_report = {
                "timestamp": datetime.utcnow().isoformat(),
                "run_id": self.run_id,
                "case_id": case_id,
                "opendrift_version": "1.14.11",
                "checks": validation_checks,
                "status": "PASS" if all(c.get("status") == "PASS" for c in validation_checks.values()) else "PARTIAL",
            }

            validation_file = self.run_dir / "validation_report.json"
            write_json_manifest(validation_report, validation_file)
            all_artifacts.append(("validation", str(validation_file)))

            # ========== Generate Summary ==========
            summary = {
                "status": "SUCCESS",
                "run_id": self.run_id,
                "case_id": case_id,
                "scene_id": scene_id,
                "observation_time": to_iso_utc(observation_time),
                "phase2_run_id": self.run_id,
                "opendrift_version": "1.14.11",
                "particle_count": particle_count,
                "release_ages": release_ages_hours,
                "successful_release_ages": [ra.release_age_hours for ra in successful_ages],
                "best_release_age_hours": best_release_age,
                "best_release_time": to_iso_utc(release_time),
                "forecast_hours": forecast_hours,
                "output_directory": str(self.run_dir),
                "artifacts_count": len(all_artifacts),
                "validation_status": validation_report["status"],
                "timestamp": datetime.utcnow().isoformat(),
            }

            summary_file = self.run_dir / "summary.json"
            write_json_manifest(summary, summary_file)

            # ========== Generate Run Config ==========
            run_config = {
                "case_id": case_id,
                "scene_id": scene_id,
                "observation_time": to_iso_utc(observation_time),
                "particle_count": particle_count,
                "release_ages": release_ages_hours,
                "forecast_hours": forecast_hours,
                "forcing_files": {
                    "current_file": str(forcing_config.current_file) if forcing_config.current_file else None,
                    "wind_file": str(forcing_config.wind_file) if forcing_config.wind_file else None,
                    "combined_file": str(forcing_config.combined_file) if forcing_config.combined_file else None,
                },
                "random_seed": self.settings.random_seed,
                "opendrift_version": "1.14.11",
                "simulation_timestep_hours": self.settings.simulation_time_step_hours,
                "output_interval_hours": self.settings.output_interval_hours,
                "crs": "EPSG:4326",
                "coordinate_order": "[longitude, latitude]",
                "run_id": self.run_id,
            }

            config_file = self.run_dir / "run_config.json"
            write_json_manifest(run_config, config_file)

            logger.info(f"\n{'='*70}")
            logger.info(f"✓ Phase 2 COMPLETE")
            logger.info(f"{'='*70}")
            logger.info(f"Output directory: {self.run_dir}")
            logger.info(f"Artifacts: {len(all_artifacts)}")
            logger.info(f"Validation: {validation_report['status']}\n")

            return {
                "status": "SUCCESS",
                "run_id": self.run_id,
                "case_id": case_id,
                "output_dir": str(self.run_dir),
                "summary": summary,
                "validation": validation_report,
            }

        except Exception as e:
            logger.error(f"\n✗ Phase 2 FAILED: {e}", exc_info=True)
            return self._create_failed_response(str(e))

    def _audit_forcing(self, forcing_config: ForcingConfig, validation_checks: dict) -> dict:
        """Audit forcing files."""
        forcing_files = forcing_config.get_files()
        forcing_audit = {}

        for file_path in forcing_files:
            audit_result = audit_forcing_file(file_path)
            forcing_audit[file_path.name] = {
                "passed": audit_result.passed,
                "errors": audit_result.errors,
                "warnings": audit_result.warnings,
            }

            if not audit_result.passed:
                raise Exception(f"Forcing validation failed: {audit_result.errors}")

            logger.info(f"✓ {file_path.name} passed audit")

        validation_checks["forcing_audit"] = {"status": "PASS"}
        return forcing_audit

    def _rank_release_ages(
        self,
        hindcast_result,
        observation_time: datetime,
        spill_polygon_geojson: dict,
        readers: list,
        seeding_config: SeedingConfig,
    ) -> tuple[dict, dict, dict]:
        """
        Forward-reconstruct and rank every successful release age.

        Returns:
            scores:
                Metrics and rank for each candidate age.
            reconstructions:
                Successful reconstruction result by age.
            origins:
                Candidate origin latitude/longitude by age.
        """
        scores = {}
        reconstructions = {}
        origins = {}

        reconstruction_runner = ReconstructionRunner(
            self.engine
        )

        for release_age in hindcast_result.release_ages:
            age_hours = release_age.release_age_hours
            release_time = (
                observation_time
                - timedelta(hours=age_hours)
            )

            if release_age.simulation_result is None:
                scores[age_hours] = {
                    "status": "FAILED",
                    "score": 0.0,
                    "rank": None,
                    "release_time": to_iso_utc(
                        release_time
                    ),
                    "reason": "hindcast_failed",
                }
                continue

            try:
                age_density = calculate_origin_density(
                    hindcast_result,
                    release_age_hours=age_hours,
                )

                if age_density is None:
                    raise ValueError(
                        "Origin density was not generated"
                    )

                (
                    lat_min,
                    lat_max,
                    lon_min,
                    lon_max,
                ) = get_density_bounds(
                    age_density,
                    level=0.9,
                )

                origin_lat = (lat_min + lat_max) / 2
                origin_lon = (lon_min + lon_max) / 2

                reconstruction = reconstruction_runner.run(
                    run_id=(
                        f"{self.run_id}-{age_hours}H"
                    ),
                    best_origin_lat=origin_lat,
                    best_origin_lon=origin_lon,
                    observation_time=observation_time,
                    observed_polygon_geojson=(
                        spill_polygon_geojson
                    ),
                    release_age_hours=age_hours,
                    readers=readers,
                    seeding_config=seeding_config,
                )

                if reconstruction is None:
                    raise ValueError(
                        "Forward reconstruction failed"
                    )

                metrics = reconstruction.metrics

                scores[age_hours] = {
                    "status": "SUCCESS",
                    "score": metrics.overall_score,
                    "rank": None,
                    "release_time": to_iso_utc(
                        release_time
                    ),
                    "iou": metrics.overlap_score,
                    "particle_coverage": (
                        metrics.particle_coverage
                    ),
                    "centroid_error_km": (
                        metrics.centroid_distance_km
                    ),
                }

                reconstructions[age_hours] = (
                    reconstruction
                )
                origins[age_hours] = {
                    "lat": origin_lat,
                    "lon": origin_lon,
                }

            except Exception as exc:
                logger.exception(
                    "Release-age evaluation failed for %sh",
                    age_hours,
                )
                scores[age_hours] = {
                    "status": "FAILED",
                    "score": 0.0,
                    "rank": None,
                    "release_time": to_iso_utc(
                        release_time
                    ),
                    "reason": str(exc),
                }

        successful_scores = sorted(
            (
                (age, info)
                for age, info in scores.items()
                if info["status"] == "SUCCESS"
            ),
            key=lambda item: item[1]["score"],
            reverse=True,
        )

        for rank, (age, _) in enumerate(
            successful_scores,
            start=1,
        ):
            scores[age]["rank"] = rank

        return scores, reconstructions, origins

    def _get_best_release_age(self, scores: dict) -> Optional[int]:
        """Get best release age based on scores."""
        valid_scores = [
            (age, info) for age, info in scores.items()
            if info["status"] == "SUCCESS"
        ]

        if not valid_scores:
            return None

        # Return the one with lowest rank (rank 1 is best)
        best = min(valid_scores, key=lambda x: x[1].get("rank", float("inf")))
        return best[0]

    def _create_failed_response(self, error_msg: str) -> dict:
        """Create a failed response."""
        return {
            "status": "FAILED",
            "run_id": self.run_id,
            "error": error_msg,
            "output_dir": str(self.run_dir) if self.run_dir else None,
        }

def cli_main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="VARUN Phase 2 Oil Spill Drift Simulation"
    )

    parser.add_argument(
        "--case-id",
        required=True,
        help="Case ID, e.g. CASE-2026-001",
    )

    parser.add_argument(
        "--scene-id",
        default="SCENE-001",
        help="Scene ID",
    )

    parser.add_argument(
        "--observation-time",
        required=True,
        help="Observation time in ISO format, e.g. 2026-09-03T08:30:00",
    )

    parser.add_argument(
        "--polygon-file",
        required=True,
        help="Path to spill polygon GeoJSON",
    )

    parser.add_argument(
        "--current-file",
        help="Path to ocean current NetCDF",
    )

    parser.add_argument(
        "--wind-file",
        help="Path to wind NetCDF",
    )

    parser.add_argument(
        "--combined-file",
        help="Path to combined current and wind NetCDF",
    )

    parser.add_argument(
        "--particles",
        type=int,
        default=1500,
        help="Number of particles",
    )

    parser.add_argument(
        "--forecast-hours",
        type=float,
        default=24.0,
        help="Forward forecast duration in hours",
    )

    args = parser.parse_args()

    if args.combined_file:
        if args.current_file or args.wind_file:
            parser.error(
                "--combined-file cannot be used with "
                "--current-file or --wind-file"
            )
    elif not args.current_file or not args.wind_file:
        parser.error(
            "Provide either --combined-file, or both "
            "--current-file and --wind-file"
        )

    # ---------------------------------------------------------
    # Load spill polygon from actual GeoJSON file
    # ---------------------------------------------------------

    polygon_path = Path(args.polygon_file)

    if not polygon_path.exists():
        raise FileNotFoundError(
            f"Polygon file not found: {polygon_path}"
        )

    with polygon_path.open("r", encoding="utf-8") as f:
        polygon_data = json.load(f)

    # Support FeatureCollection
    if polygon_data.get("type") == "FeatureCollection":

        features = polygon_data.get("features", [])

        if not features:
            raise ValueError(
                "Polygon GeoJSON FeatureCollection contains no features"
            )

        geometries = []

        for feature in features:
            geometry = feature.get("geometry")

            if geometry is None:
                continue

            geometries.append(shape(geometry))

        if not geometries:
            raise ValueError(
                "No valid geometry found in FeatureCollection"
            )

        polygon_geometry = unary_union(geometries)

    # Support Feature
    elif polygon_data.get("type") == "Feature":

        geometry = polygon_data.get("geometry")

        if geometry is None:
            raise ValueError(
                "GeoJSON Feature does not contain geometry"
            )

        polygon_geometry = shape(geometry)

    # Support direct Polygon / MultiPolygon
    elif polygon_data.get("type") in ("Polygon", "MultiPolygon"):

        polygon_geometry = shape(polygon_data)

    else:
        raise ValueError(
            f"Unsupported GeoJSON type: {polygon_data.get('type')}"
        )

    if polygon_geometry.is_empty:
        raise ValueError("Spill polygon geometry is empty")

    if not polygon_geometry.is_valid:
        polygon_geometry = polygon_geometry.buffer(0)

    if polygon_geometry.is_empty:
        raise ValueError(
            "Spill polygon became empty after geometry repair"
        )

    spill_polygon = mapping(polygon_geometry)

    print("\n" + "=" * 70)
    print("VARUN PHASE 2")
    print("=" * 70)
    print(f"Case ID:          {args.case_id}")
    print(f"Scene ID:         {args.scene_id}")
    print(f"Observation time: {args.observation_time}")
    print(f"Polygon file:     {polygon_path}")
    print(f"Polygon type:     {polygon_geometry.geom_type}")
    print(f"Polygon bounds:   {polygon_geometry.bounds}")
    print(f"Current file:     {args.current_file}")
    print(f"Wind file:        {args.wind_file}")
    print(f"Combined file:    {args.combined_file}")
    print(f"Particles:        {args.particles}")
    print(f"Forecast hours:   {args.forecast_hours}")
    print("=" * 70 + "\n")

    # ---------------------------------------------------------
    # Parse observation time
    # ---------------------------------------------------------

    observation_time = datetime.fromisoformat(
        args.observation_time
    )

    # ---------------------------------------------------------
    # Create forcing configuration
    # ---------------------------------------------------------

    forcing_config = ForcingConfig(
        current_file=(
            Path(args.current_file)
            if args.current_file
            else None
        ),
        wind_file=(
            Path(args.wind_file)
            if args.wind_file
            else None
        ),
        combined_file=(
            Path(args.combined_file)
            if args.combined_file
            else None
        ),
    )

    runner = Phase2CompleteRunner()

    result = runner.run(
        case_id=args.case_id,
        scene_id=args.scene_id,
        observation_time=observation_time,
        spill_polygon_geojson=spill_polygon,
        forcing_config=forcing_config,
        particle_count=args.particles,
        release_ages_hours=[6, 12, 24, 36],
        forecast_hours=args.forecast_hours,
    )

    print("\n" + "=" * 70)
    print("PHASE 2 RESULT")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )

    print("=" * 70)

    raise SystemExit(
        0 if result.get("status") == "SUCCESS" else 1
    )


if __name__ == "__main__":
    cli_main()
