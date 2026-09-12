import numpy as np

from app.inference import pipeline as pipeline_module


class FakePredictor:
    def __init__(self, *args, **kwargs):
        pass

    def predict_probability(self, tile):
        height, width = tile.shape[-2:]

        return np.ones(
            (height, width),
            dtype=np.float32,
        )


def test_pipeline_returns_full_scene_probability_map(
    monkeypatch,
):
    monkeypatch.setattr(
        pipeline_module,
        "UNetPredictor",
        FakePredictor,
    )

    image = np.stack(
        [
            np.arange(360, dtype=np.float32).reshape(18, 20),
            np.arange(360, 720, dtype=np.float32).reshape(18, 20),
        ]
    )

    result = pipeline_module.run_inference(
        image=image,
        image_width=20,
        image_height=18,
        model_path="unused-by-fake-model.pth",
        tile_size=16,
        overlap=4,
        confidence_threshold=0.5,
        min_area_pixels=1,
    )

    assert result["probability_map"].shape == (18, 20)
    assert result["probability_map"].dtype == np.float32
    assert np.allclose(result["probability_map"], 1.0)

    assert result["mask"].shape == (18, 20)
    assert result["mask"].dtype == np.uint8
    assert np.all(result["mask"] == 1)

    assert result["oil_detected"] is True
    assert result["oil_pixel_count"] == 360
    assert result["total_pixel_count"] == 360
    assert result["oil_coverage_percent"] == 100.0
    assert len(result["detections"]) == 1