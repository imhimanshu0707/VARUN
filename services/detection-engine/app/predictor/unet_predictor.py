from typing import Any

import numpy as np
import torch

from app.postprocessing.mask_to_detection import mask_to_detections
from app.predictor.unet import UNet


class UNetPredictor:
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        min_area: int = 20,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = UNet(
            in_channels=2,
            out_channels=1,
        )

        state = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.threshold = threshold
        self.min_area = min_area

    def _prepare_tensor(self, image: Any) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image
        else:
            tensor = torch.as_tensor(
                image,
                dtype=torch.float32,
            )

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim != 4:
            raise ValueError(
                "Expected image shape (C,H,W) or (B,C,H,W), "
                f"got {tuple(tensor.shape)}"
            )

        if tensor.shape[0] != 1:
            raise ValueError(
                "Probability inference currently expects one tile "
                f"at a time, got batch size {tensor.shape[0]}"
            )

        if tensor.shape[1] != 2:
            raise ValueError(
                f"U-Net expects 2 input channels, got {tensor.shape[1]}"
            )

        height, width = tensor.shape[-2:]

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "U-Net tile height and width must be divisible by 16, "
                f"got {height}x{width}"
            )

        return tensor.to(
            device=self.device,
            dtype=torch.float32,
        )

    @torch.inference_mode()
    def predict_probability(self, image: Any) -> np.ndarray:
        """
        Return the oil probability raster for one tile.

        Output shape: (height, width)
        Output range: 0.0 to 1.0
        Output dtype: float32
        """

        tensor = self._prepare_tensor(image)

        logits = self.model(tensor)
        probabilities = torch.sigmoid(logits)

        probability = probabilities[0, 0].cpu().numpy()

        return probability.astype(
            np.float32,
            copy=False,
        )

    def predict(self, image: Any):
        """
        Backward-compatible bounding-box output.
        """

        probability = self.predict_probability(image)
        mask = probability >= self.threshold

        return mask_to_detections(
            mask=mask,
            probability=probability,
            min_area=self.min_area,
        )