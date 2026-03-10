import asyncio
import logging
from typing import Optional, Tuple, cast

import torch
from torch import nn

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)

from .base import BaseTransformerModel

logger = logging.getLogger(__name__)


class ClassificationModel(BaseTransformerModel[PreTrainedModel]):
    """
    Classification model wrapper for sequence classification tasks.
    """

    async def initialize(self) -> None:
        def blocking_init() -> (
            Tuple[Optional[PreTrainedModel], Optional[AutoTokenizer], bool]
        ):
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    clean_up_tokenization_spaces=True,
                )

                model_with_precision, use_float16 = self._setup_model_precision(model)
                model_with_precision.eval()

                return (
                    cast(PreTrainedModel, model_with_precision),
                    tokenizer,
                    use_float16,
                )
            except Exception as e:
                logger.error(f"Failed to initialize model {self.model_name}: {e}")
                raise

        self.model, self.tokenizer, self.use_float16 = await asyncio.to_thread(
            blocking_init
        )
        logger.info(
            f"Classification model '{self.model_name}' initialized successfully."
        )

    async def predict(self, text: str) -> Tuple[Tuple[float, float], int]:
        """
        Performs a forward pass and returns (negative_prob, positive_prob) tuple
        along with the token count.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        async with self._inference_lock:
            return await asyncio.to_thread(self._predict_blocking, text)

    def _predict_blocking(self, text: str) -> Tuple[Tuple[float, float], int]:
        try:
            assert self.model is not None, "Model not initialized"
            assert self.tokenizer is not None, "Tokenizer not initialized"

            inputs, token_count = self._tokenize(text)
            with torch.no_grad():
                if self.use_float16 and self.device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(**inputs)  # type: ignore
                else:
                    outputs = self.model(**inputs)  # type: ignore

            logits = outputs.logits
            probs = nn.functional.softmax(logits, dim=1)[0].cpu().tolist()
            return (float(probs[0]), float(probs[1])), token_count
        except Exception as e:
            logger.error(f"Prediction failed for text '{text}': {e}")
            raise
