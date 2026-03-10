import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import torch

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Pipeline,
    pipeline,
)

from .base import BaseTransformerModel

logger = logging.getLogger(__name__)

ModelInitResult: TypeAlias = Tuple[
    AutoModelForTokenClassification, AutoTokenizer, Pipeline, bool
]


class NERModel(BaseTransformerModel[AutoModelForTokenClassification]):  # type: ignore
    """
    Entity recognition model wrapper.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.pipe: Optional[Pipeline] = None

    async def initialize(self) -> None:
        def blocking_init() -> ModelInitResult:
            try:
                model = AutoModelForTokenClassification.from_pretrained(self.model_name)
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                model, use_float16 = self._setup_model_precision(model)
                model.eval()

                ner_pipe = pipeline(
                    "ner",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                    framework="pt",
                    aggregation_strategy="simple",
                )

                return model, tokenizer, ner_pipe, use_float16
            except Exception as e:
                logger.error(f"Failed to initialize NER model {self.model_name}: {e}")
                raise

        (
            self.model,
            self.tokenizer,
            self.pipe,
            self.use_float16,
        ) = await asyncio.to_thread(blocking_init)
        logger.info(f"Entity model '{self.model_name}' initialized successfully.")

    async def predict(self, text: str) -> Tuple[List[Dict[str, Any]], int]:
        """
        Performs entity recognition on the provided text.
        """
        if self.model is None or self.tokenizer is None or self.pipe is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        async with self._inference_lock:
            return await asyncio.to_thread(self._predict_blocking, text)

    def _predict_blocking(self, text: str) -> Tuple[List[Dict[str, Any]], int]:
        try:
            _, token_count = self._tokenize(text)
            assert self.pipe is not None, "Pipe cannot be None at this point"

            with torch.no_grad():
                if self.use_float16 and self.device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        results = self.pipe(text)
                else:
                    results = self.pipe(text)
            return results, token_count
        except Exception as e:
            logger.error(f"NER prediction failed for text '{text}': {e}")
            raise
