import asyncio
import logging
from typing import Generic, List, Optional, Tuple, TypeAlias, TypeVar, Union

import torch
from torch import nn

from transformers import AutoTokenizer, Pipeline, PreTrainedModel
from transformers.tokenization_utils_base import (
    BatchEncoding,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=PreTrainedModel)

TokenizerOutput: TypeAlias = BatchEncoding


class BaseTransformerModel(Generic[T]):
    """
    Base class for Transformer model wrappers with common functionality
    for initialization, inference, and cleanup. Handles different model types
    (classification, NER) through generic type T.
    """

    def __init__(self, model_name: str):
        """
        Initializes the base model wrapper.

        Args:
            model_name: The path or identifier of the pretrained model (e.g., "bert-base-uncased").
        """
        self.model_name: str = model_name
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model: Optional[T] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.pipe: Optional[Pipeline] = None
        self.use_float16: bool = False
        self._inference_lock = asyncio.Lock()

        logger.debug(
            f"BaseTransformerModel initialized for '{model_name}' on device '{self.device}'"
        )

    async def initialize(self) -> None:
        """
        Abstract method for loading and preparing the specific model and tokenizer.
        Must be implemented by subclasses. Should handle loading from pretrained,
        setting precision, moving to device, and setting eval mode.
        """
        raise NotImplementedError("Subclasses must implement initialize()")

    async def close(self) -> None:
        """
        Gracefully cleans up model and tokenizer resources.
        Moves model to CPU, deletes references, and clears CUDA cache if applicable.
        Uses asyncio.to_thread to avoid blocking the event loop.
        """
        model_name = self.model_name

        def blocking_close() -> None:
            """Performs cleanup in a separate thread."""
            nonlocal model_name
            try:
                logger.debug(f"Starting blocking close for model '{model_name}'")
                if hasattr(self, "model") and self.model is not None:
                    logger.debug(f"Moving model '{model_name}' to CPU.")
                    self.model.cpu()  # type: ignore[attr-defined]
                    del self.model
                    self.model = None
                    logger.debug(f"Model '{model_name}' reference deleted.")

                if hasattr(self, "tokenizer") and self.tokenizer is not None:
                    logger.debug(f"Deleting tokenizer reference for '{model_name}'.")
                    del self.tokenizer
                    self.tokenizer = None

                if hasattr(self, "pipe") and self.pipe is not None:
                    logger.debug(f"Deleting pipeline reference for '{model_name}'.")
                    del self.pipe
                    self.pipe = None

                if self.device.type == "cuda":
                    logger.debug(f"Emptying CUDA cache for '{model_name}'.")
                    torch.cuda.empty_cache()

                logger.debug(f"Blocking close finished for model '{model_name}'")

            except Exception as e:
                logger.error(
                    f"Error during model cleanup for '{model_name}': {e}", exc_info=True
                )

        logger.info(f"Initiating cleanup for model '{self.model_name}'...")
        await asyncio.to_thread(blocking_close)
        logger.info(f"Model '{self.model_name}' closed and cleaned up successfully.")

    def _tokenize(self, text: Union[str, List[str]]) -> Tuple[TokenizerOutput, int]:
        """
        Common tokenization method using the initialized AutoTokenizer.
        Handles padding, truncation, tensor conversion, moving to device,
        and token counting.

        Args:
            text: A single string or a list of strings to tokenize.

        Returns:
            A tuple containing:
                - The tokenized output (typically BatchEncoding) with tensors on the correct device.
                - The calculated token count of the first sequence after padding/truncation.

        Raises:
            RuntimeError: If the tokenizer has not been initialized.
        """
        if self.tokenizer is None:
            logger.error("Attempted to tokenize before tokenizer was initialized.")
            raise RuntimeError("Tokenizer not initialized.")

        logger.debug(f"Tokenizing input text on device '{self.device}'")
        inputs: BatchEncoding = self.tokenizer(  # type: ignore
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            logger.warning(
                f"Tokenizer output type {type(inputs)} lacks .to() method. Manual tensor moving might be needed if not already on device {self.device}."
            )

        token_count = 0
        if hasattr(inputs, "input_ids") and isinstance(inputs.input_ids, torch.Tensor):
            tensor = inputs.input_ids
            if tensor.ndim == 2:
                token_count = tensor.shape[1]
            elif tensor.ndim == 1:
                token_count = tensor.shape[0]
            else:
                logger.warning(
                    f"Unexpected input_ids tensor dimension: {tensor.ndim}. Cannot determine token count reliably."
                )
        elif (
            isinstance(inputs, dict)
            and "input_ids" in inputs
            and isinstance(inputs["input_ids"], torch.Tensor)
        ):
            logger.debug("Tokenizer output treated as dict for token counting.")
            tensor = inputs["input_ids"]
            if tensor.ndim == 2:
                token_count = tensor.shape[1]
            elif tensor.ndim == 1:
                token_count = tensor.shape[0]
            else:
                logger.warning(
                    f"Unexpected input_ids tensor dimension in dict: {tensor.ndim}."
                )

        if token_count == 0:
            logger.warning(
                f"Could not determine token count from tokenizer output. Type: {type(inputs)}, Keys/Attrs: {dir(inputs) if not isinstance(inputs, dict) else list(inputs.keys())}"
            )

        logger.debug(
            f"Tokenization complete. Token count (first sequence): {token_count}"
        )
        return inputs, token_count

    def _setup_model_precision(self, model: nn.Module) -> Tuple[nn.Module, bool]:
        """
        Configures the model's precision (float32 vs float16/half) based on hardware.
        Moves the model to the determined device.

        Args:
            model: The loaded PyTorch model (nn.Module).

        Returns:
            A tuple containing:
                - The model configured for the appropriate precision and device.
                - A boolean indicating if float16 (half precision) is being used.
        """
        use_float16 = False
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                logger.info(
                    f"Using half precision (float16) for model '{self.model_name}' on {self.device}."
                )
                model = model.half().to(self.device)
                use_float16 = True
            except RuntimeError as e:
                logger.warning(
                    f"Failed to use half precision for model '{self.model_name}' on {self.device}: {e}. Falling back to float32."
                )
                model = model.float().to(self.device)
        else:
            logger.info(
                f"Using standard precision (float32) for model '{self.model_name}' on {self.device}."
            )
            model = model.float().to(self.device)

        return model, use_float16
