from typing import TYPE_CHECKING, Optional

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from src.core.config import AppConfig

if TYPE_CHECKING:
    from src.core.shutdown import ShutdownManager
    from src.domain.transformers import ClassificationModel, NERModel


class AppState:
    """Holds references to models, engines, config, etc."""

    def __init__(self) -> None:
        self.config: Optional[AppConfig] = None
        self.shutdown_manager: Optional["ShutdownManager"] = None
        self.profanity_model: Optional["ClassificationModel"] = None
        self.ner_model: Optional["NERModel"] = None
        self.presidio_analyzer_engine: Optional[AnalyzerEngine] = None
        self.presidio_anonymizer_engine: Optional[AnonymizerEngine] = None


app_state = AppState()
