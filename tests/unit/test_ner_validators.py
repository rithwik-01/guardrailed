from unittest.mock import AsyncMock, patch

import pytest

from src.domain.validators.ner.main import (
    check_competitors,
    check_locations,
    check_persons,
)
from src.exceptions import NotInitializedError
from src.shared import Policy, PolicyType, SafetyCode


@pytest.fixture
def mock_ner_model():
    """Create a mock NER model."""
    model = AsyncMock()
    model.predict = AsyncMock()
    model.predict.return_value = ([], 0)
    return model


@pytest.fixture
def competitor_policy():
    """Create a policy for competitor detection."""
    return Policy(
        id=PolicyType.COMPETITOR_CHECK,  # COMPETITOR_CHECK
        name="Competitor Detection",
        is_user_policy=True,
        is_llm_policy=True,
        action=0,
        message="Competitor detected",
        state=True,
        competitors=["CompetitorX", "CompetitorY", "CompetitorZ"],
        threshold=0.5,
        metadata={},
        locations=None,
        persons=None,
        protected_prompts=None,
        max_tokens=None,
    )


@pytest.fixture
def person_policy():
    """Create a policy for person detection."""
    return Policy(
        id=PolicyType.PERSON_CHECK,  # PERSON_CHECK
        name="Person Detection",
        is_user_policy=True,
        is_llm_policy=True,
        action=0,
        message="Specified person detected",
        state=True,
        persons=["John Doe", "Jane Smith", "Bob Johnson"],
        threshold=0.5,
        metadata={},
        competitors=None,
        locations=None,
        protected_prompts=None,
        max_tokens=None,
    )


@pytest.fixture
def location_policy():
    """Create a policy for location detection."""
    return Policy(
        id=PolicyType.LOCATION_CHECK,  # LOCATION_CHECK
        name="Location Detection",
        is_user_policy=True,
        is_llm_policy=True,
        action=0,
        message="Specified location detected",
        state=True,
        locations=["New York", "London", "Tokyo"],
        threshold=0.5,
        metadata={},
        competitors=None,
        persons=None,
        protected_prompts=None,
        max_tokens=None,
    )


class TestNERValidators:
    @patch("src.domain.validators.ner.main.app_state")
    @pytest.mark.asyncio
    async def test_check_competitors_match(
        self, mock_app_state, mock_ner_model, competitor_policy
    ):
        """Test detection of competitors."""
        mock_app_state.ner_model = mock_ner_model
        mock_ner_model.predict.return_value = (
            [{"entity_group": "ORG", "word": "CompetitorX", "score": 0.95}],
            10,
        )
        message = "We should analyze what CompetitorX is doing in the market."
        result = await check_competitors(message, competitor_policy)
        assert result.safety_code == SafetyCode.COMPETITOR_DETECTED
        mock_ner_model.predict.assert_called_once_with(message)

    @patch("src.domain.validators.ner.main.app_state")
    @pytest.mark.asyncio
    async def test_check_persons_match(
        self, mock_app_state, mock_ner_model, person_policy
    ):
        """Test detection of persons."""
        mock_app_state.ner_model = mock_ner_model
        mock_ner_model.predict.return_value = (
            [{"entity_group": "PER", "word": "John Doe", "score": 0.95}],
            10,
        )
        message = "I heard that John Doe is working on a new project."
        result = await check_persons(message, person_policy)
        assert result.safety_code == SafetyCode.PERSON_DETECTED
        mock_ner_model.predict.assert_called_once_with(message)

    @patch("src.domain.validators.ner.main.app_state")
    @pytest.mark.asyncio
    async def test_check_locations_match(
        self, mock_app_state, mock_ner_model, location_policy
    ):
        """Test detection of locations."""
        mock_app_state.ner_model = mock_ner_model
        mock_ner_model.predict.return_value = (
            [{"entity_group": "LOC", "word": "New York", "score": 0.95}],
            10,
        )
        message = "The meeting will be held in New York next week."
        result = await check_locations(message, location_policy)
        assert result.safety_code == SafetyCode.LOCATION_DETECTED
        mock_ner_model.predict.assert_called_once_with(message)

    @patch("src.domain.validators.ner.main.app_state")
    @pytest.mark.asyncio
    async def test_check_competitors_no_match(
        self, mock_app_state, mock_ner_model, competitor_policy
    ):
        """Test when no competitors are detected."""
        mock_app_state.ner_model = mock_ner_model
        mock_ner_model.predict.return_value = (
            [{"entity_group": "ORG", "word": "OtherCompany", "score": 0.95}],
            10,
        )
        message = "We should analyze what OtherCompany is doing in the market."
        result = await check_competitors(message, competitor_policy)
        assert result.safety_code == SafetyCode.SAFE
        mock_ner_model.predict.assert_called_once_with(message)

    @patch("src.domain.validators.ner.main.app_state")
    @pytest.mark.asyncio
    async def test_check_competitors_score_below_threshold(
        self, mock_app_state, mock_ner_model, competitor_policy
    ):
        """Test when competitor is detected but score is below threshold."""
        mock_app_state.ner_model = mock_ner_model
        mock_ner_model.predict.return_value = (
            [{"entity_group": "ORG", "word": "CompetitorX", "score": 0.3}],
            10,
        )
        message = "We should analyze what CompetitorX is doing in the market."
        result = await check_competitors(message, competitor_policy)
        assert result.safety_code == SafetyCode.SAFE
        mock_ner_model.predict.assert_called_once_with(message)

    @patch("src.domain.validators.ner.main.app_state")
    @pytest.mark.asyncio
    async def test_check_competitors_wrong_entity(
        self, mock_app_state, mock_ner_model, competitor_policy
    ):
        """Test when competitor name is detected but with wrong entity type."""
        mock_app_state.ner_model = mock_ner_model
        mock_ner_model.predict.return_value = (
            [{"entity_group": "PER", "word": "CompetitorX", "score": 0.95}],
            10,
        )
        message = "We should talk to CompetitorX about the market."
        result = await check_competitors(message, competitor_policy)
        assert result.safety_code == SafetyCode.SAFE
        mock_ner_model.predict.assert_called_once_with(message)

    @patch("src.domain.validators.ner.main.app_state")
    @pytest.mark.asyncio
    async def test_check_competitors_empty_policy_list(
        self, mock_app_state, mock_ner_model
    ):
        """Test check_competitors returns safe immediately if competitors list is empty."""
        mock_app_state.ner_model = mock_ner_model
        mock_ner_model.predict.reset_mock()
        empty_policy = Policy(
            id=PolicyType.LOCATION_CHECK,
            name="Empty Comp",
            is_user_policy=True,
            is_llm_policy=True,
            action=0,
            message="Comp",
            state=True,
            competitors=[],
            threshold=0.5,
            metadata={},
            locations=None,
            persons=None,
            protected_prompts=None,
            max_tokens=None,
        )
        message = "Any message"
        result = await check_competitors(message, empty_policy)
        assert result.safety_code == SafetyCode.SAFE
        mock_ner_model.predict.assert_not_called()

    @patch("src.domain.validators.ner.main.app_state")
    @pytest.mark.asyncio
    async def test_model_not_initialized(self, mock_app_state, competitor_policy):
        """Test behavior when NER model is not initialized."""
        mock_app_state.ner_model = None
        message = "This is about CompetitorX."
        with pytest.raises(NotInitializedError, match="NER model"):
            await check_competitors(message, competitor_policy)

    @patch("src.domain.validators.ner.main.app_state")
    @pytest.mark.asyncio
    async def test_persons_model_not_initialized(self, mock_app_state, person_policy):
        mock_app_state.ner_model = None
        message = "About John Doe"
        with pytest.raises(NotInitializedError, match="NER model"):
            await check_persons(message, person_policy)

    @patch("src.domain.validators.ner.main.app_state")
    @pytest.mark.asyncio
    async def test_locations_model_not_initialized(
        self, mock_app_state, location_policy
    ):
        mock_app_state.ner_model = None
        message = "About New York"
        with pytest.raises(NotInitializedError, match="NER model"):
            await check_locations(message, location_policy)

    @patch("src.domain.validators.ner.main.app_state")
    @pytest.mark.asyncio
    async def test_check_competitors_uses_precomputed_ner(
        self, mock_app_state, mock_ner_model, competitor_policy
    ):
        """Test check_competitors uses pre-computed ner_results and doesn't call predict."""
        mock_app_state.ner_model = mock_ner_model
        mock_ner_model.predict.reset_mock()
        precomputed_results = [
            {"entity_group": "ORG", "word": "CompetitorY", "score": 0.98}
        ]
        message = "Some message text (content doesn't matter here)"
        result = await check_competitors(
            message, competitor_policy, ner_results=precomputed_results
        )
        assert result.safety_code == SafetyCode.COMPETITOR_DETECTED
        mock_ner_model.predict.assert_not_called()

    @patch("src.domain.validators.ner.main.app_state")
    @pytest.mark.asyncio
    async def test_check_persons_uses_precomputed_ner(
        self, mock_app_state, mock_ner_model, person_policy
    ):
        """Test check_persons uses pre-computed ner_results."""
        mock_app_state.ner_model = mock_ner_model
        mock_ner_model.predict.reset_mock()
        precomputed_results = [
            {"entity_group": "PER", "word": "Jane Smith", "score": 0.88}
        ]
        message = "Irrelevant text"
        result = await check_persons(
            message, person_policy, ner_results=precomputed_results
        )
        assert result.safety_code == SafetyCode.PERSON_DETECTED
        mock_ner_model.predict.assert_not_called()

    @patch("src.domain.validators.ner.main.app_state")
    @pytest.mark.asyncio
    async def test_check_locations_uses_precomputed_ner(
        self, mock_app_state, mock_ner_model, location_policy
    ):
        """Test check_locations uses pre-computed ner_results."""
        mock_app_state.ner_model = mock_ner_model
        mock_ner_model.predict.reset_mock()
        precomputed_results = [{"entity_group": "LOC", "word": "london", "score": 0.77}]
        message = "Irrelevant text"
        result = await check_locations(
            message, location_policy, ner_results=precomputed_results
        )
        assert result.safety_code == SafetyCode.LOCATION_DETECTED
        mock_ner_model.predict.assert_not_called()
