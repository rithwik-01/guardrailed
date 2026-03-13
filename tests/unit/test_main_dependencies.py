import os
from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

import pytest
import yaml

from src.core import app_state
from src.exceptions import InitializationError
from src.presentation.dependencies.main import get_loaded_policies
from src.shared import Action, Policy, PolicyType


@pytest.fixture(autouse=True)
def clear_policy_cache():
    """Ensures the internal policy cache is cleared before each test."""
    with (
        patch("src.presentation.dependencies.main._loaded_policies", None),
        patch("src.presentation.dependencies.main._policy_file_mtime", None),
    ):
        yield


@pytest.fixture
def valid_policies_dict():
    """Provides a valid dictionary representing policy data."""
    return {
        "policies": [
            {
                "id": PolicyType.PII_LEAKAGE.value,
                "name": "Observe PII",
                "state": True,
                "action": Action.OBSERVE.value,
                "message": "PII observed",
                "is_user_policy": True,
                "is_llm_policy": True,
            },
            {
                "id": PolicyType.PROFANITY.value,
                "name": "Block Toxicity",
                "state": True,
                "action": Action.OVERRIDE.value,
                "threshold": 0.75,
                "message": "Toxic content blocked",
                "is_user_policy": True,
                "is_llm_policy": True,
            },
            {
                "id": PolicyType.PROMPT_LEAKAGE.value,
                "name": "Disabled Leak Check",
                "state": False,
                "action": Action.OVERRIDE.value,
                "protected_prompts": ["secret"],
                "message": "Leak check (off)",
                "is_user_policy": False,
                "is_llm_policy": True,
            },
        ]
    }


def create_policy_file(path: Path, data: dict):
    """Writes dictionary data to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


@pytest.mark.asyncio
async def test_load_success_real_file(tmp_path, valid_policies_dict):
    """Test loading policies successfully from a real temporary file."""
    policy_file = tmp_path / "policies_ok.yaml"
    create_policy_file(policy_file, valid_policies_dict)

    with patch.object(app_state.config, "policies_file_path", str(policy_file)):
        policies = await get_loaded_policies()

        assert len(policies) == 3
        assert isinstance(policies[0], Policy)
        pii_policy = next(p for p in policies if p.id == PolicyType.PII_LEAKAGE.value)
        profanity_policy = next(
            p for p in policies if p.id == PolicyType.PROFANITY.value
        )
        leak_policy = next(
            p for p in policies if p.id == PolicyType.PROMPT_LEAKAGE.value
        )

        assert pii_policy.action == Action.OBSERVE.value
        assert pii_policy.state is True
        assert profanity_policy.action == Action.OVERRIDE.value
        assert profanity_policy.threshold == 0.75
        assert leak_policy.state is False


@pytest.mark.asyncio
async def test_caching_no_file_change(tmp_path, valid_policies_dict):
    """Test policies are cached when mtime does not change."""
    dummy_file_path_str = str(tmp_path / "policies_cache_dummy.yaml")

    m_open = mock_open(read_data=yaml.dump(valid_policies_dict))

    with (
        patch.object(app_state.config, "policies_file_path", dummy_file_path_str),
        patch("pathlib.Path.is_file", return_value=True) as mock_is_file,
        patch(
            "pathlib.Path.stat", return_value=MagicMock(st_mtime=1000.0)
        ) as mock_stat,
        patch("builtins.open", m_open) as mock_builtin_open,
    ):
        policies1 = await get_loaded_policies()
        assert len(policies1) == 3
        mock_is_file.assert_called()
        mock_stat.assert_called()
        mock_builtin_open.assert_called_once_with(
            dummy_file_path_str, "r", encoding="utf-8"
        )

        mock_is_file.reset_mock()
        mock_stat.reset_mock()

        policies2 = await get_loaded_policies()
        assert policies1 is policies2
        mock_is_file.assert_called_once()
        mock_stat.assert_called_once()
        assert mock_builtin_open.call_count == 1


@pytest.mark.asyncio
async def test_reload_on_file_change(tmp_path, valid_policies_dict):
    """Test policies are reloaded when mtime changes."""
    dummy_file_path_str = str(tmp_path / "policies_reload_dummy.yaml")

    first_read_content = yaml.dump(valid_policies_dict)
    modified_data = {"policies": [valid_policies_dict["policies"][0]]}
    second_read_content = yaml.dump(modified_data)

    mock_handle1 = mock_open(read_data=first_read_content).return_value
    mock_handle2 = mock_open(read_data=second_read_content).return_value
    open_side_effect = [mock_handle1, mock_handle2]

    mtime_tracker = {"mtime": 1000.0}

    def stat_side_effect(*args, **kwargs):
        current_mtime = mtime_tracker["mtime"]
        mtime_tracker["mtime"] += 100.0
        return MagicMock(st_mtime=current_mtime)

    with (
        patch.object(app_state.config, "policies_file_path", dummy_file_path_str),
        patch("pathlib.Path.is_file", return_value=True) as mock_is_file,
        patch("pathlib.Path.stat", side_effect=stat_side_effect) as mock_stat,
        patch("builtins.open", side_effect=open_side_effect) as mock_builtin_open,
    ):
        policies1 = await get_loaded_policies()
        assert len(policies1) == 3
        mock_is_file.assert_called()
        mock_stat.assert_called()
        mock_builtin_open.assert_called_once_with(
            dummy_file_path_str, "r", encoding="utf-8"
        )

        mock_is_file.reset_mock()
        mock_stat.reset_mock()

        policies2 = await get_loaded_policies()
        assert len(policies2) == 1
        assert policies2[0].id == valid_policies_dict["policies"][0]["id"]
        assert policies1 is not policies2
        mock_is_file.assert_called_once()
        mock_stat.assert_called_once()
        assert mock_builtin_open.call_count == 2
        expected_calls = [
            call(dummy_file_path_str, "r", encoding="utf-8"),
            call(dummy_file_path_str, "r", encoding="utf-8"),
        ]
        mock_builtin_open.assert_has_calls(expected_calls)


@pytest.mark.asyncio
async def test_file_not_found(tmp_path):
    """Test error handling when the policies file doesn't exist."""
    non_existent_file_str = str(tmp_path / "policies_missing.yaml")
    with (
        patch.object(app_state.config, "policies_file_path", non_existent_file_str),
        patch("pathlib.Path.is_file", return_value=False),
    ):
        with pytest.raises(InitializationError) as exc_info:
            await get_loaded_policies()
        assert f"Policies file not found at: {non_existent_file_str}" in str(
            exc_info.value
        )


@pytest.mark.asyncio
async def test_file_becomes_missing_uses_cache(tmp_path, valid_policies_dict):
    """Test uses cache if file exists initially but later disappears."""
    policy_file = tmp_path / "policies_temp.yaml"
    policy_file_str = str(policy_file)
    create_policy_file(policy_file, valid_policies_dict)

    with patch.object(app_state.config, "policies_file_path", policy_file_str):
        policies1 = await get_loaded_policies()
        assert len(policies1) == 3

        os.remove(policy_file)
        assert not policy_file.exists()

        with (
            patch("src.presentation.dependencies.main.logger") as mock_logger,
            patch("pathlib.Path.is_file", return_value=False),
        ):
            policies2 = await get_loaded_policies()

            assert policies1 is policies2
            mock_logger.error.assert_called_once()
            assert (
                f"Policies file not found at: {policy_file_str}"
                in mock_logger.error.call_args[0][0]
            )
            mock_logger.warning.assert_called_once_with(
                "Using previously loaded policies as file is now missing or inaccessible."
            )


@pytest.mark.asyncio
async def test_invalid_yaml_syntax(tmp_path):
    """Test error handling for syntactically invalid YAML."""
    policy_file = tmp_path / "invalid_syntax.yaml"
    policy_file_str = str(policy_file)
    policy_file.write_text("policies: [\n  id: 1\n name: oops", encoding="utf-8")
    with patch.object(app_state.config, "policies_file_path", policy_file_str):
        with pytest.raises(InitializationError) as exc_info:
            await get_loaded_policies()
        assert "YAML parse error:" in str(exc_info.value)
        assert "expected ',' or ']'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_invalid_yaml_structure_no_policies_key(tmp_path):
    """Test error handling for incorrect YAML structure (missing 'policies' key)."""
    policy_file = tmp_path / "invalid_structure_1.yaml"
    policy_file_str = str(policy_file)
    create_policy_file(policy_file, [{"id": 1}])
    with patch.object(app_state.config, "policies_file_path", policy_file_str):
        with pytest.raises(InitializationError) as exc_info:
            await get_loaded_policies()
        assert "Invalid structure:" in str(exc_info.value)
        assert (
            "YAML structure invalid: Expected a top-level 'policies' dictionary key"
            in str(exc_info.value)
        )


@pytest.mark.asyncio
async def test_invalid_yaml_structure_policies_not_list(tmp_path):
    """Test error handling for incorrect YAML structure ('policies' is not a list)."""
    policy_file = tmp_path / "invalid_structure_2.yaml"
    policy_file_str = str(policy_file)
    create_policy_file(policy_file, {"policies": {"id": 1}})
    with patch.object(app_state.config, "policies_file_path", policy_file_str):
        with pytest.raises(InitializationError) as exc_info:
            await get_loaded_policies()
        assert "Invalid structure:" in str(exc_info.value)
        assert (
            "YAML structure invalid: Expected 'policies' key to contain a list"
            in str(exc_info.value)
        )


@pytest.mark.asyncio
async def test_skips_invalid_policy_data(tmp_path, valid_policies_dict):
    """Test that individual policies with invalid data are skipped or corrected."""
    invalid_entries_dict = {
        "policies": [
            valid_policies_dict["policies"][0],
            {"id": "wrong_type", "name": "Bad ID"},
            {
                "id": 12,
                "name": "Bad Action",
                "action": 999,
            },
            valid_policies_dict["policies"][1],
            "just a string",
            {"name": "Missing ID"},
        ]
    }
    policy_file = tmp_path / "partial_invalid.yaml"
    policy_file_str = str(policy_file)
    create_policy_file(policy_file, invalid_entries_dict)

    with (patch.object(app_state.config, "policies_file_path", policy_file_str),):
        policies = await get_loaded_policies()

        assert len(policies) == 2
        assert policies[0].id == PolicyType.PII_LEAKAGE.value
        assert policies[0].action == Action.OBSERVE.value

        assert policies[1].id == PolicyType.PROFANITY.value
        assert policies[1].name == "Block Toxicity"
        assert policies[1].action == Action.OVERRIDE.value
        assert policies[1].state is True
