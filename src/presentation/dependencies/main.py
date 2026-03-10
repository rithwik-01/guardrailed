import logging
from pathlib import Path
from typing import List, Optional

import yaml

from src.core.shutdown import ShutdownManager
from src.core.state import app_state
from src.exceptions import InitializationError
from src.shared import Action, Policy, PolicyType

logger = logging.getLogger(__name__)


def init_shutdown_manager() -> None:
    if app_state.shutdown_manager is None:
        app_state.shutdown_manager = ShutdownManager()


_loaded_policies: Optional[List[Policy]] = None
_policy_file_mtime: Optional[float] = None


async def get_loaded_policies() -> List[Policy]:
    """
    Loads policies from the YAML file specified in the config.
    Reloads if the file has changed. Caches the result in memory.
    """
    global _loaded_policies, _policy_file_mtime
    if app_state.config is None:
        raise InitializationError("Policies", "App config is not initialized")
    policies_path_str = app_state.config.policies_file_path
    policies_path_str = app_state.config.policies_file_path
    policies_path = Path(policies_path_str)

    try:
        if not policies_path.is_file():
            raise FileNotFoundError(f"Policies file not found at: {policies_path_str}")
        current_mtime = policies_path.stat().st_mtime
    except FileNotFoundError as e:
        logger.error(str(e))
        if _loaded_policies is None:
            raise InitializationError("Policies", str(e))
        else:
            logger.warning(
                "Using previously loaded policies as file is now missing or inaccessible."
            )
            assert _loaded_policies is not None
            return _loaded_policies
    except OSError as e:
        logger.error(f"Error accessing policy file {policies_path_str}: {e}")
        if _loaded_policies is None:
            raise InitializationError("Policies", f"Error accessing file: {e}")
        else:
            logger.warning("Using previously loaded policies due to file access error.")
            assert _loaded_policies is not None
            return _loaded_policies

    if (
        _loaded_policies is None
        or _policy_file_mtime is None
        or current_mtime > _policy_file_mtime
    ):
        logger.info(
            f"Loading policies from {policies_path_str} (Reason: {'Initial load' if _loaded_policies is None else 'File changed'})..."
        )
        raw_data = None
        try:
            with open(policies_path_str, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)

            if not isinstance(raw_data, dict):
                raise ValueError(
                    "YAML structure invalid: Expected a top-level 'policies' dictionary key"
                )

            if "policies" not in raw_data:
                raise ValueError(
                    "YAML structure invalid: Expected top-level 'policies' key."
                )

            raw_policies_list = raw_data.get("policies")
            if not isinstance(raw_policies_list, list):
                raise ValueError(
                    "YAML structure invalid: Expected 'policies' key to contain a list"
                )

            loaded_policies = []
            seen_ids = set()
            for i, policy_data in enumerate(raw_policies_list):
                if not isinstance(policy_data, dict):
                    logger.warning(
                        f"Skipping policy at index {i}: Expected a dictionary, got {type(policy_data).__name__}."
                    )
                    continue

                policy_id = policy_data.get("id")
                policy_name = policy_data.get("name")

                if policy_id is None:
                    logger.warning(
                        f"Skipping policy at index {i} (Name: {policy_name or 'N/A'}): Missing 'id' field."
                    )
                    continue
                if not isinstance(policy_id, int):
                    logger.warning(
                        f"Skipping policy at index {i} (Name: {policy_name or 'N/A'}, ID: {policy_id}): 'id' field must be an integer."
                    )
                    continue
                if policy_id in seen_ids:
                    logger.warning(
                        f"Skipping policy at index {i} (Name: {policy_name or 'N/A'}): Duplicate policy 'id' {policy_id} found."
                    )
                    continue

                try:
                    policy_type = PolicyType(policy_id)

                    policy_data.setdefault("state", False)
                    policy_data.setdefault("is_user_policy", True)
                    policy_data.setdefault("is_llm_policy", True)
                    policy_data.setdefault("action", Action.OVERRIDE.value)
                    policy_data.setdefault("name", f"Policy {policy_id}")
                    policy_data.setdefault(
                        "message", f"Policy {policy_type.name} violated."
                    )

                    if policy_type == PolicyType.PII_LEAKAGE:
                        policy_data.setdefault("pii_threshold", 0.5)
                        if (
                            "pii_categories" not in policy_data
                            and "pii_entities" not in policy_data
                        ):
                            policy_data["pii_categories"] = ["DEFAULT"]

                    action_val = policy_data["action"]
                    try:
                        Action(action_val)
                    except ValueError:
                        logger.warning(
                            f"Policy id {policy_id} (Name: {policy_name or 'N/A'}): Invalid 'action' value '{action_val}'. Defaulting to '{Action.OVERRIDE.name}'."
                        )
                        policy_data["action"] = Action.OVERRIDE.value

                    seen_ids.add(policy_id)

                    policy_obj = Policy(**policy_data)
                    loaded_policies.append(policy_obj)

                except ValueError as ve:
                    logger.error(
                        f"Validation error processing policy id {policy_id} (Name: {policy_name or 'N/A'}, index {i}): {ve}",
                        exc_info=False,
                    )
                    continue
                except Exception as e:
                    logger.error(
                        f"Error processing policy id {policy_id} (Name: {policy_name or 'N/A'}, index {i}): Data={policy_data}. Error: {e}",
                        exc_info=False,
                    )
                    continue

            _loaded_policies = loaded_policies
            _policy_file_mtime = current_mtime
            logger.info(
                f"Successfully loaded and validated {len(_loaded_policies)} policies from {policies_path_str}."
            )

        except yaml.YAMLError as e:
            logger.error(
                f"Error parsing YAML file {policies_path_str}: {e}", exc_info=True
            )
            if _loaded_policies is None:
                raise InitializationError("Policies", f"YAML parse error: {e}")
            else:
                logger.warning(
                    "Using previously loaded policies due to YAML parse error."
                )
                assert _loaded_policies is not None
                return _loaded_policies
        except ValueError as e:
            logger.error(
                f"Invalid policy file structure in {policies_path_str}: {e}",
                exc_info=False,
            )
            if _loaded_policies is None:
                raise InitializationError("Policies", f"Invalid structure: {e}")
            else:
                logger.warning(
                    "Using previously loaded policies due to file structure error."
                )
                assert _loaded_policies is not None
                return _loaded_policies
        except Exception as e:
            logger.error(
                f"Failed to load/process policies file {policies_path_str}: {e}",
                exc_info=True,
            )
            if _loaded_policies is None:
                raise InitializationError("Policies", f"Failed to load/process: {e}")
            else:
                logger.warning(
                    "Using previously loaded policies due to file load/process error."
                )
                assert _loaded_policies is not None
                return _loaded_policies

    assert _loaded_policies is not None
    return _loaded_policies
