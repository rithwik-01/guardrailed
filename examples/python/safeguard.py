import json

import requests

GUARDRAILED_GATEWAY_URL = "http://localhost:8000"


def run_safeguard_example():
    """Demonstrates using the /safeguard endpoint for direct validation."""
    print("Running safeguard examples")

    # 1. Safe Request
    print("Testing a safe message...")
    safe_payload = {
        "messages": [{"role": "user", "content": "This is a perfectly safe message."}]
    }
    response = requests.post(f"{GUARDRAILED_GATEWAY_URL}/safeguard", json=safe_payload)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:", json.dumps(response.json(), indent=2))
    print("--> Expected: safety_code: 0 (SAFE)")

    # 2. Block PII Request
    # Requires PII policy to be active with action: 0 (OVERRIDE)
    print("Testing a message with PII to be BLOCKED...")
    pii_payload = {
        "messages": [{"role": "user", "content": "My email is email@test.com."}]
    }
    response = requests.post(f"{GUARDRAILED_GATEWAY_URL}/safeguard", json=pii_payload)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:", json.dumps(response.json(), indent=2))
    print("--> Expected: safety_code: 20 (PII_DETECTED), action: 0 (OVERRIDE)")


if __name__ == "__main__":
    run_safeguard_example()
