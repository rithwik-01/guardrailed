import os

from openai import OpenAI

# Make sure to set your OpenAI API key as an environment variable
# export OPENAI_API_KEY="sk-..."
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Point the OpenAI client to your Guardrailed Gateway instance
client = OpenAI(api_key=OPENAI_API_KEY, base_url="http://localhost:8000/v1")


def run_openai_proxy_examples():
    """Demonstrates using the Guardrailed Gateway as an OpenAI proxy."""
    print("Running OpenAI Proxy examples")

    # 1. Safe Request
    print("\n Testing a safe request that should pass...")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=64,
        )
        print("Response:", response.choices[0].message.content.strip())
        print("--> Expected: A valid response from OpenAI (e.g., 'Paris').")
    except Exception as e:
        print(f"An error occurred: {e}")

    # 2. Blocked Request
    # Requires PII policy to be active with action: 0 (OVERRIDE)
    print("\n Testing a request with PII to be BLOCKED...")
    try:
        # Using with_raw_response to inspect headers for the block
        response_with_raw = client.with_raw_response.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "My email is test@example.com"}],
        )
        completion = response_with_raw.parse()  # Get the parsed Pydantic model

        print("Blocked Message:", completion.choices[0].message.content)
        print("Finish Reason:", completion.choices[0].finish_reason)

        # Check for the custom header
        is_blocked = response_with_raw.headers.get("X-Guardrailed-Blocked")
        safety_code = response_with_raw.headers.get("X-Guardrailed-Safety-Code")

        print(f"Header 'X-Guardrailed-Blocked': {is_blocked}")
        print(f"Header 'X-Guardrailed-Safety-Code': {safety_code}")
        print(
            "--> Expected: finish_reason: 'content_filter' and X-Guardrailed-Blocked: 'true'"
        )

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    run_openai_proxy_examples()
