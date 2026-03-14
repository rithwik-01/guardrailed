# Guardrailed AI Gateway - API Examples

This directory contains practical examples in **Python**, and **JavaScript** for interacting with the Guardrailed AI Gateway. Each example is designed to be small and focused.

## Prerequisites

1.  **Running Gateway:** The Guardrailed AI Gateway must be running. The default URL `http://localhost:8000` is used in these examples.
2.  **API Keys:** For the proxy examples, you must set the corresponding API key as an environment variable:
    ```bash
    export OPENAI_API_KEY="sk-..."
    export GEMINI_API_KEY="..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    ```
3.  **Policy Configuration:** The behavior of these examples depends on your `policies.yaml` file. Before running a script, ensure the relevant policy is enabled (`state: true`) and has the correct `action` set.

## How to Run Examples

### Python (`/python`)

These examples use the official LLM SDKs (like `openai`) and the `requests` library.

**First, install the dependencies:**
```bash
# Navigate to the python examples directory
cd examples/python

# Install required packages
pip install -r requirements.txt
```

**Then, run an example script:**
```bash
# Run the safeguard endpoint examples
python3 safeguard.py

# Run the OpenAI proxy examples
python3 openai_proxy.py
```

### JavaScript (`/javascript`)

These examples use Node.js with `axios` and the official `openai` SDK.

**First, install the dependencies:**
```bash
# Navigate to the javascript examples directory
cd examples/javascript

# Install required packages
npm install
```

**Then, run an example script:**
```bash
# Run the safeguard endpoint examples
node safeguard.js

# Run the OpenAI proxy examples
node openai_proxy.js