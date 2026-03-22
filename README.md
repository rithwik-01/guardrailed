![Guardrailed](readme_assets/guardrailed-logo.svg)

Guardrailed is an open-source AI gateway that provides guardrails for LLM applications, intercepting API calls to validate and sanitize requests and responses in real-time.

## How it works

Guardrailed acts as a self-hosted proxy between your application and LLM providers like OpenAI, Google Gemini, and Anthropic Claude. It applies custom policies locally to check content for prompt injections, toxic language, data leaks, and PII before forwarding requests to upstream models or returning responses to your application.

## Why use Guardrailed

Guardrailed defends against common vulnerabilities like prompt injections and insecure outputs, providing a critical security layer for your AI applications. The policy engine automatically detects and redacts personally identifiable information before it leaves your network, helping you maintain compliance with data privacy regulations. Content filtering capabilities block toxic language, profanity, and other undesirable content to protect your users and brand. Guardrailed serves as a unified proxy for multiple LLM providers, letting you swap models without changing your application's safety logic.

## Use cases

Public-facing chatbots use Guardrailed to prevent users from submitting sensitive data and ensure LLM responses remain appropriate and on-brand. Internal AI tools protect against accidental leakage of secrets, API keys, or confidential project information in prompts or responses. Regulated industries in healthcare and finance deploy Guardrailed as a foundation for HIPAA or GDPR compliance through PII redaction and audit logging. Content generation platforms integrate Guardrailed to ensure AI-generated content meets community guidelines and safety standards.

## Quick start

Prerequisites: Docker and Docker Compose.

1. Clone the repository:
   ```bash
   git clone https://github.com/rithwik-01/guardrailed.git
   cd guardrailed
   ```

2. Prepare your environment:
   ```bash
   cp .env.example .env
   ```
   The example policies.yaml and docker-compose.yml are ready to use.

3. Launch the gateway:
   ```bash
   docker-compose up -d
   ```
   The first launch downloads machine learning models to the guardrailed_hf_cache volume. Monitor startup progress with `docker-compose logs -f`.

4. Test a guardrail:
   ```bash
   curl -s -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gpt-3.5-turbo",
       "messages": [{"role": "user", "content": "My email address is test@example.com"}]
     }'
   ```
   The response shows `finish_reason: "content_filter"` when the PII policy blocks the request.

## API usage

### OpenAI proxy

Configure your OpenAI client to point to Guardrailed.

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="http://localhost:8000/v1"
)
```

Inspect the `X-Guardrailed-Blocked: true` header or check for `finish_reason: "content_filter"` to reliably detect blocked requests.

### Gemini and Claude proxies

Guardrailed provides drop-in proxy capability for Google Gemini and Anthropic Claude.

Gemini configuration:
```python
import google.generativeai as genai
genai.configure(
    api_key=os.environ.get("GEMINI_API_KEY"),
    client_options={"api_endpoint": "http://localhost:8000"},
    transport='rest'
)
```

Claude configuration:
```python
from anthropic import Anthropic
client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    base_url="http://localhost:8000/anthropic"
)
```

### Direct validation

Use the `/safeguard` endpoint to validate content without proxying to an LLM.

```bash
curl -s -X POST "http://localhost:8000/safeguard" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "My email is example@gmail.com."}
    ]
  }'
```

Response:
```json
{
  "safety_code": 10,
  "message": "Potentially sensitive information detected.",
  "action": 0
}
```

## Policy configuration

Guardrailed loads all policy definitions from `policies.yaml` at startup. See the full policy configuration reference in `docs/POLICIES.md`.
