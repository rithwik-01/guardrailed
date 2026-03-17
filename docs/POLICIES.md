# Policy configuration

Guardrailed loads all policy definitions from `policies.yaml` at startup to control content validation behavior.

## Overview

A policy is a rule that Guardrailed evaluates for each message passing through the gateway. Policies define which validation checks to perform, what conditions trigger a violation, which action to take when a violation occurs, and whether the policy applies to user input, LLM output, or both. Each policy specifies a check type, a trigger condition, an action to take on violation, and a scope (input, output, or both).

## Policy fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| id | integer | Yes | - | Numeric identifier mapping to a specific validator |
| name | string | Yes | - | Human-readable name used in logs |
| state | boolean | Yes | - | Enables or disables the policy |
| is_user_policy | boolean | No | true | Apply policy to user prompts sent to the LLM |
| is_llm_policy | boolean | No | true | Apply policy to LLM responses returned to the user |
| action | integer | No | 0 | Action to take on violation (0=OVERRIDE, 1=OBSERVE, 2=REDACT) |
| message | string | No | - | Message returned in response body on violation |
| threshold | float | No | - | Confidence threshold for model-based policies (0.0-1.0) |
| metadata | dict | No | {} | Optional annotations for custom metadata |

## Policy-specific fields

### PII leakage (id: 1)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| pii_categories | list[string] | - | High-level Presidio categories to scan |
| pii_entities | list[string] | - | Specific Presidio entity types to detect |
| pii_threshold | float | 0.5 | Minimum confidence score for PII detection |

### Prompt leakage (id: 2)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| protected_prompts | list[string] | - | Sensitive strings or keywords to detect |
| prompt_leakage_threshold | float | 0.85 | Similarity score for fuzzy matching (0.0-1.0) |

### NER checks (id: 3, 4, 5)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| competitors | list[string] | - | Organization names for competitor check (id: 3) |
| persons | list[string] | - | Person names for person check (id: 4) |
| locations | list[string] | - | Location names for location check (id: 5) |
| threshold | float | - | Confidence threshold for NER detection |

### Toxicity (id: 6)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| threshold | float | - | Confidence cutoff for profanity classification |

## Reference tables

### Policy types

| ID | Name | Description | Key fields |
|----|------|-------------|------------|
| 1 | PII_LEAKAGE | Detects PII using Presidio | pii_entities, pii_threshold |
| 2 | PROMPT_LEAKAGE | Detects secrets via fuzzy matching | protected_prompts, prompt_leakage_threshold |
| 3 | COMPETITOR_CHECK | Detects competitor mentions via NER | competitors, threshold |
| 4 | PERSON_CHECK | Detects person mentions via NER | persons, threshold |
| 5 | LOCATION_CHECK | Detects location mentions via NER | locations, threshold |
| 6 | PROFANITY | Detects toxic language via classification | threshold |

### Actions

| Value | Name | Description |
|-------|------|-------------|
| 0 | OVERRIDE | Block the interaction and return a modified response body |
| 1 | OBSERVE | Log the violation but allow the interaction to proceed |
| 2 | REDACT | Modify content to remove detected violations |
| 3 | RETRY | Suggest a retry for internal errors |

### Safety codes

Guardrailed returns a safety code in the `X-Guardrailed-Safety-Code` response header and logs when a policy is violated.

| Code | Name | Policy ID |
|-------|------|-----------|
| 0 | SAFE | N/A |
| 10 | PII_DETECTED | 1 |
| 20 | PROMPT_LEAKED | 2 |
| 30 | COMPETITOR_DETECTED | 3 |
| 40 | PERSON_DETECTED | 4 |
| 50 | LOCATION_DETECTED | 5 |
| 60 | PROFANITY | 6 |
| -10 | GENERIC_UNSAFE | N/A |
| -70 | UNEXPECTED | N/A |
| -80 | TIMEOUT | N/A |

## Examples

### Block PII in user prompts

This policy blocks requests containing email addresses or U.S. Social Security Numbers with high confidence, checking only user input.

```yaml
- id: 1
  name: "Block PII from Users"
  state: true
  is_user_policy: true
  is_llm_policy: false
  action: 0
  message: "Your request was blocked because it contained sensitive information (PII)."
  pii_entities:
    - EMAIL_ADDRESS
    - US_SSN
  pii_threshold: 0.85
```

### Observe toxicity in LLM responses

This policy logs warnings when toxic language is detected in LLM responses but does not block the response.

```yaml
- id: 6
  name: "Monitor LLM Toxicity"
  state: true
  is_user_policy: false
  is_llm_policy: true
  action: 1
  message: "Potentially toxic content detected in LLM response."
  threshold: 0.75
```

### Prevent internal name leaks

This policy prevents the LLM from mentioning internal project codenames in responses.

```yaml
- id: 2
  name: "Prevent Internal Codename Leaks"
  state: true
  is_user_policy: false
  is_llm_policy: true
  action: 0
  message: "This response was blocked as it contained confidential information."
  protected_prompts:
    - "Project Titan"
    - "Bluebird Initiative"
    - "Internal-API-Key-v2"
  prompt_leakage_threshold: 0.90
```

## Full example

This example demonstrates a complete `policies.yaml` file with active and inactive policies.

```yaml
policies:
  # PII & Data Leakage Policies
  - id: 1
    name: "Block High-Confidence PII (Email, Phone, SSN)"
    state: true
    is_user_policy: true
    is_llm_policy: true
    action: 0
    message: "Interaction blocked due to sensitive data (PII)."
    pii_entities:
      - EMAIL_ADDRESS
      - PHONE_NUMBER
      - US_SSN
    pii_threshold: 0.75

  - id: 2
    name: "Prevent Secret Key Leakage from LLM"
    state: true
    is_user_policy: false
    is_llm_policy: true
    action: 0
    message: "Response blocked for security reasons."
    protected_prompts:
      - "sk-internal-..."
      - "Project-QuantumLeap"
    prompt_leakage_threshold: 0.95

  # Content & Behavior Policies
  - id: 6
    name: "Observe Profanity"
    state: true
    is_user_policy: true
    is_llm_policy: true
    action: 1
    message: "Profane language was detected."
    threshold: 0.8

  - id: 3
    name: "Block Competitor Mentions by LLM"
    state: true
    is_user_policy: false
    is_llm_policy: true
    action: 0
    message: "This response was modified to remove competitor names."
    competitors:
      - "Acme Corporation"
      - "Global-Tech Inc."
    threshold: 0.8

  # Inactive Policy
  - id: 4
    name: "Monitor for Executive Mentions (Inactive)"
    state: false
    is_user_policy: true
    is_llm_policy: true
    action: 1
    persons:
      - "Jane Doe"
      - "John Smith"
    threshold: 0.9
```

## Best practices

### Start in observe mode

Enable new policies with `action: 1` (OBSERVE) to monitor what they flag without disrupting your application. Review logs to verify accuracy and tune thresholds before switching to `action: 0` (OVERRIDE) to enforce blocking.

### Use version control

Store `policies.yaml` in git to track changes, review updates, and roll back if needed. The policy file is application logic that deserves the same version control practices as your code.

### Be specific

Define precise policies with high thresholds rather than broad rules with low thresholds. For example, block specific `pii_entities` like `EMAIL_ADDRESS` and `CREDIT_CARD` instead of using the generic `DEFAULT` category to reduce false positives.

### Tailor input and output scope

Configure `is_user_policy` and `is_llm_policy` independently based on your threat model. You might block PII from users with `is_user_policy: true` while allowing the LLM to return contact information with `is_llm_policy: false`.
