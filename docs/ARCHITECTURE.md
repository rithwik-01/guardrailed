# Architecture

## Overview

Guardrailed is a FastAPI service that intercepts requests between client applications and LLM providers, applying policy-based validation to both user inputs and model outputs. The service operates in two modes: as a transparent proxy for OpenAI, Gemini, and Anthropic API calls, or as a direct validation endpoint for custom workflows. All validation logic executes locally within your infrastructure using machine learning models for PII detection, NER-based entity recognition, toxicity classification, and fuzzy matching for secret detection.

## Request flow

```mermaid
graph TD
    A[Client Request] --> B[FastAPI Router]
    B --> C{Route Type}
    C -->|/v1/*| D[OpenAI Proxy Handler]
    C -->|/v1beta/*| E[Gemini Proxy Handler]
    C -->|/anthropic/*| F[Claude Proxy Handler]
    C -->|/safeguard| G[Direct Validation Handler]

    D --> H[Extract Messages]
    E --> H
    F --> H
    G --> H

    H --> I[ContentValidator.validate_content - INPUT STAGE]

    I --> J[Load Active Policies]
    J --> K{Input Checks Enabled?}

    K -->|Yes| L[PII Leakage Check - PresidioAnalyzer]
    L --> M[PII Leakage Result]

    K -->|Yes| N[Prompt Leakage Check - RapidFuzz]
    N --> O[Prompt Leakage Result]

    K -->|Yes| P[NER Competitor Check]
    P --> Q[NER Result]

    K -->|Yes| R[NER Person Check]
    R --> S[NER Result]

    K -->|Yes| T[NER Location Check]
    T --> U[NER Result]

    K -->|Yes| V[Toxicity Check - ClassificationModel]
    V --> W[Toxicity Result]

    M --> X{Any Violation?}
    O --> X
    Q --> X
    S --> X
    U --> X
    W --> X

    X -->|Block| Y[Return Blocked Response]
    Y --> Z[Add X-Guardrailed Headers]
    Z --> AA[Return to Client]

    X -->|Pass| AB{Proxy Mode?}
    AB -->|Yes| AC[Forward to Upstream LLM]
    AC --> AD[httpx.AsyncClient POST]
    AD --> AE[Receive LLM Response]

    AE --> AF[ContentValidator.validate_content - OUTPUT STAGE]
    AF --> AG[Load Active Policies]
    AG --> AH{Output Checks Enabled?}

    AH -->|Yes| AI[Run Output Validators]
    AI --> AJ{Any Violation?}

    AJ -->|Block| Y
    AJ -->|Pass| AK[Return LLM Response]
    AK --> AA

    AB -->|No| AK
```

## Component map

```mermaid
graph LR
    subgraph FastAPI_Application
        A[src/main.py]
        B[openai_proxy.py]
        B2[claude_proxy.py]
        B3[gemini_proxy.py]
        C[safeguard.py]
        D[health.py]
    end

    subgraph Policy_Layer
        E[get_loaded_policies]
        F[policies.yaml]
    end

    subgraph Validation_Engine
        G[ContentValidator]
        H[ValidationContext]
    end

    subgraph Validators
        I[PII Leakage - PresidioAnalyzer]
        J[Prompt Leakage - RapidFuzz]
        K[NER Competitor - NERModel]
        L[NER Person - NERModel]
        M[NER Location - NERModel]
        N[Toxicity - ClassificationModel]
    end

    subgraph ML_Models
        O[Presidio Engines]
        P[ClassificationModel - s-nlp/roberta_toxicity_classifier]
        Q[NERModel - dslim/bert-base-NER]
    end

    subgraph LLM_Clients
        R[httpx.AsyncClient - OpenAI]
        S[httpx.AsyncClient - Gemini]
        T[httpx.AsyncClient - Anthropic]
    end

    subgraph Configuration
        U[AppConfig]
        V[AppState]
    end

    A --> E
    A --> B
    A --> B2
    A --> B3
    A --> C
    A --> D

    B --> G
    B2 --> G
    B3 --> G
    C --> G

    E --> F
    G --> H
    H --> E

    G --> I
    G --> J
    G --> K
    G --> L
    G --> M
    G --> N

    I --> O
    N --> P
    K --> Q
    L --> Q
    M --> Q

    B --> R
    B2 --> T
    B3 --> S

    A --> U
    A --> V
    E --> V
```

## Endpoint reference

| Method | Path | Description | Upstream |
|--------|------|-------------|----------|
| GET | /health | Health check endpoint returns service status | None |
| POST | /safeguard | Direct content validation endpoint | None |
| POST | /v1/chat/completions | OpenAI chat completions proxy | OpenAI |
| POST | /v1beta/models/{model}:generateContent | Gemini content generation proxy | Gemini |
| POST | /anthropic/v1/messages | Anthropic Claude messages proxy | Anthropic |

## Policy engine internals

The policy engine filters and applies validation rules through ContentValidator in the domain layer. When processing a message, the engine calls `_get_active_policies_for_role()` which filters the loaded policy list by the `is_user_policy` flag for input validation and `is_llm_policy` flag for output validation. Active policies are grouped by their `id` field into PolicyType enumerations, then dispatched to corresponding validator functions.

Each policy type invokes a specific validator with the policy's configured threshold and message context. PII leakage checks use Presidio's analyzer with spaCy and Transformers backends to detect entities like email addresses and credit card numbers. Prompt leakage uses RapidFuzz for fuzzy string matching against protected keyword lists. The three NER-based checks (competitors, persons, locations) share a single NERModel wrapper around Hugging Face's bert-base-NER, while the toxicity check uses a separate ClassificationModel wrapper around the roberta_toxicity_classifier.

When a violation is detected, the engine returns a Status object containing a SafetyCode, action type, and message. The OVERRIDE action (action: 0) causes the router to return a blocked response with X-Guardrailed headers indicating the safety code and action taken. The OBSERVE action (action: 1) logs the violation but allows the request to proceed unchanged. The REDACT action (action: 2) modifies the content to remove detected violations, though this is currently only supported in the direct validation endpoint. For all violation types, the engine returns consistent Status and SafetyCode objects that routers use to construct appropriate HTTP responses.
