# Detector Evaluation

Guardrailed's thresholds and model choices are set from measurements, not defaults.
This document records how the prompt injection detector was evaluated and what the
numbers were.

## Running the eval

```bash
python -m evals.injection_eval
python -m evals.injection_eval --model protectai/deberta-v3-base-prompt-injection-v2
python -m evals.injection_eval --thresholds 0.5,0.9 --json results.json
```

The corpora live in `evals/data/`:

| File | Contents |
|---|---|
| `injection_attacks.jsonl` | 30 injection attempts across six families: direct override, persona/jailbreak, delimiter spoofing, prompt extraction, task hijack, indirect (document-borne) injection. |
| `injection_benign.jsonl` | 30 legitimate prompts that contain injection trigger vocabulary ("ignore", "system prompt", "override", "developer mode", "API keys"). |

The benign set follows the design of the **NotInject** benchmark
([arXiv:2410.22770](https://arxiv.org/abs/2410.22770)), which showed guardrail models
dropping to near-random accuracy on benign text that merely *mentions* injection
vocabulary. A guardrail that blocks "how do I ignore whitespace in git diff?" is not
a working guardrail.

The eval also scores each attack rewritten with zero-width characters and with
Cyrillic homoglyphs, both raw and after the gateway's sanitization step, following
the character-injection attack classes in
[arXiv:2504.11168](https://arxiv.org/pdf/2504.11168).

## Results (2026-08-17, 30 attacks / 30 benign)

Detection rate on attacks, and false-positive rate on benign trigger-word prompts:

| Model | Threshold | Attacks detected | Benign false positives |
|---|---|---|---|
| `deepset/deberta-v3-base-injection` | 0.5 | 100% | **30.0%** |
| `deepset/deberta-v3-base-injection` | 0.75 | 100% | 26.7% |
| `deepset/deberta-v3-base-injection` | 0.9 | 100% | 23.3% |
| `deepset/deberta-v3-base-injection` | 0.95 | 100% | 23.3% |
| `protectai/deberta-v3-base-prompt-injection-v2` | 0.5 | 100% | **13.3%** |
| `protectai/deberta-v3-base-prompt-injection-v2` | 0.95 | 100% | 13.3% |

### Decisions taken from this

1. **Default model switched** to `protectai/deberta-v3-base-prompt-injection-v2`.
   Same detection on this corpus, less than half the false-positive rate.
   Override with `INJECTION_MODEL_URL`.

2. **Threshold tuning does not fix over-defense on either model.** Both are highly
   confident: scores sit near 0 or 1, so raising the threshold from 0.5 to 0.95
   barely moves the false-positive rate. Do not expect to tune your way out of this.

3. **Run the injection policy as `action: 1` (OBSERVE) first.** At a ~13% false
   positive rate on developer-flavoured benign traffic, blocking on day one will
   block real users. Observe, review the logged violations against your own traffic,
   then switch to OVERRIDE if the rate is acceptable for your workload.

### On the evasion columns

Both models flagged the zero-width and homoglyph variants of these attacks at 100%
even *without* sanitization, so this corpus does not reproduce the detection collapse
reported in arXiv:2504.11168. The likely reason is visible in the false-positive
column: these models fire readily on anything unusual, and mangled text is unusual.

Sanitization is still load-bearing, just not for this model on this corpus. The
deterministic validators are trivially evaded without it, which is pinned by
`tests/unit/test_sanitization.py::TestEvasionRegression`:

- Prompt leakage, protected string `Internal Secret Codeword: Alpha`, with a
  zero-width space between every character: **not detected** raw, **detected** after
  sanitization.
- Prompt leakage, protected string `API_KEY_XYZ`, written with Cyrillic lookalikes:
  **not detected** raw, **detected** after sanitization.

The same applies to Presidio's entity regexes, which do not match across injected
zero-width characters.

## Limitations

- 30 samples per class is enough to catch a 2x difference in false-positive rate,
  not enough to distinguish 12% from 14%. Treat these as directional.
- The corpora are hand-written English. Non-English injection is not covered, and
  the ProtectAI model card notes the model does not handle non-English prompts.
- Attack coverage is prompt-level. Multi-turn and tool-output-borne injection are
  not represented.
- Extend the corpora with traffic from your own deployment before trusting a
  threshold in production.
