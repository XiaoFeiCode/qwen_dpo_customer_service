# Preference Construction

The project uses weak-supervision preference construction for scalable DPO data generation.

## Chosen

The original customer service response is treated as `chosen` when it passes basic filters:

- non-empty assistant response
- minimum response length
- preceding user context exists

This assumes human customer service replies are usually better aligned with platform workflow than synthetic negative responses.

## Rejected

Rejected responses are generated with controlled corruption strategies:

1. `terse`: overly short response with missing process guidance
2. `vague`: pushes the user away without service ownership
3. `overpromise`: unsupported refund or compensation commitment
4. `privacy_leak`: unsafe handling of private information

The generated negative samples are intentionally simple and auditable. They are meant to teach preference boundaries, not to simulate all possible bad replies.

## Filtering

Generated pairs can be filtered by:

- category
- response length
- refusal boundary flag
- keyword coverage
- rule-based chosen/rejected quality score

For large-scale training, a manual review slice should be sampled from each category before launching full DPO training.
