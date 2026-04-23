# Self Recognition Near Foil

- Config: `C:\Users\joshj\joseph-stroud-identity-stability-research\configs\identity_battery\self_recognition_nearfoil_qwen_v3_smoke.yaml`
- Rows: `72`
- Valid rows: `72`
- Identity prompt template: `tokenizer_chat`
- Identity stop strings: `None`
- Duplicate semantic overlap threshold: `0.999`
- Duplicate style distance threshold: `1e-06`
- Max sentence repetition rate: `0.7`
- Generation do sample: `True`
- Generation temperature/top_p/top_k: `0.7` / `0.8` / `20`
- Generation presence penalty: `1.5`
- Purpose: estimate an ownership curve from near foils to farther foils instead of a single 3-way oddball choice.
