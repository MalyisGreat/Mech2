# Self Prediction Calibration

- Config: `configs\identity_battery\self_prediction_clean_smoke.yaml`
- Rows: `16`
- Label-bias correction: `True`
- Identity prompt template: `instruction`
- Identity stop strings: `['\nTask:', '\nResponse:', '\nInstruction:', '\nUser:', '\nAssistant:', '\nSystem:']`
- Purpose: test whether the model can predict, prompt by prompt, how it itself is about to answer.
