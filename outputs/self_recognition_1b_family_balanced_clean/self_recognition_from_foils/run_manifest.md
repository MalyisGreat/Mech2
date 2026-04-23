# Self Recognition From Foils

- Config: `configs\identity_battery\self_recognition_1b_family_balanced_clean.yaml`
- Rows: `720`
- Seeds: `[42, 123, 314, 1618, 2718]`
- Choice mode: `balanced_permutations`
- Identity prompt template: `instruction`
- Identity stop strings: `['\nTask:', '\nResponse:', '\nInstruction:', '\nUser:', '\nAssistant:', '\nSystem:']`
- Purpose: test whether the model can identify which candidate answer is most like its own default answer.
