# Self Other Boundary

- Config: `configs\identity_battery\self_other_boundary_confirm_clean.yaml`
- Smoke mode: `False`
- Rows: `72`
- Seeds: `[11, 17, 23]`
- Identity prompt template: `instruction`
- Identity stop strings: `['\nTask:', '\nResponse:', '\nInstruction:', '\nUser:', '\nAssistant:', '\nSystem:']`
- Purpose: test whether the model keeps a discriminable self/other answer boundary under identity framing and whether contrary steering collapses that boundary.
