# Stop model-free training at the first complete fixed-validation success

The user selected: first complete 50/50, save that policy, stop new updates, then run the existing final checkpoint evaluation. This is a goal-completion rule, independent of minimum training budget and plateau patience. It applies to the existing enabled model-free fixed validation, including P0. Other modes and evaluation protocols remain unchanged.

The alternatives discussed were waiting for multiple new-policy validations or exhausting the training budget. Both spend additional training after the finite success metric has saturated; the selected rule instead preserves the exact validated policy and uses the existing frozen post-training assessment for further measurement.

Only a complete, correct-version, protocol-consistent 50-case result can trigger `fixed_validation_target_reached`. Collection stays paused, no new training cohort is admitted, and the normal final snapshot must have the same policy step as the full-success result. The automatic-completion validator and both supervisors must accept and verify this reason. Faults and manual stops keep their existing precedence. The plateau and budget conditions remain fallbacks.

The currently running old-code experiment cannot acquire new Python control behavior by updating the development checkout. It will use its existing authorized graceful-stop path and save its current version for final evaluation; historical first-success weights must not be fabricated or claimed recovered. The delayed successor must be rebound to the verified new GitHub commit without changing the active runtime checkout, and must still wait for successful parent cleanup/evaluation.

Training and validation share 50 initial states. The stopping event records finite-set success, not held-out generalization or a proven population success probability of one. Existing overwrite-only checkpoint semantics are retained, with a mandatory final save at the exact stopping version.
