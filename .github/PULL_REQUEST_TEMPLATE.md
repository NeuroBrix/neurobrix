## Description

<!-- What does this PR do? Why is it needed? -->

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Hardware profile
- [ ] Triton kernel
- [ ] Documentation
- [ ] Refactoring (no functional change)
- [ ] Performance improvement
- [ ] Other: <!-- describe -->

## Related Issues

<!-- Link any related issues: Fixes #123, Closes #456 -->

## ZERO Principles Checklist

- [ ] **ZERO HARDCODE** — No hardcoded model-specific values. All values come from the `.nbx` container or hardware profile.
- [ ] **ZERO FALLBACK** — No silent defaults. Missing data causes explicit errors.
- [ ] **ZERO SEMANTIC** — No domain knowledge in the runtime. The engine does not interpret content.

## Code Hygiene Checklist

- [ ] No unused imports, functions, or dead code
- [ ] No commented-out code (use git history)
- [ ] All new functions have docstrings
- [ ] Type hints on all function signatures

## Testing

<!-- How did you test this change? -->

- [ ] Existing tests pass (`python -m pytest tests/`)
- [ ] New tests added for this change
- [ ] Manually validated with a model: <!-- which model? -->

## Screenshots / Benchmarks

<!-- If applicable, add screenshots or performance benchmarks -->
