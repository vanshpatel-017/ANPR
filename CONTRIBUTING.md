# Contributing

## Scope
This repository documents ANPR pipeline engineering and reproducibility contracts.

## Guidelines
- Keep changes focused and reviewable.
- Do not commit private data, labels, or model checkpoints.
- Prefer path indirection via `.env` variables over hard-coded machine-specific paths.
- For notebook edits, preserve execution order and artifact contracts.

## Pull Request checklist
- [ ] No private assets included
- [ ] Notebook outputs are cleaned unless explicitly needed
- [ ] README/docs updated if pipeline contracts changed
- [ ] Paths remain configurable for internal environments
