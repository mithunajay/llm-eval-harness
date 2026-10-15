# Copilot instructions

- Keep the harness dependency-light; runtime code uses the Python standard library.
- Run `python -m pytest -q` from the project root after changing code.
- Preserve offline behavior: mock providers and the opener regression demo must not need API credentials or network access.
- Keep task-specific prompting and checks in `evalkit/tasks.py` and the corresponding check module.
- When changing a dataset, validate and freeze it with the matching schema so its metadata hash and version stay in sync.
- Never add API keys, real credentials, or unverified benchmark results to source files or documentation.
- Keep provider/API failures explicit; do not turn invalid model or judge responses into passing results.
