# AGENTS.md

## Build / Lint / Test
- Install deps: `pip install -r requirements.txt`
- Run app: `python app.py`
- Lint: `ruff check .`
- Format: `ruff format .`
- Run all tests: `pytest`
- Run single test: `pytest tests/test_file.py::test_name`

## Code Style
- Imports: stdlib, third‑party, local, blank line between groups.
- Formatting: 2 spaces, line length 88.
- Types: add type hints, explicit returns.
- Naming: files snake_case.py, funcs/vars snake_case, classes PascalCase, const UPPER_SNAKE.
- Error handling: raise Exceptions, wrap external calls in try/except, log via app.logger.
- Docstrings: one‑line summary.

## Additional Rules
- No .cursor rules.
- No Copilot instructions.
- Follow README for env vars.
