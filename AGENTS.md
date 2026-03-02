# AGENTS.md

## Experimental Strategy

The vision and strategy is set out in the EXPERIMENTAL_STRATEGY.md - do not make changes to this without express permission

## Project rules

- Use `uv` for Python dependencies and running commands.
  - Prefer: `uv sync`, `uv run ...`
  - Do not use: `pip`, `poetry`, `conda`

- Do not transform or rewrite data already in `data/`.
  - No reformatting, renaming, moving, or regenerating files under `data/` unless explicitly instructed.

- Do not modify anything in `notebooks/`.
  - Treat notebooks as read-only reference material.

- In general, do not build anything without going through the "planning stage" first and getting explicit permission.

- Break tasks down into smaller deliverables where possible, and prioritise simplicity and transparency.

- Try not to use jargon for the sake of it - make it easy for me to understand. If you ask a question and give me options, you should explain the pros and cons of each option. You should give me an option to ask for further clarification, since I might not understand.
