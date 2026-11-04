# zfinance

Local-first personal finance app: imports bank CSV exports, categorizes transactions with an
ML model that learns from corrections, and shows a Streamlit dashboard. See README.md.

## Privacy
- Real transactions live only in `data/` (gitignored). Never commit databases, bank exports,
  trained models, `.env` files or notebooks with output.
- Tests and examples use the synthetic files in `sample_data/` (regenerate with
  `python -m zfinance.synth`).

## Git rules
- Commit messages, PR titles, PR descriptions, and code comments must never mention Claude, Anthropic, AI, or any assistant.
- No Co-Authored-By trailers. No "Generated with" footers.
- Use conventional commit style: `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`.
- Make many small commits while building, not one big commit at the end. Target 16-25 changed lines per commit (added + deleted). Hard cap: 100 lines per commit.
- Excluded from the count: generated `sample_data/*.csv` files and `.claude/settings.json`.
- Each commit is one logical unit that leaves the repo working. Don't split a function or file mid-way just to hit the target. Under 16 lines is fine when the unit is naturally small; never pad.
- Never force-push or rewrite published history.
