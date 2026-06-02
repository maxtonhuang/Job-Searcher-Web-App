# Singapore Job Matcher

A chatbot-style assistant that reads your resume, searches Singapore job portals,
and returns ranked openings with links, salary, and a short fit explanation.

Sources: MyCareersFuture (public JSON API), Adzuna and Jooble (free developer
APIs), and JSearch (RapidAPI, limited free quota).

## File guide

| File | Role |
|---|---|
| `parse.py` | Read resume from PDF or pasted text. No LLM. |
| `sources.py` | Fetch + normalize jobs from each portal. No LLM. |
| `prompts.py` | ICCO system prompts (resume extraction, ranking, chat). |
| `analyzer.py` | The functions that call the LLM. |
| `main.py` | `run_search` pipeline + a CLI for testing. |
| `app.py` | Streamlit UI. |
| `llm.py` | LiteLLM wrapper (`ask_json` / `ask_text`). |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

copy .env.example .env           # then edit .env
```

In `.env` set `OPENAI_API_KEY`, `ADZUNA_APP_ID`, and `ADZUNA_APP_KEY`.
The model defaults to `openai/gpt-4.1-mini`.

## Run

Test the pipeline from the command line (no UI):

```bash
python main.py --resume my_resume.pdf --role "software engineer" --salary-min 5000
```

Run the app:

```bash
streamlit run app.py
```

## Notes

- Salaries are normalized to monthly SGD. MyCareersFuture is monthly already;
  Adzuna is annual and is divided by 12 (and flagged "estimated" when predicted).
- The salary range is a soft preference: out-of-range jobs rank lower but are kept.
- MyCareersFuture search results have no description, so the full text is now
  fetched up front for each MCF job (one extra request per job) so the ranker can
  read it to infer level and years of experience.
- Each job is sorted into a level bucket (Intern, Junior, Mid, Senior, Lead/Manager,
  Director/Exec, or Unknown) with the signal used shown on the card, and the sidebar
  checkboxes filter the visible jobs by level.
- The chat box works on the jobs already on screen: it can filter, re-order, or
  answer questions. It does not run a fresh portal search.
