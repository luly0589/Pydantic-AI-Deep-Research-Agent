# Deep Research Agent

A local Gradio application that researches a free-text question or a ticker-shaped input with DuckDuckGo, then uses Pydantic AI and OpenAI's `gpt-5-mini` to create a structured report with linked evidence.

> **Security note:** Keep `OPENAI_API_KEY` only in `.env`. The file is excluded by `.gitignore`; never commit, share, or paste your key into source code. Rotate a key immediately if it is exposed.

## Features

- Accepts a broad research question (for example, `How will AI data centers affect electricity demand?`) or a ticker-shaped input such as `NVDA`.
- Runs one DuckDuckGo discovery search, then plans **three or four** distinct research angles.
- Runs the angle searches concurrently and attempts to extract readable text from the leading results.
- Uses Pydantic models to validate the research plan and final report shape.
- Renders an executive summary, section-by-section findings, linked evidence, risks/uncertainties, and a watch list.
- Prompts the agent to prefer primary company sources for financial results and guidance when those sources are available.

## Research workflow

```text
Query or ticker
    -> DuckDuckGo discovery search
    -> GPT-5 mini plans 3-4 research angles
    -> concurrent DuckDuckGo searches
    -> page-text extraction where available
    -> GPT-5 mini produces a cited report
```

The application uses DDGS with its default backend selection. A search returns up to five results per angle; the app tries to extract the first two pages per angle and limits extracted page text before synthesis. This keeps the report grounded while controlling request volume and prompt size.

## Requirements

- Python 3.10 or newer
- An OpenAI API key with access to `gpt-5-mini`
- Internet access for OpenAI and DuckDuckGo requests

## Setup

Run these commands in PowerShell from the project folder.

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Add your key to `.env`:

   ```env
   OPENAI_API_KEY=sk-...
   ```

4. Start the application:

   ```powershell
   python app.py
   ```

5. Open the URL printed in the terminal, normally `http://127.0.0.1:7860`.

## Using the app

Enter one topic per run.

| Input | Example | Expected behavior |
| --- | --- | --- |
| Ticker-shaped input | `NVDA` | Treats the input as a ticker candidate and uses discovery results to resolve company and industry context. |
| General research question | `How will AI data centers affect electricity demand?` | Plans non-overlapping angles suited to the topic. |

The report's evidence bullets use clickable citations in the form `S1: Source title`. Citations are created only from the sources gathered during that run. If page extraction fails, the synthesis can still use the DuckDuckGo snippet, but it is prompted to treat that evidence cautiously.

## Project structure

```text
.
- app.py              # Research pipeline, Pydantic models, agents, and Gradio UI
- requirements.txt    # Runtime dependencies
- .env                # Local OpenAI key (not committed)
- .gitignore          # Excludes secrets and local Python files
- docs2.md            # Local Pydantic AI reference material
```

## Configuration and limits

The constants near the top of `app.py` control research breadth:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `MAX_ANGLES` | 4 | Maximum number of planned research sections. |
| `RESULTS_PER_SEARCH` | 5 | Maximum DuckDuckGo results collected per search. |
| `PAGES_PER_ANGLE` | 2 | Leading results per angle selected for text extraction. |
| `MAX_PAGE_CHARS` | 3,500 | Maximum extracted characters retained for each page. |

## Troubleshooting

| Issue | What to check |
| --- | --- |
| "Add your OpenAI API key..." | Set `OPENAI_API_KEY` in `.env`, save the file, and restart the app. |
| The app cannot start | Activate `.venv` and rerun `python -m pip install -r requirements.txt`. |
| Research fails | The app names the failed stage in the browser. Check the terminal that ran `python app.py` for the full traceback, then verify internet access, API-key validity, model access, and the `ddgs` installation. |
| DDGS returns no results | The app retries the default DDGS search three times. Wait a minute, then retry with a more specific query; a search provider may temporarily limit automated requests. |
| A citation has limited detail | Some sites block extraction or publish little readable text. Open the linked source and treat the report as a starting point. |
| A short word is treated as a ticker candidate | Enter a more descriptive question. Ticker detection only identifies the input's format; discovery results provide the actual context. |

## Important limitations

- Reports are generated from live search results and may be incomplete, stale, or contain conflicting claims.
- The app does not validate that a ticker candidate is an active or correct listing.
- The report is for research and education, not financial, legal, medical, or investment advice.
- Respect the terms and rate limits of the services you use.
