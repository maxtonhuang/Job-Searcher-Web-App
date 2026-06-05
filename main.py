"""
main.py - the search pipeline plus a CLI for testing without Streamlit.

run_search() is the single orchestration function that app.py also calls:
  parse resume -> extract profile -> derive query -> fetch jobs -> rank.

CLI usage (handy for verifying the pipeline cheaply before touching the UI):
  python main.py --resume resume.pdf --role "software engineer" --salary-min 5000
  python main.py --resume-text "....." --role "data analyst"
"""

import argparse
import sys
from typing import Any, Callable

from parse import read_resume_pdf, read_resume_text
from analyzer import extract_resume_profile, rank_jobs, expand_queries
from sources import (
    fetch_mcf, fetch_adzuna, fetch_jsearch, fetch_jooble, dedupe_jobs,
    enrich_mcf_descriptions, extract_years_experience,
    ALL_SOURCES, SOURCE_MCF, SOURCE_ADZUNA, SOURCE_JSEARCH, SOURCE_JOOBLE,
)

# How many jobs to pull per source. The pool is ranked once; the UI pages
# through the ranked pool 10 at a time at no extra cost.
DEFAULT_LIMIT_PER_SOURCE = 50

# With several queries per source we fetch fewer per query, then dedupe and cap.
PER_QUERY_LIMIT = 12


def run_search(
    resume_text: str,
    preferences: dict,
    sources: list[str] | None = None,
    limit_per_source: int = DEFAULT_LIMIT_PER_SOURCE,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Run the full pipeline and return:
        {"profile": dict, "search_term": str, "queries": list[str], "jobs": list[dict]}

    The LLM expands the resume into several related search titles. MyCareersFuture
    and Adzuna are searched with every query; JSearch uses only the primary query
    to protect its limited free quota. Results are deduped, capped per source, and
    ranked best-fit first.
    """
    notify = progress if progress is not None else (lambda _msg: None)
    sources = sources or ALL_SOURCES
    salary_min = preferences.get("salary_min")

    notify("Reading resume...")
    profile = extract_resume_profile(resume_text)

    notify("Generating related search titles...")
    queries = expand_queries(profile, preferences)
    primary = queries[0]
    notify("Searching for: " + ", ".join(queries))

    mcf_jobs: list[dict] = []
    adzuna_jobs: list[dict] = []
    jsearch_jobs: list[dict] = []
    jooble_jobs: list[dict] = []

    if SOURCE_MCF in sources:
        for q in queries:
            mcf_jobs += fetch_mcf(q, limit=PER_QUERY_LIMIT)
        mcf_jobs = dedupe_jobs(mcf_jobs)[:limit_per_source]

    if SOURCE_ADZUNA in sources:
        for q in queries:
            adzuna_jobs += fetch_adzuna(
                q, results_per_page=PER_QUERY_LIMIT, salary_min_monthly=salary_min
            )
        adzuna_jobs = dedupe_jobs(adzuna_jobs)[:limit_per_source]

    if SOURCE_JSEARCH in sources:
        # Single query only: each JSearch call spends your 200/month quota.
        jsearch_jobs = dedupe_jobs(fetch_jsearch(primary, num_results=limit_per_source))

    if SOURCE_JOOBLE in sources:
        for q in queries:
            jooble_jobs += fetch_jooble(q, num_results=PER_QUERY_LIMIT)
        jooble_jobs = dedupe_jobs(jooble_jobs)[:limit_per_source//2]

    jobs = dedupe_jobs(mcf_jobs + adzuna_jobs + jsearch_jobs + jooble_jobs)

    mcf_count = sum(1 for j in jobs if j.get("source") == SOURCE_MCF)
    if mcf_count:
        notify(f"Loading {mcf_count} MyCareersFuture descriptions...")
        jobs = enrich_mcf_descriptions(jobs)

    notify(f"Ranking {len(jobs)} jobs...")
    ranked = rank_jobs(profile, preferences, jobs)

    for job in ranked:
        # Years of experience: read it straight from the full description when the
        # text states a figure; this overrides the model's less reliable guess.
        det_years = extract_years_experience(job.get("description", ""))
        if det_years is not None:
            job["years_experience"] = det_years

        # Do not claim a level signal the data cannot back up. If the model said it
        # used salary or experience but there is none, relabel the source "default".
        src = job.get("level_source", "")
        no_salary = job.get("salary_min") is None and job.get("salary_max") is None
        if src == "salary" and no_salary:
            job["level_source"] = "default"
        elif src == "experience" and job.get("years_experience") is None:
            job["level_source"] = "default"

    # How many jobs each requested source contributed to the final pool, so the UI
    # can show it. A zero means that source returned nothing this run (an outage or
    # simply no matches; the fetchers degrade quietly to an empty list either way).
    source_counts = {s: 0 for s in sources}
    for job in ranked:
        contributed = job.get("source", "")
        if contributed in source_counts:
            source_counts[contributed] += 1

    return {
        "profile": profile,
        "search_term": primary,
        "queries": queries,
        "jobs": ranked,
        "source_counts": source_counts,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Singapore job matcher - resume in, ranked jobs out."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--resume", help="Path to a PDF resume.")
    src.add_argument("--resume-text", help="Resume text pasted inline.")
    parser.add_argument("--role", default="", help="Target role / keywords.")
    parser.add_argument("--salary-min", type=float, default=None,
                        help="Minimum monthly salary in SGD (soft preference).")
    parser.add_argument("--salary-max", type=float, default=None,
                        help="Maximum monthly salary in SGD (soft preference).")
    parser.add_argument("--location", default="", help="Preferred location.")
    parser.add_argument("--top", type=int, default=10,
                        help="How many ranked jobs to print.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.resume:
            resume_text = read_resume_pdf(args.resume)
        else:
            resume_text = read_resume_text(args.resume_text)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    preferences = {
        "role": args.role,
        "salary_min": args.salary_min,
        "salary_max": args.salary_max,
        "location": args.location,
    }

    try:
        result = run_search(resume_text, preferences, progress=print)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    jobs = result["jobs"]
    print(f"\nTop {min(args.top, len(jobs))} of {len(jobs)} jobs")
    print(f"(searched: {', '.join(result['queries'])})\n")
    for i, job in enumerate(jobs[:args.top], start=1):
        sal = _fmt_salary_cli(job)
        print(f"{i}. [{job['fit_score']}] {job['title']} - {job['company']} "
              f"({job['source']})")
        print(f"   {sal} | {job['location']}")
        lvl = _fmt_level_cli(job)
        if lvl:
            print(f"   {lvl}")
        if job.get("why_match"):
            print(f"   Why: {job['why_match']}")
        print(f"   {job['url']}\n")
    return 0


def _fmt_salary_cli(job: dict) -> str:
    lo, hi = job.get("salary_min"), job.get("salary_max")
    if not lo and not hi:
        return "Salary not disclosed"
    est = " (estimated)" if job.get("salary_estimated") else ""
    if lo and hi:
        return f"S${lo:,.0f}-S${hi:,.0f}{est}"
    return f"S${(lo or hi):,.0f}{est}"


def _fmt_level_cli(job: dict) -> str:
    parts = []
    if job.get("level"):
        src = job.get("level_source")
        parts.append(f"Level: {job['level']}" + (f" (from {src})" if src else ""))
    yrs = job.get("years_experience")
    if yrs is not None:
        parts.append(f"{yrs}+ yrs exp")
    return " | ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
