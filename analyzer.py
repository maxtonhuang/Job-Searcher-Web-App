"""
analyzer.py - the functions that call the LLM (via llm.ask_json).

Same role as the Resume Analyzer's analyzer.py: each function builds a user
message, calls ask_json once, and returns structured data. No HTTP calls live
here (those are in sources.py).
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor

from llm import ask_json
from prompts import (
    RESUME_PROFILE_PROMPT,
    JOB_MATCH_PROMPT,
    CHAT_FOLLOWUP_PROMPT,
    QUERY_EXPANSION_PROMPT,
    FOLLOWUP_ROUTER_PROMPT,
    SEMANTIC_MAP_PROMPT,
)


# Canonical seniority buckets shared by the ranker, the level filter, and the UI.
# The ranking prompt must classify every job into one of these exact strings.
LEVELS = ["Intern", "Junior", "Mid", "Senior", "Lead/Manager", "Director/Exec"]


# Variants the model or a job board might return, mapped onto the buckets. Keys are
# normalized (lowercase, with hyphens and slashes turned into spaces).
_LEVEL_ALIASES = {
    "intern": "Intern", "internship": "Intern", "trainee": "Intern",
    "attachment": "Intern",
    "junior": "Junior", "junior executive": "Junior", "non executive": "Junior",
    "fresh entry level": "Junior", "fresh entry": "Junior", "entry level": "Junior",
    "entry": "Junior", "fresh": "Junior", "associate": "Junior", "graduate": "Junior",
    "mid": "Mid", "mid level": "Mid", "midlevel": "Mid", "intermediate": "Mid",
    "executive": "Mid", "professional": "Mid", "experienced": "Mid",
    "senior": "Senior", "senior executive": "Senior", "sr": "Senior",
    "lead manager": "Lead/Manager", "team lead": "Lead/Manager", "lead": "Lead/Manager",
    "principal": "Lead/Manager", "supervisor": "Lead/Manager",
    "middle management": "Lead/Manager", "manager": "Lead/Manager",
    "management": "Lead/Manager",
    "director exec": "Director/Exec", "director": "Director/Exec",
    "senior management": "Director/Exec", "head of": "Director/Exec",
    "head": "Director/Exec", "vice president": "Director/Exec", "vp": "Director/Exec",
    "chief": "Director/Exec",
}


def _norm(value) -> str:
    """Lowercase and flatten hyphens/slashes so aliases match loosely."""
    s = (value or "").replace("-", " ").replace("/", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def _clean_level(value) -> str:
    """
    Map the model's (or a portal's) level string onto one of the six buckets.

    Accepts exact bucket names, then known aliases, then a loose containment check
    (longest alias first, so "senior executive" wins over "executive"). Returns ""
    only when nothing matches, which the UI shows as Unknown.
    """
    v = (value or "").strip()
    if v in LEVELS:
        return v
    n = _norm(v)
    if not n:
        return ""
    if n in _LEVEL_ALIASES:
        return _LEVEL_ALIASES[n]
    for key in sorted(_LEVEL_ALIASES, key=len, reverse=True):
        if key in n:
            return _LEVEL_ALIASES[key]
    return ""


def _clean_years(value):
    """Coerce the model's years_experience into a non-negative int, or None."""
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None


# ---------------------------------------------------------------------------
# Resume extraction
# ---------------------------------------------------------------------------

def extract_resume_profile(resume_text: str) -> dict:
    """Turn plain resume text into a compact structured profile dict."""
    user = f"RESUME TEXT:\n\n{resume_text}"
    return ask_json(RESUME_PROFILE_PROMPT, user, max_tokens=10000)


def expand_queries(profile: dict, preferences: dict) -> list[str]:
    """Infer the candidate's field and return related job-search query strings."""
    raw_role = preferences.get("role", "") or ""
    # Users can seed multiple roles, separated by commas, semicolons, or newlines.
    seed_roles: list[str] = []
    for r in re.split(r"[,;\n]+", raw_role):
        rn = r.strip()
        if rn and rn.lower() not in {s.lower() for s in seed_roles}:
            seed_roles.append(rn)

    payload = {
        "skills": (profile.get("skills") or [])[:25],
        "recent_titles": profile.get("recent_titles", []),
        "education": profile.get("education", []),
        "summary": profile.get("summary", ""),
        "target_roles": seed_roles,
    }
    user = f"PROFILE:\n{json.dumps(payload, indent=2)}"
    result = ask_json(QUERY_EXPANSION_PROMPT, user, temperature=0.3, max_tokens=10000)
    raw = result.get("queries", []) if isinstance(result, dict) else []

    cleaned: list[str] = []
    # Put the user's seed roles first so the LLM cannot drop them.
    for q in seed_roles + (raw if isinstance(raw, list) else []):
        if isinstance(q, str) and q.strip():
            qn = q.strip()
            if qn.lower() not in {c.lower() for c in cleaned}:
                cleaned.append(qn)

    if not cleaned:
        fallback = (profile.get("recent_titles") or ["software engineer"])[0]
        cleaned = [fallback]
    # Allow a generous pool: seeds always survive, plus room for resume-derived
    # and related titles. Hard ceiling guards against runaway portal fetches.
    cap = min(20, max(15, len(seed_roles) + 12))
    return cleaned[:cap]


# ---------------------------------------------------------------------------
# Job ranking
# ---------------------------------------------------------------------------

def _monthly_salary(job: dict):
    """
    Return (salary_min, salary_max) only when the figure is monthly; else (None, None).

    Hourly, weekly, or daily pay must not be compared on a monthly basis, so we hide
    it from the ranker and the chat assistant rather than have them treat, say, an
    hourly 15 to 20 as if it were a monthly salary.
    """
    period = (job.get("salary_period") or "month").lower()
    if period != "month":
        return None, None
    return job.get("salary_min"), job.get("salary_max")


def _compact_for_ranking(job: dict) -> dict:
    """Trim a normalized job down to the fields the ranker needs (saves tokens)."""
    sal_min, sal_max = _monthly_salary(job)
    return {
        "id": job.get("id", ""),
        "source": job.get("source", ""),
        "title": job.get("title", ""),
        "company": job.get("company", ""),
        "salary_min": sal_min,
        "salary_max": sal_max,
        "location": job.get("location", ""),
        "level": job.get("level", ""),
        "employment_type": job.get("employment_type", ""),
        "skills": (job.get("skills") or [])[:25],
        "category": job.get("category", ""),
        "snippet": (job.get("description") or "")[:1000],
    }


def rank_jobs(profile: dict, preferences: dict, jobs: list[dict]) -> list[dict]:
    """
    Rank jobs against the candidate profile and preferences.

    Returns the same job dicts with fit_score, why_match, and summary added,
    sorted by fit_score descending. Jobs the model omits are appended at the end.
    """
    if not jobs:
        return []

    compact = [_compact_for_ranking(j) for j in jobs]
    user = (
        f"CANDIDATE PROFILE:\n{json.dumps(profile, indent=2)}\n\n"
        f"PREFERENCES:\n{json.dumps(preferences, indent=2)}\n\n"
        f"JOBS:\n{json.dumps(compact, indent=2)}"
    )
    result = ask_json(JOB_MATCH_PROMPT, user, temperature=0.2, max_tokens=30000)
    ranked_meta = result.get("ranked", []) if isinstance(result, dict) else []

    by_id = {j.get("id", ""): j for j in jobs}
    ordered: list[dict] = []
    seen: set[str] = set()

    for entry in ranked_meta:
        jid = entry.get("id", "")
        job = by_id.get(jid)
        if job is None or jid in seen:
            continue
        seen.add(jid)
        enriched = dict(job)
        enriched["fit_score"] = int(entry.get("fit_score", 0) or 0)
        enriched["why_match"] = entry.get("why_match", "")
        enriched["summary"] = (job.get("description") or "")[:250]
        model_level = _clean_level(entry.get("level", ""))
        if model_level:
            enriched["level"] = model_level
            enriched["level_source"] = entry.get("level_source", "") or ""
        else:
            # Model gave nothing usable; fall back to the portal's own level field.
            enriched["level"] = _clean_level(job.get("level", ""))
            enriched["level_source"] = "portal" if enriched["level"] else ""
        enriched["years_experience"] = _clean_years(entry.get("years_experience"))
        ordered.append(enriched)

    # Any job the model did not return goes to the bottom with a zero score.
    for jid, job in by_id.items():
        if jid not in seen:
            enriched = dict(job)
            enriched["fit_score"] = 0
            enriched["why_match"] = ""
            enriched["summary"] = ""
            enriched["level"] = _clean_level(job.get("level", ""))
            enriched["level_source"] = "portal" if enriched["level"] else ""
            enriched["years_experience"] = None
            ordered.append(enriched)

    return ordered


# ---------------------------------------------------------------------------
# Chat follow-up
# ---------------------------------------------------------------------------

def _compact_for_chat(job: dict) -> dict:
    sal_min, sal_max = _monthly_salary(job)
    return {
        "id": job.get("id", ""),
        "source": job.get("source", ""),
        "title": job.get("title", ""),
        "company": job.get("company", ""),
        "salary_min": sal_min,
        "salary_max": sal_max,
        "location": job.get("location", ""),
        "level": job.get("level", ""),
        "fit_score": job.get("fit_score", 0),
        "skills": (job.get("skills") or [])[:25],
        "snippet": (job.get("description") or "")[:1000],
    }


def answer_followup(profile: dict, jobs: list[dict], question: str) -> dict:
    """
    Answer a follow-up question over the current jobs.

    Returns {"answer": str, "job_ids": list[str] | None}. When job_ids is a list,
    the UI re-renders that filtered/re-ordered subset; when None, it just shows
    the answer text.
    """
    brief_profile = {
        "skills": (profile.get("skills") or [])[:25],
        "recent_titles": profile.get("recent_titles", []),
        "total_years_experience": profile.get("total_years_experience", ""),
    }
    compact = [_compact_for_chat(j) for j in jobs]
    user = (
        f"CANDIDATE PROFILE (brief):\n{json.dumps(brief_profile, indent=2)}\n\n"
        f"CURRENT JOBS:\n{json.dumps(compact, indent=2)}\n\n"
        f"USER QUESTION:\n{question}"
    )
    result = ask_json(CHAT_FOLLOWUP_PROMPT, user, temperature=0.2, max_tokens=30000)
    if not isinstance(result, dict):
        return {"answer": "Sorry, I could not process that.", "job_ids": None}

    job_ids = result.get("job_ids")
    if not isinstance(job_ids, list):
        job_ids = None
    return {"answer": result.get("answer", ""), "job_ids": job_ids}


# ---------------------------------------------------------------------------
# Hybrid follow-up: route, then keyword filter / semantic map / general handler
# ---------------------------------------------------------------------------

_SEMANTIC_BATCH_SIZE = 12
_SEMANTIC_MAX_WORKERS = 6

# In-process cache of semantic verdicts, keyed by (predicate, job id, content hash).
# The content hash means a reused id with different text never returns a stale
# verdict. Scope is the running process, so repeated questions in a session are free.
_semantic_cache: dict = {}


def _job_haystack(job: dict) -> str:
    """Lowercased title + skills + full description, for literal term matching."""
    parts = [
        job.get("title", ""),
        " ".join(job.get("skills") or []),
        job.get("description", "") or "",
    ]
    return " ".join(parts).lower()


def _term_present(haystack: str, term: str) -> bool:
    """
    True if term appears in haystack as a whole token (case-insensitive). Token
    boundaries are transitions between alphanumeric and non-alphanumeric, so "java"
    does not match inside "javascript" and "c++" is matched literally.
    """
    t = (term or "").strip().lower()
    if not t:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def keyword_filter_jobs(jobs: list[dict], terms, polarity: str,
                        match: str = "any") -> list[str]:
    """
    Deterministic literal filter. Returns the ids to KEEP.

    polarity "exclude" keeps jobs where NONE of the terms appear; "include" keeps
    jobs where any (match "any") or all (match "all") of the terms appear.
    """
    terms = [t for t in (terms or []) if isinstance(t, str) and t.strip()]
    if not terms:
        return [j.get("id", "") for j in jobs]
    kept: list[str] = []
    for job in jobs:
        hay = _job_haystack(job)
        present = [t for t in terms if _term_present(hay, t)]
        if polarity == "exclude":
            keep = len(present) == 0
        elif match == "all":
            keep = len(present) == len(terms)
        else:
            keep = len(present) > 0
        if keep:
            kept.append(job.get("id", ""))
    return kept


def _semantic_compact(job: dict) -> dict:
    return {
        "id": job.get("id", ""),
        "title": job.get("title", ""),
        "skills": (job.get("skills") or [])[:25],
        "snippet": (job.get("description") or "")[:1000],
    }


def _evaluate_batch(predicate: str, batch: list[dict]) -> dict:
    """Return {id: bool} for one small batch. Any failure or omission is False."""
    user = (
        f"PROPERTY:\n{predicate}\n\n"
        f"JOBS:\n{json.dumps([_semantic_compact(j) for j in batch], indent=2)}"
    )
    try:
        result = ask_json(SEMANTIC_MAP_PROMPT, user, temperature=0.0, max_tokens=10000)
    except Exception:
        return {j.get("id", ""): False for j in batch}
    verdicts: dict = {}
    raw = result.get("results", []) if isinstance(result, dict) else []
    for entry in raw:
        if isinstance(entry, dict) and "id" in entry:
            verdicts[str(entry["id"])] = bool(entry.get("verdict"))
    for j in batch:
        verdicts.setdefault(j.get("id", ""), False)
    return verdicts


def semantic_filter_jobs(jobs: list[dict], predicate: str,
                         polarity: str) -> list[str]:
    """
    Judgement filter for fuzzy properties. Evaluates the predicate per job in small
    parallel batches (cached), then returns the ids to KEEP. polarity "include"
    keeps jobs where the predicate is true; "exclude" keeps jobs where it is false
    (which includes jobs whose text is silent on the property).
    """
    predicate = (predicate or "").strip()
    if not predicate:
        return [j.get("id", "") for j in jobs]

    verdicts: dict = {}
    todo: list[dict] = []
    for job in jobs:
        key = (predicate, job.get("id", ""), hash(_job_haystack(job)))
        if key in _semantic_cache:
            verdicts[job.get("id", "")] = _semantic_cache[key]
        else:
            todo.append(job)

    batches = [todo[i:i + _SEMANTIC_BATCH_SIZE]
               for i in range(0, len(todo), _SEMANTIC_BATCH_SIZE)]
    if batches:
        with ThreadPoolExecutor(max_workers=_SEMANTIC_MAX_WORKERS) as pool:
            results = pool.map(lambda b: _evaluate_batch(predicate, b), batches)
            for batch, result in zip(batches, results):
                for job in batch:
                    jid = job.get("id", "")
                    v = result.get(jid, False)
                    verdicts[jid] = v
                    _semantic_cache[(predicate, jid, hash(_job_haystack(job)))] = v

    kept: list[str] = []
    for job in jobs:
        jid = job.get("id", "")
        v = verdicts.get(jid, False)
        keep = (not v) if polarity == "exclude" else v
        if keep:
            kept.append(jid)
    return kept


def _filter_summary(kept: int, total: int) -> str:
    return f"Showing {kept} of {total} matching jobs."


def handle_followup(profile: dict, jobs: list[dict], question: str) -> dict:
    """
    Entry point for the chat box. Routes the question with one cheap LLM call, then
    handles it with the most reliable method available:
      - "keyword": deterministic literal filter (no further LLM calls),
      - "semantic": batched, parallel, cached per-job judgement,
      - "other": the existing single-pass handler (re-order or informational).
    Returns {"answer": str, "job_ids": list[str] | None}.
    """
    plan = route_followup(question)
    mode = plan.get("mode", "other")
    polarity = plan.get("polarity", "include") or "include"
    answer = plan.get("answer", "") or ""

    if mode == "keyword":
        ids = keyword_filter_jobs(
            jobs, plan.get("terms", []), polarity, plan.get("match", "any") or "any"
        )
        return {"answer": answer or _filter_summary(len(ids), len(jobs)),
                "job_ids": ids}

    if mode == "semantic":
        ids = semantic_filter_jobs(jobs, plan.get("predicate", ""), polarity)
        return {"answer": answer or _filter_summary(len(ids), len(jobs)),
                "job_ids": ids}

    return answer_followup(profile, jobs, question)


def route_followup(question: str) -> dict:
    """Classify a chat question into a handling plan. Cheap: it never sees the jobs."""
    user = f"USER QUESTION:\n{question}"
    try:
        result = ask_json(FOLLOWUP_ROUTER_PROMPT, user, temperature=0.0, max_tokens=5000)
    except Exception:
        return {"mode": "other", "answer": ""}
    return result if isinstance(result, dict) else {"mode": "other", "answer": ""}