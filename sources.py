"""
sources.py - job-listing fetchers. No LLM calls.

This is the source-adapter layer. Each portal has its own fetch function, and
every one of them returns a list of NORMALIZED job dicts with the same keys, so
the rest of the app never needs to know which portal a job came from. Adding a
new portal later is just writing one more adapter here.

Normalized job dict:
    {
        "source":           str,            # "MyCareersFuture" or "Adzuna"
        "id":               str,
        "title":            str,
        "company":          str,
        "salary_min":       float | None,    # MONTHLY SGD
        "salary_max":       float | None,    # MONTHLY SGD
        "salary_estimated": bool,            # True if the figure is predicted
        "location":         str,
        "employment_type":  str,
        "level":            str,
        "skills":           list[str],
        "category":         str,
        "url":              str,
        "posted_date":      str,             # YYYY-MM-DD
        "description":      str,             # may be "" until enriched (MCF)
    }

Salary note: MyCareersFuture reports MONTHLY salary; Adzuna reports ANNUAL. We
normalize everything to MONTHLY SGD here (Adzuna annual / 12), so the rest of the
app can compare on one basis.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests

MCF_SEARCH_URL = "https://api.mycareersfuture.gov.sg/v2/search"
MCF_DETAIL_URL = "https://api.mycareersfuture.gov.sg/v2/jobs/{uuid}"
ADZUNA_SEARCH_URL = "https://api.adzuna.com/v1/api/jobs/sg/search/1"
JSEARCH_SEARCH_URL = "https://jsearch.p.rapidapi.com/search"
JOOBLE_SEARCH_URL = "https://jooble.org/api/{key}"

_TIMEOUT = 20

SOURCE_MCF = "MyCareersFuture"
SOURCE_ADZUNA = "Adzuna"
SOURCE_JSEARCH = "JSearch"
SOURCE_JOOBLE = "Jooble"
ALL_SOURCES = [SOURCE_MCF, SOURCE_ADZUNA, SOURCE_JSEARCH, SOURCE_JOOBLE]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace from a description blob."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = text.replace("&amp;", "&").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()


_YEARS_RE = re.compile(
    r"(\d{1,2})\s*\+?\s*(?:-|to)?\s*(\d{1,2})?\s*\+?\s*(?:years?|yrs?)",
    re.IGNORECASE,
)


# Words near a "N years" mention that mark it as an experience requirement.
_EXP_CONTEXT = (
    "experien", "hands on", "hands-on", "exposure", "minimum", "at least",
    "relevant", "proven", "track record", "background", "working", "expertise",
)
# Nouns right after a "N years" mention that mean it is NOT about experience,
# e.g. "2 year contract", "5 year warranty", "10 years ago".
_NON_EXP_AFTER = (
    "contract", "warranty", "lease", "bond", "visa", "guarantee", "old", "ago",
    "tenure",
)


def extract_years_experience(text: str):
    """
    Read the minimum years of experience a job asks for, straight from its text.

    Finds a number next to "year"/"yr" and accepts it as an experience requirement
    when any signal supports that reading: the figure carries a "+" (e.g. "3+
    years"), or an experience keyword sits in the surrounding window. Rejects
    mentions whose following noun marks them as unrelated (e.g. "2 year contract",
    "10 years ago"). Returns the lower bound of the first qualifying mention, or
    None. Deterministic, so it does not depend on the model.
    """
    if not text:
        return None
    low = text.lower()
    for m in _YEARS_RE.finditer(low):
        after = low[m.end(): m.end() + 15]
        if any(bad in after for bad in _NON_EXP_AFTER):
            continue
        has_plus = "+" in m.group(0)
        window = low[max(0, m.start() - 45): m.end() + 40]
        if has_plus or any(good in window for good in _EXP_CONTEXT):
            return int(m.group(1))
    return None


_MONEY_RE = re.compile(r"(\d{1,3}(?:,\d{3})+|\d{3,7})\s*(k)?", re.IGNORECASE)


def _money_value(num_str: str, has_k: bool) -> float:
    n = float(num_str.replace(",", ""))
    return n * 1000 if has_k else n


def extract_salary_from_text(text: str):
    """
    Best-effort monthly SGD salary from free text, for sources with no structured
    figure. Reads one or two amounts (handling "k") found right after a currency
    marker or the word "salary", and converts to monthly: an explicit period in the
    text wins, otherwise an amount of 25,000 or more is assumed annual (divided by
    12). Returns (monthly_min, monthly_max); either may be None. Heuristic, so the
    caller always flags the result estimated.
    """
    if not text:
        return None, None
    low = text.lower()
    for am in re.finditer(r"s?\$|sgd|salary|compensation|remuneration", low):
        window = low[am.start(): am.start() + 70]
        nums = []
        for mm in _MONEY_RE.finditer(window):
            nums.append(_money_value(mm.group(1), bool(mm.group(2))))
            if len(nums) == 2:
                break
        if not nums:
            continue
        is_month = "month" in window or "/mo" in window or " pm" in window
        is_annual = ("annum" in window or "annual" in window or "year" in window
                     or "/yr" in window or "p.a" in window)
        is_hour = "hour" in window or "/hr" in window

        def to_monthly(v):
            if is_month:
                return round(v)
            if is_annual:
                return round(v / 12)
            if is_hour:
                return None  # cannot reliably turn hourly into monthly
            return round(v / 12) if v >= 25000 else round(v)

        lo = to_monthly(min(nums))
        hi = to_monthly(max(nums)) if len(nums) > 1 else None
        if lo is not None or hi is not None:
            return lo, hi
    return None, None


def _nested(d, *keys, default=None):
    """Walk nested dict/list keys, returning default if any step is missing."""
    cur = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        elif isinstance(cur, list) and isinstance(k, int) and 0 <= k < len(cur):
            cur = cur[k]
        else:
            return default
        if cur is None:
            return default
    return cur


def _label(item, *keys) -> str:
    """Pull the first non-empty value from a dict for any of the given keys."""
    if isinstance(item, dict):
        for k in keys:
            if item.get(k):
                return str(item[k])
    return ""


# ---------------------------------------------------------------------------
# MyCareersFuture
# ---------------------------------------------------------------------------

def fetch_mcf(search_term: str, limit: int = 25) -> list[dict]:
    """Search MyCareersFuture and return normalized jobs (empty list on failure)."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    params = {"limit": limit, "page": 0}
    payload = {"search": search_term, "sessionId": ""}
    try:
        resp = requests.post(
            MCF_SEARCH_URL, params=params, json=payload,
            headers=headers, timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results", []) or []
    except Exception as exc:
        print(f"[sources] MyCareersFuture fetch failed: {exc}")
        return []
    return [_normalize_mcf(r) for r in results]


def _normalize_mcf(r: dict) -> dict:
    company = _nested(r, "postedCompany", "name") or "Confidential"
    location = (
        _nested(r, "address", "building")
        or _nested(r, "address", "districts", 0, "location")
        or _nested(r, "address", "districts", 0, "region")
        or "Singapore"
    )
    employment = ", ".join(
        filter(None, (_label(et, "employmentType", "name")
                      for et in (r.get("employmentTypes") or [])))
    )
    category = next(
        (_label(c, "category", "name") for c in (r.get("categories") or [])), ""
    )
    skills = [s.get("skill", "") for s in (r.get("skills") or []) if s.get("skill")]
    
    # MyCareersFuture states the period (Monthly, Annually, Hourly, ...). Normalize
    # annual figures to monthly; leave hourly (and other) figures as-is but record
    # the period so the UI labels them correctly instead of assuming "/month".
    sal_type = (_nested(r, "salary", "type", "salaryType", default="") or "").lower()
    sal_min = _nested(r, "salary", "minimum")
    sal_max = _nested(r, "salary", "maximum")
    if sal_type == "annually":
        sal_min = round(sal_min / 12) if isinstance(sal_min, (int, float)) and sal_min else None
        sal_max = round(sal_max / 12) if isinstance(sal_max, (int, float)) and sal_max else None
        salary_period = "month"
    elif sal_type in ("hourly", "weekly", "daily"):
        salary_period = {"hourly": "hour", "weekly": "week", "daily": "day"}[sal_type]
    else:
        salary_period = "month"  # Monthly or unstated: treat as monthly

    return {
        "source": SOURCE_MCF,
        "id": r.get("uuid", ""),
        "title": r.get("title", ""),
        "company": company,
        "salary_min": sal_min,
        "salary_max": sal_max,
        "salary_estimated": False,
        "salary_period": salary_period,
        "location": location,
        "employment_type": employment,
        "level": _nested(r, "positionLevels", 0, "position", default=""),
        "skills": skills,
        "category": category,
        "url": _nested(r, "metadata", "jobDetailsUrl", default=""),
        "posted_date": _nested(r, "metadata", "newPostingDate", default=""),
        "description": "",  # not in search results; enrich on demand below
    }


def fetch_mcf_detail(uuid: str) -> str:
    """
    Return the full description text for one MCF job, or "" on failure.

    The search endpoint does not include descriptions, so we fetch the job
    detail by uuid only when the user expands a specific card. If the detail
    endpoint or its shape ever changes, this degrades quietly to "".
    """
    if not uuid:
        return ""
    try:
        resp = requests.get(
            MCF_DETAIL_URL.format(uuid=uuid),
            headers={"Accept": "application/json"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[sources] MCF detail fetch failed for {uuid}: {exc}")
        return ""
    return _strip_html(data.get("description", "") or "")


# How many MCF detail pages to fetch at once. Each MCF job needs its own detail
# request, so fetching them in parallel turns a slow sequence of round trips into a
# few concurrent batches. Lower this if MCF starts throttling the requests.
_MCF_ENRICH_WORKERS = 8


def enrich_mcf_descriptions(jobs: list[dict]) -> list[dict]:
    """
    Fill in the description for each MyCareersFuture job by fetching its detail page.

    MCF search results carry no description, so we load each one up front here so the
    ranker can read it to infer level and years of experience. The requests run in a
    small thread pool so 20-plus jobs do not become 20-plus sequential round trips.
    Non-MCF jobs and any that already have a description are left as is. Failures
    degrade quietly to "" because fetch_mcf_detail returns "" on any error.
    """
    targets = [
        job for job in jobs
        if job.get("source") == SOURCE_MCF and not job.get("description")
    ]
    if not targets:
        return jobs
    with ThreadPoolExecutor(max_workers=_MCF_ENRICH_WORKERS) as pool:
        descriptions = pool.map(lambda j: fetch_mcf_detail(j.get("id", "")), targets)
    for job, desc in zip(targets, descriptions):
        if desc:
            job["description"] = desc
    return jobs


# ---------------------------------------------------------------------------
# Adzuna
# ---------------------------------------------------------------------------

def fetch_adzuna(
    search_term: str,
    results_per_page: int = 25,
    salary_min_monthly: float | None = None,
) -> list[dict]:
    """Search Adzuna (Singapore) and return normalized jobs (empty on failure)."""
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        print("[sources] Adzuna credentials missing in .env; skipping Adzuna.")
        return []

    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": results_per_page,
        "what": search_term,
        "where": "Singapore",
        "content-type": "application/json",
    }
    # Adzuna salary filters are annual; convert the monthly preference up.
    if salary_min_monthly:
        params["salary_min"] = int(salary_min_monthly * 12)

    try:
        resp = requests.get(ADZUNA_SEARCH_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        results = resp.json().get("results", []) or []
    except Exception as exc:
        print(f"[sources] Adzuna fetch failed: {exc}")
        return []
    return [_normalize_adzuna(r) for r in results]


def _normalize_adzuna(r: dict) -> dict:
    annual_min = r.get("salary_min")
    annual_max = r.get("salary_max")
    monthly_min = round(annual_min / 12) if annual_min else None
    monthly_max = round(annual_max / 12) if annual_max else None
    return {
        "source": SOURCE_ADZUNA,
        "id": str(r.get("id", "")),
        "title": r.get("title", ""),
        "company": _nested(r, "company", "display_name", default="Unknown"),
        "salary_min": monthly_min,
        "salary_max": monthly_max,
        "salary_estimated": bool(r.get("salary_is_predicted") in (1, "1", True)),
        "location": _nested(r, "location", "display_name", default="Singapore"),
        "employment_type": " ".join(
            filter(None, [r.get("contract_time", ""), r.get("contract_type", "")])
        ),
        "level": "",
        "skills": [],
        "category": _nested(r, "category", "label", default=""),
        "url": r.get("redirect_url", ""),
        "posted_date": (r.get("created", "") or "")[:10],
        "description": _strip_html(r.get("description", "") or ""),
    }


# ---------------------------------------------------------------------------
# JSearch (Google for Jobs: LinkedIn, Indeed, Glassdoor, and more) via RapidAPI
# ---------------------------------------------------------------------------

def fetch_jsearch(search_term: str, num_results: int = 25) -> list[dict]:
    """Search JSearch (RapidAPI) and return normalized jobs (empty on failure)."""
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        print("[sources] RAPIDAPI_KEY missing in .env; skipping JSearch.")
        return []
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }
    params = {
        "query": f"{search_term} in Singapore",
        "page": "1",
        "num_pages": "1",
        "country": "sg",
    }
    try:
        resp = requests.get(JSEARCH_SEARCH_URL, headers=headers,
                            params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        results = resp.json().get("data", []) or []
    except Exception as exc:
        print(f"[sources] JSearch fetch failed: {exc}")
        return []
    return [_normalize_jsearch(r) for r in results[:num_results]]


def _jsearch_monthly(amount, period: str):
    """Convert a JSearch salary figure to monthly SGD, or None if the unit is unclear."""
    if not amount:
        return None
    period = (period or "").upper()
    if period == "MONTH":
        return round(amount)
    if period == "YEAR":
        return round(amount / 12)
    return None  # HOUR / WEEK / unknown: skip rather than show a misleading figure


def _normalize_jsearch(r: dict) -> dict:
    period = r.get("job_salary_period", "")
    city = r.get("job_city") or ""
    country = r.get("job_country") or "Singapore"
    location = ", ".join(filter(None, [city, country])) or "Singapore"
    publisher = r.get("job_publisher", "")
    title = r.get("job_title", "")
    description = _strip_html(r.get("job_description", "") or "")

    sal_min = _jsearch_monthly(r.get("job_min_salary"), period)
    sal_max = _jsearch_monthly(r.get("job_max_salary"), period)
    if sal_min is None and sal_max is None:
        # No structured figure; read one out of the title and description.
        sal_min, sal_max = extract_salary_from_text(f"{title}. {description}")

    return {
        "source": SOURCE_JSEARCH,
        "id": str(r.get("job_id", "")),
        "title": title,
        "company": r.get("employer_name", "") or "Unknown",
        "salary_min": sal_min,
        "salary_max": sal_max,
        "salary_estimated": True,  # JSearch pay varies in source and currency
        "location": location,
        "employment_type": r.get("job_employment_type", "") or "",
        "level": "",
        "skills": r.get("job_required_skills") or [],
        "category": "",
        "publisher": publisher,
        "url": r.get("job_apply_link", ""),
        "posted_date": (r.get("job_posted_at_datetime_utc", "") or "")[:10],
        "description": description[:1500],
    }


# ---------------------------------------------------------------------------
# Jooble (aggregator across many boards) via its free REST API
# ---------------------------------------------------------------------------

def fetch_jooble(search_term: str, num_results: int = 25) -> list[dict]:
    """Search Jooble and return normalized jobs (empty list on failure)."""
    api_key = os.getenv("JOOBLE_API_KEY")
    if not api_key:
        print("[sources] JOOBLE_API_KEY missing in .env; skipping Jooble.")
        return []
    payload = {"keywords": search_term, "location": "Singapore"}
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(
            JOOBLE_SEARCH_URL.format(key=api_key),
            json=payload, headers=headers, timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("jobs", []) or []
    except Exception as exc:
        print(f"[sources] Jooble fetch failed: {exc}")
        return []
    return [_normalize_jooble(r) for r in results[:num_results]]


def _normalize_jooble(r: dict) -> dict:
    snippet = _strip_html(r.get("snippet", "") or "")
    # Jooble returns salary as free text (e.g. "S$5,000 per month"), so we read it
    # with the same heuristic used for JSearch and always flag the figure estimated.
    sal_min, sal_max = extract_salary_from_text(f"{r.get('salary', '')}. {snippet}")
    return {
        "source": SOURCE_JOOBLE,
        "id": str(r.get("id", "")),
        "title": r.get("title", ""),
        "company": r.get("company", "") or "Unknown",
        "salary_min": sal_min,
        "salary_max": sal_max,
        "salary_estimated": True,
        "location": r.get("location", "") or "Singapore",
        "employment_type": r.get("type", "") or "",
        "level": "",
        "skills": [],
        "category": "",
        "publisher": r.get("source", "") or "",
        "url": r.get("link", ""),
        "posted_date": (r.get("updated", "") or "")[:10],
        "description": snippet[:1500],
    }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def fetch_all(
    search_term: str,
    sources: list[str],
    limit_per_source: int = 25,
    salary_min_monthly: float | None = None,
) -> list[dict]:
    """Fetch from every requested source and return one combined, normalized list."""
    jobs: list[dict] = []
    if SOURCE_MCF in sources:
        jobs += fetch_mcf(search_term, limit=limit_per_source)
    if SOURCE_ADZUNA in sources:
        jobs += fetch_adzuna(
            search_term,
            results_per_page=limit_per_source,
            salary_min_monthly=salary_min_monthly,
        )
    if SOURCE_JSEARCH in sources:
        jobs += fetch_jsearch(search_term, num_results=limit_per_source)
    if SOURCE_JOOBLE in sources:
        jobs += fetch_jooble(search_term, num_results=limit_per_source)
    return jobs


def dedupe_jobs(jobs: list[dict]) -> list[dict]:
    """
    Remove duplicates: exact (source, id) repeats from overlapping queries, and the
    same role appearing across sources (same title + same company).
    """
    seen_ids: set = set()
    seen_pairs: set = set()
    out: list[dict] = []
    for j in jobs:
        key_id = (j.get("source", ""), j.get("id", ""))
        title = (j.get("title", "") or "").strip().lower()
        company = (j.get("company", "") or "").strip().lower()
        if key_id in seen_ids:
            continue
        if title and company and (title, company) in seen_pairs:
            continue
        seen_ids.add(key_id)
        if title and company:
            seen_pairs.add((title, company))
        out.append(j)
    return out