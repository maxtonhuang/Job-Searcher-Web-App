"""
app.py - Streamlit UI for the Singapore job matcher.

Form in, ranked job cards out, with a Show more pager and a chat box that can
filter, re-order, or answer questions about the jobs already on screen.
"""

import tempfile
from pathlib import Path

import streamlit as st

from main import run_search
from sources import ALL_SOURCES, SOURCE_MCF, fetch_mcf_detail
from analyzer import answer_followup, LEVELS
from parse import read_resume_pdf

PAGE_SIZE = 50
LEVEL_FILTER_OPTIONS = LEVELS + ["Unknown"]
EMP_FILTER_OPTIONS = ["Full Time", "Part Time", "Contract", "Internship", "Other"]

st.set_page_config(page_title="SG Job Matcher", layout="wide")
st.title("Singapore Job Matcher")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    st.session_state.setdefault("result", None)       # run_search output
    st.session_state.setdefault("view_jobs", [])       # current (maybe filtered) list
    st.session_state.setdefault("shown", PAGE_SIZE)    # how many cards are visible
    st.session_state.setdefault("chat_history", [])    # list of (role, text)
    st.session_state.setdefault("desc_cache", {})      # job id -> full description
    st.session_state.setdefault("levels", list(LEVEL_FILTER_OPTIONS))  # visible levels
    st.session_state.setdefault("emp_types", list(EMP_FILTER_OPTIONS))  # visible types
    st.session_state.setdefault("sources_filter", list(ALL_SOURCES))   # visible sources

_init_state()


def _reset_after_search(result: dict) -> None:
    st.session_state.result = result
    st.session_state.view_jobs = list(result["jobs"])
    st.session_state.shown = PAGE_SIZE
    st.session_state.chat_history = []


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_run_search(resume_text: str, preferences: dict,
                       sources: list[str], _progress) -> dict:
    """
    Cache identical searches so repeating one (same resume, preferences, and
    sources) within the hour does not re-hit the job APIs or spend LLM tokens or
    Jooble quota. The _progress callback is excluded from the cache key by its
    leading underscore, so progress messages show on a cache miss and are skipped
    on a hit.
    """
    return run_search(resume_text, preferences, sources=sources, progress=_progress)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_salary(job: dict) -> str:
    lo, hi = job.get("salary_min"), job.get("salary_max")
    if not lo and not hi:
        return "Salary not disclosed"
    est = " (estimated)" if job.get("salary_estimated") else ""
    if lo and hi:
        return f"S${lo:,.0f} - S${hi:,.0f}{est}"
    return f"S${(lo or hi):,.0f}{est}"


def _job_bucket(job: dict) -> str:
    """The level bucket used for filtering; anything unrecognized counts as Unknown."""
    lvl = (job.get("level") or "").strip()
    return lvl if lvl in LEVELS else "Unknown"


def _emp_bucket(job: dict) -> str:
    """Map a job's raw employment_type onto a coarse bucket for filtering."""
    raw = (job.get("employment_type") or "").lower()
    if not raw:
        return "Other"
    if any(k in raw for k in ("intern", "attachment", "trainee")):
        return "Internship"
    if "part" in raw:
        return "Part Time"
    if any(k in raw for k in ("contract", "temp", "fixed", "freelance", "contractor")):
        return "Contract"
    if any(k in raw for k in ("full", "permanent", "perm")):
        return "Full Time"
    return "Other"


def _salary_sort_key(job: dict) -> float:
    """A sortable monthly salary, or -1 when pay is unknown or not monthly."""
    period = (job.get("salary_period") or "month").lower()
    if period != "month":
        return -1.0
    val = job.get("salary_max") or job.get("salary_min")
    return float(val) if val else -1.0


def render_card(job: dict, resume_skills: set) -> None:
    jid = job.get("id", "")
    with st.container(border=True):
        top = st.columns([4, 1, 1])
        with top[0]:
            st.markdown(f"### {job.get('title') or 'Untitled role'}")
            st.markdown(f"**{job.get('company', '')}**  -  {job.get('source', '')}")
        with top[1]:
            st.metric("Fit", job.get("fit_score", 0))
        with top[2]:
            yrs = job.get("years_experience")
            st.metric("Years exp", f"{yrs}+" if yrs is not None else "n/a")

        meta = f"{fmt_salary(job)}  |  {job.get('location', '')}"
        bucket = job.get("level")
        if bucket:
            src = job.get("level_source")
            if src == "default":
                meta += f"  |  Level: {bucket} (default)"
            elif src:
                meta += f"  |  Level: {bucket} (from {src})"
            else:
                meta += f"  |  Level: {bucket}"
        if job.get("employment_type"):
            meta += f"  |  {job['employment_type']}"
        if job.get("posted_date"):
            meta += f"  |  Posted {job['posted_date']}"
        if job.get("publisher"):
            meta += f"  |  via {job['publisher']}"
        st.caption(meta)

        if job.get("why_match"):
            st.markdown(f"**Why it matches:** {job['why_match']}")
        if job.get("summary"):
            st.write(job["summary"])

        skills = job.get("skills") or []
        if skills:
            marked = [
                f"**{s}**" if s.strip().lower() in resume_skills else s
                for s in skills[:10]
            ]
            st.caption("Skills: " + ", ".join(marked))

        if job.get("url"):
            st.markdown(f"[Open job posting]({job['url']})")

        matched = job.get("matched_queries") or []
        if matched:
            st.caption("Found via: " + ", ".join(matched))

        _render_description(job, jid)


def _render_description(job: dict, jid: str) -> None:
    """Show the full description, loading it on demand for MyCareersFuture jobs."""
    with st.expander("Full description"):
        cached = st.session_state.desc_cache.get(jid)
        if cached:
            st.write(cached)
        elif job.get("description"):
            st.write(job["description"])
        elif job.get("source") == SOURCE_MCF:
            if st.button("Load full description", key=f"load_{jid}"):
                with st.spinner("Fetching description..."):
                    desc = fetch_mcf_detail(jid)
                st.session_state.desc_cache[jid] = desc or "No description available."
                st.rerun()
        else:
            st.write("No description available.")


# ---------------------------------------------------------------------------
# Sidebar input form
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Your search")
    with st.form("search_form"):
        resume_file = st.file_uploader("Resume PDF", type=["pdf"])

        role = st.text_input(
            "Target role / keywords",
            placeholder="e.g. software engineer, programmer, it support",
            help="Enter one or more roles, separated by commas, semicolons, or new lines. "
                 "The LLM will add related titles based on all of them and your resume.",
        )
        c1, c2 = st.columns(2)
        salary_min = c1.number_input("Min salary (S$/mo)", min_value=0,
                                     value=0, step=500)
        salary_max = c2.number_input("Max salary (S$/mo)", min_value=0,
                                     value=0, step=500)
        location = st.text_input("Preferred location", value="Singapore")
        sources = st.multiselect("Sources", ALL_SOURCES, default=ALL_SOURCES)

        submitted = st.form_submit_button("Search jobs", type="primary")

    if submitted:
        if resume_file is None:
            st.error("Please upload a PDF resume.")
            st.stop()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / resume_file.name
                path.write_bytes(resume_file.getvalue())
                resume_text = read_resume_pdf(str(path))
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        if not sources:
            st.error("Pick at least one source.")
            st.stop()

        preferences = {
            "role": role,
            "salary_min": salary_min or None,
            "salary_max": salary_max or None,
            "location": location,
        }

        with st.status("Working...", expanded=True) as status:
            try:
                result = _cached_run_search(
                    resume_text, preferences, sources,
                    lambda m: status.write(m),
                )
            except Exception as exc:
                status.update(label="Search failed", state="error")
                st.error(f"Unexpected error: {exc}")
                st.stop()
            status.update(label="Done", state="complete")

        if not result["jobs"]:
            st.warning("No jobs came back. Try broader keywords or another source.")
        _reset_after_search(result)

    if st.session_state.result is not None:
        st.divider()
        st.subheader("Filter by level")
        st.session_state.levels = [
            lvl for lvl in LEVEL_FILTER_OPTIONS
            if st.checkbox(lvl, value=True, key=f"lvl_{lvl}")
        ]
        st.subheader("Filter by employment type")
        st.session_state.emp_types = [
            t for t in EMP_FILTER_OPTIONS
            if st.checkbox(t, value=True, key=f"emp_{t}")
        ]
        st.subheader("Filter by source")
        present_sources = [
            s for s in ALL_SOURCES
            if any(j.get("source") == s for j in st.session_state.result["jobs"])
        ]
        st.session_state.sources_filter = [
            s for s in present_sources
            if st.checkbox(s, value=True, key=f"src_{s}")
        ]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

result = st.session_state.result

if result is None:
    st.info("Fill in the form on the left and click Search jobs to begin.")
    st.stop()

view_jobs = st.session_state.view_jobs
selected_levels = set(st.session_state.levels)
selected_emp = set(st.session_state.emp_types)
selected_sources = set(st.session_state.sources_filter)
visible_jobs = [
    j for j in view_jobs
    if _job_bucket(j) in selected_levels
    and _emp_bucket(j) in selected_emp
    and j.get("source", "") in selected_sources
]

resume_skills = {
    s.strip().lower()
    for s in (result.get("profile", {}).get("skills") or [])
    if isinstance(s, str) and s.strip()
}

shown_now = min(st.session_state.shown, len(visible_jobs))
st.subheader(f"Showing {shown_now} of {len(visible_jobs)} matches")
queries = result.get("queries") or []
if queries:
    st.caption("Searched: " + ", ".join(queries))
counts = result.get("source_counts", {})
if counts:
    summary = "   |   ".join(f"{s}: {n}" for s, n in counts.items())
    st.caption(f"Returned per source -> {summary}   (0 = none returned this run)")
if resume_skills:
    st.caption("Skills shown in **bold** appear on your resume.")

sort_label = st.selectbox(
    "Sort by",
    ["Best fit", "Salary (high to low)", "Most recent"],
    index=0,
    key="sort_by",
)
if sort_label == "Salary (high to low)":
    visible_jobs = sorted(visible_jobs, key=_salary_sort_key, reverse=True)
elif sort_label == "Most recent":
    visible_jobs = sorted(
        visible_jobs, key=lambda j: j.get("posted_date") or "", reverse=True
    )
# "Best fit" keeps the ranked order already in view_jobs.

shown = min(st.session_state.shown, len(visible_jobs))
for job in visible_jobs[:shown]:
    render_card(job, resume_skills)

remaining = len(visible_jobs) - shown
if remaining > 0:
    step = min(PAGE_SIZE, remaining)
    if st.button(f"Show {step} more ({remaining} remaining)"):
        st.session_state.shown += PAGE_SIZE
        st.rerun()


# ---------------------------------------------------------------------------
# Chat follow-up
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Ask a follow-up")
st.caption("Try: \"only roles above 8000\", \"hide recruiters\", "
           "\"which best fit my Python experience\", \"MyCareersFuture only\".")

# Show a reset button only when the chat has narrowed the list down from the
# original ranked pool. Clicking it restores the full pool but keeps the chat
# history so the user can see how they got here.
if len(st.session_state.view_jobs) != len(result["jobs"]):
    if st.button(f"Reset to all {len(result['jobs'])} results"):
        st.session_state.view_jobs = list(result["jobs"])
        st.session_state.shown = PAGE_SIZE
        st.rerun()

with st.container(border=True, height=260):
    for role_, text in st.session_state.chat_history:
        with st.chat_message(role_):
            st.write(text)

question = st.chat_input("Refine or ask about these jobs")
if question:
    st.session_state.chat_history.append(("user", question))
    with st.spinner("Thinking..."):
        reply = answer_followup(result["profile"], view_jobs, question)
    st.session_state.chat_history.append(("assistant", reply["answer"]))

    if reply["job_ids"]:
        by_id = {j.get("id", ""): j for j in result["jobs"]}
        filtered = [by_id[i] for i in reply["job_ids"] if i in by_id]
        if filtered:
            st.session_state.view_jobs = filtered
            st.session_state.shown = PAGE_SIZE
    st.rerun()
