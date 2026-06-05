"""
parse.py - resume reading (PDF upload and pasted text). No LLM calls.

Mirrors the parse.py from the Resume Analyzer: read_resume_pdf is reused almost
verbatim, and read_resume_text is added for the paste-text input path.
"""

import re
import sys

from pypdf import PdfReader


# Resume text below this character count likely means an image-only PDF.
_MIN_RESUME_CHARS = 200

# Rough token estimate: 1 token is about 4 chars. Truncate above ~6000 tokens.
_MAX_RESUME_CHARS = 10000 * 4


def read_resume_pdf(path: str) -> str:
    """
    Extract plain text from a PDF resume using pypdf.

    Raises ValueError if the file cannot be opened or the result is too short
    (likely a scanned, image-only PDF with no text layer).
    """
    try:
        reader = PdfReader(path)
    except FileNotFoundError:
        raise ValueError(f"Resume PDF not found: {path}")
    except Exception as exc:
        raise ValueError(f"Could not open resume PDF {path}: {exc}") from exc

    num_pages = len(reader.pages)
    if num_pages > 2:
        print(
            f"Warning: resume has {num_pages} pages; ATS systems typically expect 1.",
            file=sys.stderr,
        )

    page_texts = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(page_texts)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) < _MIN_RESUME_CHARS:
        raise ValueError(
            f"Extracted resume text is only {len(text)} characters; "
            f"the PDF may be image-based or scanned (no text layer)."
        )

    if len(text) > _MAX_RESUME_CHARS:
        print(
            f"Warning: resume text is {len(text)} characters; "
            f"truncating to {_MAX_RESUME_CHARS}.",
            file=sys.stderr,
        )
        text = text[:_MAX_RESUME_CHARS]

    return text


def read_resume_text(raw: str) -> str:
    """
    Clean pasted resume text.

    Raises ValueError if the text is too short to be a real resume.
    """
    text = re.sub(r"\n{3,}", "\n\n", raw or "").strip()
    if len(text) < _MIN_RESUME_CHARS:
        raise ValueError(
            f"Pasted resume text is only {len(text)} characters; "
            f"please paste the full resume."
        )
    if len(text) > _MAX_RESUME_CHARS:
        text = text[:_MAX_RESUME_CHARS]
    return text
