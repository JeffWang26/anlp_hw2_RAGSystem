"""
Data loaders for HTML, PDF, and plain text.
Use BeautifulSoup for HTML, pypdf/pdfplumber for PDF.
Includes filtering for nav, sidebar, copyright, short lines, etc.
"""

import json
import re
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import pdfplumber
import pypdf

# Selectors for main content (extract from these if present)
MAIN_CONTENT_SELECTORS = [
    "#mw-content-text", "#bodyContent",  # Wikipedia
    "#content", "#main", "#article", "#mainContent",
    ".main-content", ".article-body", ".content-area", ".page-content",
    ".post-content", ".entry-content", ".sc-content",  # OpenCities/Granicus
]

# HTML elements to remove (nav, sidebar, footer; keep header - often has article titles)
NOISE_TAGS = {"script", "style", "nav", "footer", "aside", "iframe", "noscript"}
NOISE_CLASS_PATTERNS = re.compile(
    r"sidebar|footer|copyright|legal|cookie|banner|"
    r"advertisement|ads|social-share|share-|toc|table-of-contents|"
    r"breadcrumb|pagination|comment|related-|metadata|widget|"
    r"site-nav|main-nav|top-nav|global-nav|dropdown-menu|sub-menu|"
    r"language-selector|language-dropdown|quick-links"
)
NOISE_ID_PATTERNS = re.compile(
    r"sidebar|footer|copyright|toc|breadcrumb|"
    r"comment|related|widget|ad-|banner|nav"
)

# Lines to filter out (copyright, boilerplate, etc.)
FILTER_PATTERNS = re.compile(
    r"^(all rights reserved|©|copyright|cookie policy|privacy policy|"
    r"terms of use|terms of service|skip to content|jump to content|"
    r"main menu|navigation|search\s+wikipedia|edit this page|"
    r"last edited|page last modified|cite this page|share|follow us|"
    r"subscribe|newsletter|sign up|log in|login|create account|"
    r"quick links|select a language|back to top|opens in new tab|"
    r"select this as your preferred language|website disclaimers|"
    r"accessibility|sitemap|powered by|like us on facebook|"
    r"watch us on youtube|get involved)$",
    re.IGNORECASE,
)


def _is_noise_element(tag) -> bool:
    """Check if element is likely nav/sidebar/footer/copyright."""
    if tag is None or not hasattr(tag, "name"):
        return False
    if tag.name in NOISE_TAGS:
        return True
    attrs = getattr(tag, "attrs", None) or {}
    cls = " ".join(attrs.get("class", []) or []).lower()
    tid = (attrs.get("id") or "").lower()
    if NOISE_CLASS_PATTERNS.search(cls) or NOISE_ID_PATTERNS.search(tid):
        return True
    role = (attrs.get("role") or "").lower()
    if role in ("navigation", "banner", "contentinfo", "complementary"):
        return True
    return False


def _filter_lines(lines: List[str], min_chars: int = 15) -> List[str]:
    """Filter out short lines and common boilerplate. min_chars=15 keeps short addresses."""
    kept = []
    for line in lines:
        s = line.strip()
        if len(s) < min_chars:
            continue
        if FILTER_PATTERNS.match(s):
            continue
        if re.match(r"^[\d\.\-\•\*]+\s*$", s):  # pure bullets/numbers
            continue
        kept.append(s)
    return kept


def load_html(
    html_content: str,
    remove_noise_elements: bool = True,
    min_line_chars: int = 15,
) -> str:
    """
    Extract main text from HTML. Removes nav, sidebar, copyright, etc.
    Args:
        html_content: Raw HTML string.
        remove_noise_elements: Whether to strip nav/footer/sidebar elements.
        min_line_chars: Minimum character length for a line to be kept.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Prefer main content container if present (Wikipedia, etc.)
    root = soup
    for sel in MAIN_CONTENT_SELECTORS:
        tag = soup.select_one(sel)
        if tag:
            root = tag
            break

    if remove_noise_elements:
        for tag in root.find_all(True):
            if _is_noise_element(tag):
                tag.decompose()

    text = root.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    lines = [l for l in lines if l]
    lines = _filter_lines(lines, min_chars=min_line_chars)
    return "\n".join(lines)


def load_pdf_pypdf(filepath: str) -> str:
    """Extract text from PDF using pypdf."""
    reader = pypdf.PdfReader(filepath)
    texts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            texts.append(t)
    return "\n\n".join(texts)


def load_pdf_pdfplumber(filepath: str) -> str:
    """Extract text from PDF using pdfplumber (often better for tables)."""
    texts = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
    return "\n\n".join(texts)


def fetch_html_raw(url: str, timeout: int = 10) -> str:
    """Fetch raw HTML from URL."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def load_html_from_url(
    url: str,
    timeout: int = 10,
    **load_html_kwargs,
) -> str:
    """Fetch HTML from URL and return extracted text."""
    raw = fetch_html_raw(url, timeout=timeout)
    return load_html(raw, **load_html_kwargs)


def extract_links_from_html(html_content: str, base_url: str, same_domain_only: bool = True) -> List[str]:
    """
    Extract href links from HTML. Returns absolute URLs.
    base_url: e.g. https://www.pittsburghpa.gov/Home
    same_domain_only: if True, only return links on the same domain.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    base_netloc = urlparse(base_url).netloc
    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        if parsed.scheme not in ("http", "https"):
            continue
        if same_domain_only and parsed.netloc != base_netloc:
            continue
        if full_url not in seen:
            seen.add(full_url)
            links.append(full_url)
    return links


def load_plain_text(filepath: str) -> str:
    """Load plain text file with encoding handling."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_document(filepath: str) -> str:
    """
    Load document based on extension.
    Supports .html, .htm, .pdf, .txt
    """
    path = Path(filepath)
    suf = path.suffix.lower()
    if suf in (".html", ".htm"):
        content = load_plain_text(filepath)
        return load_html(content)
    elif suf == ".pdf":
        return load_pdf_pdfplumber(filepath)
    elif suf == ".txt" or suf == "":
        return load_plain_text(filepath)
    else:
        return load_plain_text(filepath)


def load_questions_txt(filepath: str) -> List[str]:
    """Load questions from data/train/questions.txt (one per line)."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_reference_answers(filepath: str) -> dict:
    """Load reference answers from data/train/reference_answers.json."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_leaderboard_queries(filepath: str) -> List[Tuple[str, str]]:
    """
    Load leaderboard/test queries.
    Returns list of (question, id) tuples.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [(item["question"], str(item["id"])) for item in data]
    return [(data[k], k) for k in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x)]
