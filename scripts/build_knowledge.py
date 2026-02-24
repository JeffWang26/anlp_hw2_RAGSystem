#!/usr/bin/env python3
"""
Build knowledge resource from raw data.
- Scrape/copy HTML, PDF, TXT into data/raw/
- Run this script to process and output to data/knowledge/

For HTML with sub-links: create a .baseurl file (same stem) containing the page URL, e.g.
  Home - Pittsburgh, PA.baseurl  ->  https://www.pittsburghpa.gov/Home
Then use --follow-links to crawl same-domain sub-pages (optionally recursive).

Example:
  python scripts/build_knowledge.py --input data/raw --output data/knowledge
  python scripts/build_knowledge.py --follow-links --max-links 20
  python scripts/build_knowledge.py --follow-links --max-depth 2 --max-links 20
"""

import argparse
import re
import sys
from collections import deque
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loaders import (
    load_document,
    load_html,
    fetch_html_raw,
    extract_links_from_html,
    load_plain_text,
)


def _safe_filename(url: str) -> str:
    """Convert URL to safe filename (alphanumeric + dash)."""
    s = re.sub(r"[^\w\-]", "_", url)[:80]
    return s + "_" + str(abs(hash(url)) % 10000)  # avoid collisions


def _get_base_url(f: Path, fallback: Optional[str]) -> Optional[str]:
    """Get base URL for HTML file: from .baseurl file or --base-url."""
    baseurl_file = f.parent / (f.stem + ".baseurl")
    if baseurl_file.exists():
        return baseurl_file.read_text(encoding="utf-8").strip().split("\n")[0]
    return fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw", help="Input directory")
    parser.add_argument("--output", type=str, default="data/knowledge", help="Output directory")
    parser.add_argument("--follow-links", action="store_true", help="Follow sub-links in HTML (same domain)")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for HTML pages (if no .baseurl file)")
    parser.add_argument("--max-links", type=int, default=30, help="Max links to explore per page (default 30)")
    parser.add_argument("--max-depth", type=int, default=2, help="Recursion depth: 1=direct links only, 2=+sub-sub-pages (default 2)")
    parser.add_argument("--max-total", type=int, default=500, help="Max total pages to fetch (prevents runaway, default 500)")
    args = parser.parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    visited_urls = set()

    if not inp.exists():
        inp.mkdir(parents=True, exist_ok=True)
        print(f"Created {inp}. Place HTML/PDF/TXT files there, then re-run.")
        return

    for f in sorted(inp.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in (".html", ".htm", ".pdf", ".txt"):
            continue
        try:
            text = load_document(str(f))
            # Preserve folder structure: raw/site/pg.html -> knowledge/site/pg.txt
            try:
                rel = f.relative_to(inp)
                out_subdir = out / rel.parent
            except ValueError:
                out_subdir = out
            out_subdir.mkdir(parents=True, exist_ok=True)
            out_name = f.stem + ".txt"
            out_file = out_subdir / out_name
            out_file.write_text(text, encoding="utf-8")
            print(f"Processed {f.relative_to(inp)} -> {out_file.relative_to(out)}")

            # Follow sub-links for HTML (BFS with depth)
            if args.follow_links and f.suffix.lower() in (".html", ".htm"):
                base_url = _get_base_url(f, args.base_url)
                if not base_url:
                    continue
                raw_html = load_plain_text(str(f))
                seed_links = extract_links_from_html(raw_html, base_url, same_domain_only=True)[: args.max_links]
                queue = deque((url, 1) for url in seed_links)  # (url, depth)
                queued = set(seed_links)
                fetched_count = 0
                while queue and fetched_count < args.max_total:
                    url, depth = queue.popleft()
                    if url in visited_urls:
                        continue
                    visited_urls.add(url)
                    try:
                        raw = fetch_html_raw(url)
                        sub_text = load_html(raw)
                        if len(sub_text.strip()) >= 50:
                            sub_name = _safe_filename(url) + ".txt"
                            sub_file = out / sub_name
                            sub_file.write_text(sub_text, encoding="utf-8")
                            indent = "  " * depth
                            print(f"{indent}+ depth={depth}: {url[:55]}... -> {sub_name}")
                            fetched_count += 1
                        if depth < args.max_depth:
                            new_links = extract_links_from_html(raw, url, same_domain_only=True)[: args.max_links]
                            for u in new_links:
                                if u not in visited_urls and u not in queued:
                                    queued.add(u)
                                    queue.append((u, depth + 1))
                    except Exception as e:
                        print(f"  - Skip {url[:50]}...: {e}")

        except Exception as e:
            print(f"Error processing {f}: {e}")


if __name__ == "__main__":
    main()
