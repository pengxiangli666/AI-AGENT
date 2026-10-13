"""Browser tools (Playwright) with DOM-tree reduction to cut page noise.

Two guards matter for an agent that can be told to open arbitrary URLs:
  * SSRF filter - refuse loopback / private / link-local / metadata addresses
    unless the host is under FAAS_DOMAIN (the agent's own deployed apps).
  * Output cap - the reduced page text is truncated before it reaches the model.
"""
from __future__ import annotations

import ipaddress
import os
import socket
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment

from mcp_server.registry import mcp_tool

NOISE_TAGS = ["script", "style", "noscript", "svg", "canvas", "iframe", "form", "nav", "footer", "aside", "header", "template"]
NOISE_HINTS = ("cookie", "banner", "advert", "ad-", "ads-", "popup", "modal", "newsletter", "subscribe", "sidebar", "breadcrumb", "share")
KEEP = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code", "blockquote", "table", "tr", "th", "td", "a", "title"}
MAX_CHARS = 20_000
MAX_HTML_CHARS = 2_000_000


def is_url_allowed(url: str, faas_domain: str | None = None) -> tuple[bool, str]:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False, "only http(s) URLs are allowed"
    host = p.hostname.lower()
    faas = (faas_domain or os.getenv("FAAS_DOMAIN", "localhost")).lower()
    if host == faas or host.endswith("." + faas):
        return True, "faas app"
    try:
        infos = socket.getaddrinfo(host, p.port or (443 if p.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False, "host does not resolve"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False, f"blocked address {ip}"
    return True, "public host"


def extract_dom(html: str, base_url: str = "", max_chars: int = MAX_CHARS) -> dict[str, Any]:
    """Reduce an HTML page to its content skeleton: title, headings, text blocks, code, links."""
    html = html[:MAX_HTML_CHARS]  # bound parse cost on pathological / hostile pages
    soup = BeautifulSoup(html, "html.parser")
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    # Collect first, then decompose deepest-first (reverse document order) so a node is never
    # touched after its ancestor was cleared.  (Don't use hasattr(tag, ...): on a bs4 Tag it
    # falls back to a full-tree find() per call, which made this loop quadratic.)
    doomed = list(soup(NOISE_TAGS))
    for t in soup.find_all(True):
        ident = " ".join([*(t.get("class") or []), t.get("id") or "", t.get("role") or ""]).lower()
        if ident and any(h in ident for h in NOISE_HINTS) and t.name not in ("html", "body", "main", "article"):
            doomed.append(t)
    seen: set[int] = set()
    unique = [t for t in doomed if not (id(t) in seen or seen.add(id(t)))]
    unique.sort(key=lambda t: t.sourceline or 0)  # best-effort ordering; reversed below
    for t in reversed(unique):
        try:
            t.decompose()
        except Exception:  # already cleared along with a decomposed ancestor
            pass

    root = soup.find("main") or soup.find("article") or soup.body or soup
    lines: list[str] = []
    links: list[dict[str, str]] = []
    total = 0
    for el in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "tr"]):
        if total > max_chars:
            break  # enough content; also keeps unclosed-tag nesting (quadratic get_text) bounded
        text = " ".join(el.get_text(" ", strip=True).split())
        total += len(text)
        if not text:
            continue
        if el.name.startswith("h"):
            lines.append(f"{'#' * int(el.name[1])} {text}")
        elif el.name == "pre":
            lines.append("```\n" + el.get_text().strip("\n") + "\n```")
        elif el.name == "li":
            lines.append(f"- {text}")
        elif el.name == "tr":
            lines.append("| " + " | ".join(c.get_text(" ", strip=True) for c in el.find_all(["th", "td"])) + " |")
        else:
            lines.append(text)
    for a in root.find_all("a", href=True):
        label = " ".join(a.get_text(" ", strip=True).split())
        if label and not a["href"].startswith(("#", "javascript:")):
            links.append({"text": label[:80], "href": a["href"]})

    # de-duplicate adjacent repeats and cap size
    deduped: list[str] = []
    for ln in lines:
        if not deduped or deduped[-1] != ln:
            deduped.append(ln)
    text = "\n".join(deduped)
    truncated = len(text) > max_chars
    return {
        "title": (soup.title.get_text(strip=True) if soup.title else ""),
        "url": base_url,
        "text": text[:max_chars],
        "links": links[:50],
        "truncated": truncated,
        "raw_chars": len(html),
        "reduced_chars": min(len(text), max_chars),
    }


async def _fetch_html(url: str, wait_selector: str | None, timeout_ms: int) -> tuple[str, int]:
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"] if os.getenv("PLAYWRIGHT_NO_SANDBOX") else [])
        try:
            ctx = await browser.new_context(java_script_enabled=True)
            page = await ctx.new_page()
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            if wait_selector:
                await page.wait_for_selector(wait_selector, timeout=timeout_ms)
            return await page.content(), (resp.status if resp else 0)
        finally:
            await browser.close()


@mcp_tool
async def browse_page(url: str, wait_selector: str = "", timeout_ms: int = 20000) -> dict[str, Any]:
    """Open a web page in a headless browser and return a noise-reduced text view of it.

    Args:
        url: http(s) URL to open. Private and loopback addresses are refused.
        wait_selector: Optional CSS selector to wait for (for JS-rendered pages).
        timeout_ms: Navigation timeout in milliseconds.
    """
    ok, why = is_url_allowed(url)
    if not ok:
        raise ValueError(f"URL refused: {why}")
    html, status = await _fetch_html(url, wait_selector or None, max(1000, min(timeout_ms, 60000)))
    out = extract_dom(html, url)
    out["status"] = status
    return out


@mcp_tool
async def ui_check(url: str, selector: str, expect_text: str = "", timeout_ms: int = 15000) -> dict[str, Any]:
    """Check that an element exists on a page (and optionally contains text) - for UI smoke tests.

    Args:
        url: http(s) URL to open (the deployed app, usually).
        selector: CSS selector that should match at least one element.
        expect_text: If set, the matched element's text must contain this string.
        timeout_ms: Navigation and wait timeout in milliseconds.
    """
    ok, why = is_url_allowed(url)
    if not ok:
        raise ValueError(f"URL refused: {why}")
    from playwright.async_api import TimeoutError as PWTimeout, async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"] if os.getenv("PLAYWRIGHT_NO_SANDBOX") else [])
        try:
            page = await (await browser.new_context()).new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                el = await page.wait_for_selector(selector, timeout=timeout_ms)
            except PWTimeout:
                return {"found": False, "text": "", "matches_text": False}
            text = (await el.inner_text()) if el else ""
            return {"found": True, "text": text[:500], "matches_text": (expect_text in text) if expect_text else True}
        finally:
            await browser.close()
