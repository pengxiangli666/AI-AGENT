import pytest

from mcp_server.tools.browser import extract_dom, is_url_allowed

HTML = """
<html><head><title>Docs - Widget API</title><style>.x{}</style><script>track()</script></head>
<body>
 <header><nav><a href="/">Home</a><a href="/pricing">Pricing</a></nav></header>
 <div class="cookie-banner">We use cookies <button>OK</button></div>
 <main>
   <h1>Widget API</h1>
   <p>Create widgets with <code>POST /widgets</code>.</p>
   <h2>Install</h2>
   <pre>pip install widget</pre>
   <ul><li>Fast</li><li>Small</li></ul>
   <a href="/ref">Reference</a>
   <a href="#top">top</a>
 </main>
 <div class="newsletter-signup">Subscribe!</div>
 <footer>© 2026</footer>
 <script>more()</script>
</body></html>
"""


def test_extract_dom_keeps_content_drops_noise():
    out = extract_dom(HTML, "https://example.com/docs")
    t = out["text"]
    assert out["title"] == "Docs - Widget API"
    assert "# Widget API" in t and "## Install" in t and "pip install widget" in t and "- Fast" in t
    for noise in ["cookies", "Subscribe", "track()", "more()", "© 2026", "Pricing"]:
        assert noise not in t, noise
    assert {"text": "Reference", "href": "/ref"} in out["links"]
    assert all(l["href"] != "#top" for l in out["links"])
    assert out["reduced_chars"] < out["raw_chars"]


def test_extract_dom_truncates():
    big = "<main>" + "<p>word " * 5000 + "</main>"
    out = extract_dom(big, max_chars=1000)
    assert out["truncated"] and len(out["text"]) == 1000


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:8000/", "http://169.254.169.254/latest/meta-data/", "http://10.0.0.5/", "file:///etc/passwd",
     "ftp://example.com/", "http://[::1]/", "http://0.0.0.0/"],
)
def test_ssrf_blocklist(url):
    ok, _ = is_url_allowed(url, faas_domain="apps.test")
    assert not ok


def test_faas_domain_is_allowed_even_if_private():
    assert is_url_allowed("http://app-1234abcd.apps.test:8080/", faas_domain="apps.test")[0]
    ok, why = is_url_allowed("http://app-1234abcd.apps.test/", faas_domain="apps.test")
    assert ok and why == "faas app"
    # lookalike hosts must NOT take the faas shortcut (suffix match requires the dot boundary)
    for lookalike in ("http://evilapps.test/", "http://apps.test.evil.invalid/"):
        assert is_url_allowed(lookalike, faas_domain="apps.test")[1] != "faas app"
