import pytest

from preprocess import build_text, clean_text, near_duplicate_key, prepare_text


def test_build_text_joins_title_and_body():
    assert build_text("Headline", "Body text") == "Headline\n\nBody text"


def test_build_text_handles_missing_parts():
    assert build_text(None, "Body") == "Body"
    assert build_text("Headline", None) == "Headline"
    assert build_text("  ", "") == ""
    assert build_text() == ""


@pytest.mark.parametrize("bad", [123, ["x"], {"a": 1}, b"bytes"])
def test_build_text_rejects_non_strings(bad):
    with pytest.raises(TypeError):
        build_text(bad, "ok")
    with pytest.raises(TypeError):
        build_text("ok", bad)


def test_clean_text_basic_normalisation():
    assert clean_text("  Hello,   WORLD!\n\tNew   line ") == "hello, world! new line"


def test_clean_text_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    with pytest.raises(TypeError):
        clean_text(5)


@pytest.mark.parametrize("dateline", [
    "WASHINGTON (Reuters) - The senate voted today.",
    "SAN FRANCISCO/WASHINGTON (Reuters) - The senate voted today.",
    "(Reuters) - The senate voted today.",
])
def test_clean_text_removes_reuters_datelines(dateline):
    cleaned = clean_text(dateline)
    assert "reuters" not in cleaned
    assert "washington" not in cleaned.split("senate")[0]
    assert cleaned.endswith("the senate voted today.")


def test_clean_text_removes_dateline_after_title():
    cleaned = clean_text(build_text("Senate votes", "WASHINGTON (Reuters) - Lawmakers met."))
    assert cleaned == "senate votes lawmakers met."


def test_clean_text_strips_urls_handles_tags_and_source_markers():
    raw = "See https://example.com/x and www.foo.org <b>bold</b> @someone pic.twitter.com/abc Featured image via Getty Images"
    cleaned = clean_text(raw)
    for fragment in ("http", "www", "<b>", "@someone", "pic.twitter", "getty", "featured image"):
        assert fragment not in cleaned
    assert "see" in cleaned and "bold" in cleaned


def test_clean_text_is_idempotent():
    once = clean_text("WASHINGTON (Reuters) - Hello <i>There</i> http://x.y")
    assert clean_text(once) == once


def test_prepare_text_equals_clean_of_build():
    assert prepare_text("A Title", "Some BODY") == clean_text(build_text("A Title", "Some BODY"))


def test_near_duplicate_key_ignores_punctuation_and_case():
    a = near_duplicate_key(clean_text("Hello, World!! 123"), 50)
    b = near_duplicate_key(clean_text("hello world 123"), 50)
    assert a == b == "helloworld123"
