import datetime as dt
from unittest.mock import MagicMock
import pytest

from src.news_harvester.collectors import Article
from src.news_harvester.processing.text import extract_plain_text
from src.news_harvester.processing.records import build_news_record, infer_published_datetime

# --- FIXTURES ---

LOREM_IPSUM = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum, incluyendo información sobre Yape y otras billeteras digitales.
""" * 3  # Ensure > 200 chars

@pytest.fixture
def html_easy():
    return f"""
    <html>
        <body>
            <h1>Noticia Importante sobre Yape</h1>
            <p>{LOREM_IPSUM}</p>
        </body>
    </html>
    """

@pytest.fixture
def html_difficult():
    return f"""
    <html>
        <body>
            <div id="menu">Menu Item 1 | Menu Item 2</div>
            <div class="content">
                <p>Texto irrelevante de relleno para dar volumen.</p>
                <div>Publicidad</div>
                <p>{LOREM_IPSUM}</p>
            </div>
            <footer>Copyright 2023</footer>
        </body>
    </html>
    """

@pytest.fixture
def html_impossible():
    return """
    <html>
        <body>
            <script>var x = 123;</script>
            <div>Solo contenido dinámico sin texto estático relevante.</div>
        </body>
    </html>
    """

# --- TESTS ---

def test_extract_plain_text_easy(html_easy):
    text = extract_plain_text(html_easy, keyword="Yape")
    assert "noticia importante" in text.lower() or "lorem ipsum" in text.lower()
    assert "Yape" in text
    assert len(text) > 200

def test_extract_plain_text_difficult(html_difficult):
    # Depending on the extractor used, check that boilerplate is removed
    text = extract_plain_text(html_difficult, keyword="Yape")
    assert "lorem ipsum" in text.lower()
    assert "Menu Item" not in text
    assert "Copyright" not in text

def test_extract_plain_text_impossible(html_impossible):
    text = extract_plain_text(html_impossible, keyword="Yape")
    assert text == ""

def test_infer_published_datetime():
    # Case 1: Prioritize publish_datetime
    dt1 = dt.datetime(2023, 1, 1, 10, 0, tzinfo=dt.timezone.utc)
    a1 = Article(
        url="http://example.com",
        domain="example.com",
        title="Test",
        seen_datetime=dt.datetime.now(dt.timezone.utc),
        seen_date=dt.date.today(),
        publish_datetime=dt1
    )
    assert infer_published_datetime(a1) == dt1

    # Case 2: Fallback to publish_date + seen time
    d2 = dt.date(2023, 1, 2)
    seen2 = dt.datetime(2023, 1, 5, 15, 30, tzinfo=dt.timezone.utc) # Seen later
    a2 = Article(
        url="http://example.com/2",
        domain="example.com",
        title="Test 2",
        seen_datetime=seen2,
        seen_date=seen2.date(),
        publish_date=d2
    )
    # Result should be date 2023-01-02 combined with time 15:30
    res2 = infer_published_datetime(a2)
    assert res2.date() == d2
    assert res2.time() == seen2.time()

def test_build_news_record_valid(html_easy):
    article = Article(
        url="http://example.com/valid",
        domain="example.com",
        title="Valid Title",
        seen_datetime=dt.datetime.now(dt.timezone.utc),
        seen_date=dt.date.today(),
        raw_html=html_easy
    )
    record = build_news_record(article=article, keyword="Yape")
    assert record is not None
    assert record.newspaper == "example.com"
    assert "Yape" in record.plain_text

def test_build_news_record_empty(html_impossible):
    article = Article(
        url="http://example.com/empty",
        domain="example.com",
        title="Empty Title",
        seen_datetime=dt.datetime.now(dt.timezone.utc),
        seen_date=dt.date.today(),
        raw_html=html_impossible
    )
    # Require keyword that is not present
    record = build_news_record(article=article, keyword="NonExistentKeyword")
    assert record is None
