from __future__ import annotations

import datetime as dt

import httpx
import pytest

from news_harvester.collectors.gdelt import (
    Article,
    GDELTError,
    download_article_bodies,
    fetch_articles,
)


@pytest.fixture()
def mock_client() -> httpx.Client:
    sample_payload = {
        "articles": [
            {
                "seendate": "20200305T120000Z",
                "publishdate": "20200305",
                "publishdatetime": "20200305110000",
                "url": "https://elcomercio.pe/economia/yape-lanza-nueva-funcionalidad",
                "domain": "elcomercio.pe",
                "title": "Yape lanza nueva funcionalidad",
                "language": "Spanish",
                "sourcecountry": "Peru",
            },
            {
                "seendate": "20200305T130000Z",
                "url": "https://otro-medio.com/noticia",
                "domain": "otro-medio.com",
                "title": "Noticia fuera de Perú",
                "language": "Spanish",
                "sourcecountry": "Spain",
            },
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["query"] == "Yape sourceCountry:PE"
        assert request.url.params["maxrecords"] == "100"
        return httpx.Response(200, json=sample_payload)

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def test_fetch_articles_filters_domains(mock_client: httpx.Client) -> None:
    start = dt.datetime(2020, 3, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2020, 3, 31, 23, 59, 59, tzinfo=dt.timezone.utc)

    articles = fetch_articles(
        keyword="Yape",
        start=start,
        end=end,
        source_country="PE",
        domains=["elcomercio.pe"],
        max_records=100,
        client=mock_client,
    )

    assert len(articles) == 1
    article = articles[0]
    assert article.domain == "elcomercio.pe"
    assert article.title.startswith("Yape")
    assert article.seen_date == dt.date(2020, 3, 5)
    assert article.publish_date == dt.date(2020, 3, 5)
    assert article.publish_datetime == dt.datetime(2020, 3, 5, 11, 0, tzinfo=dt.timezone.utc)


def test_download_article_bodies_assigns_html() -> None:
    article = Article(
        title="Demo",
        url="https://example.com/demo",
        domain="example.com",
        seen_datetime=dt.datetime(2020, 3, 5, tzinfo=dt.timezone.utc),
        seen_date=dt.date(2020, 3, 5),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://example.com/demo")
        return httpx.Response(200, text="<html>ok</html>")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    download_article_bodies([article], delay_seconds=0, client=client)

    assert article.raw_html == "<html>ok</html>"


def test_fetch_articles_rejects_invalid_max_records() -> None:
    start = dt.datetime(2020, 3, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2020, 3, 2, tzinfo=dt.timezone.utc)

    with pytest.raises(ValueError):
        fetch_articles(
            keyword="test",
            start=start,
            end=end,
            source_country="PE",
            max_records=300,
        )


def test_fetch_articles_raises_on_non_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="NOTJSON")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    start = dt.datetime(2020, 3, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2020, 3, 2, tzinfo=dt.timezone.utc)

    with pytest.raises(GDELTError):
        fetch_articles(
            keyword="Yape",
            start=start,
            end=end,
            source_country="PE",
            client=client,
        )


def test_fetch_articles_skips_invalid_articles(caplog: pytest.LogCaptureFixture) -> None:
    payload = {
        "articles": [
            {
                "seendate": "20200305120000",
                "seendatetime": "20200305120000",
                "url": "https://elcomercio.pe/economia/yape-lanza-nueva-funcionalidad",
                "domain": "elcomercio.pe",
            },
            {
                "seendate": "20200305120000",
                "seendatetime": "20200305120000",
                # Falta URL -> causará error
                "domain": "elcomercio.pe",
            },
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    start = dt.datetime(2020, 3, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2020, 3, 2, tzinfo=dt.timezone.utc)

    with caplog.at_level("WARNING"):
        articles = fetch_articles(
            keyword="Yape",
            start=start,
            end=end,
            source_country="PE",
            client=client,
        )

    assert len(articles) == 1
    assert any("No fue posible parsear un artículo" in message for message in caplog.messages)
