from __future__ import annotations

import datetime as dt

from news_harvester.collectors.gdelt import Article
from news_harvester.processing.records import (
    build_news_record,
    infer_published_datetime,
)


def make_article(**overrides):
    base = {
        "title": "Yape impulsa pagos digitales",
        "url": "https://elcomercio.pe/economia/yape-impulsa-pagos-digitales",
        "domain": "elcomercio.pe",
        "seen_datetime": dt.datetime(2020, 3, 10, 15, 0, tzinfo=dt.timezone.utc),
        "seen_date": dt.date(2020, 3, 10),
        "language": "Spanish",
        "source_country": "PER",
        "publish_datetime": dt.datetime(2020, 3, 9, 8, 30, tzinfo=dt.timezone.utc),
    }
    base.update(overrides)
    return Article(**base)


def test_infer_published_datetime_prefers_publish_datetime():
    article = make_article()
    result = infer_published_datetime(article)
    assert result == dt.datetime(2020, 3, 9, 8, 30, tzinfo=dt.timezone.utc)


def test_infer_published_datetime_falls_back_to_publish_date():
    article = make_article(publish_datetime=None, publish_date=dt.date(2020, 3, 9))
    result = infer_published_datetime(article)
    expected = dt.datetime(2020, 3, 9, 15, 0, tzinfo=dt.timezone.utc)
    assert result == expected


def test_infer_published_datetime_uses_seen_datetime_last():
    article = make_article(publish_datetime=None, publish_date=None)
    result = infer_published_datetime(article)
    assert result == article.seen_datetime


def test_build_news_record_generates_plain_text():
    article = make_article()
    html = """
    <article>
      <h1>Yape impulsa pagos digitales</h1>
            <p>
                Los usuarios pueden yapear a cualquier comercio y además reciben información
                detallada sobre cada transacción, lo que permite al BCP estudiar en profundidad
                la adopción de pagos móviles durante el estado de emergencia en todo el país.
            </p>
      <script>alert('ignore');</script>
    </article>
    """

    record = build_news_record(article=article, keyword="Yape", html=html)

    assert record is not None
    assert record.title == article.title
    assert record.newspaper == "elcomercio.pe"
    assert record.keyword == "Yape"
    assert "pagos móviles" in record.plain_text.lower()
    assert record.published_at == dt.datetime(2020, 3, 9, 8, 30, tzinfo=dt.timezone.utc)
    assert record.published_date == dt.date(2020, 3, 9)
    assert str(record.url) == article.url


def test_build_news_record_returns_none_if_keyword_lacks_dense_paragraph() -> None:
    article = make_article()
    html = """
        <article>
            <p>Yape es tendencia.</p>
            <p>Más detalles en breve.</p>
        </article>
        """

    record = build_news_record(article=article, keyword="Yape", html=html)

    assert record is not None
    assert record.plain_text == "Yape es tendencia."


def test_build_news_record_returns_none_if_keyword_absent() -> None:
    article = make_article()
    html = """
        <article>
            <p>
                Los comercios digitales adoptaron pagos sin contacto y desarrollaron campañas
                educativas para sus usuarios, pero no se mencionan billeteras específicas.
            </p>
        </article>
        """

    record = build_news_record(article=article, keyword="Yape", html=html)

    assert record is None
