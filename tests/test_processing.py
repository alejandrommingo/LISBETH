from __future__ import annotations

from news_harvester.processing import extract_plain_text


def test_extract_plain_text_strips_noise_tags() -> None:
    html = """
    <html>
      <head>
        <title>Yape revolución digital</title>
        <script>console.log('ignore');</script>
        <style>.hidden {display:none;}</style>
      </head>
      <body>
        <nav>Menú</nav>
        <article>
          <h1>Yape permitió pagos rápidos</h1>
          <p>Durante la pandemia, Yapear se volvió cotidiano.</p>
          <p>El comercio peruano adoptó <strong>pagos digitales</strong>.</p>
        </article>
        <footer>Publicidad</footer>
      </body>
    </html>
    """

    text = extract_plain_text(html)

    assert "Menú" not in text
    assert "Publicidad" not in text
    assert "console.log" not in text
    assert "Yape permitió pagos rápidos" in text
    assert "pagos digitales" in text
    assert "\n" in text  # conserva saltos de línea entre párrafos


def test_extract_plain_text_removes_navigation_noise() -> None:
    html = """
    <article>
      <p>
        Texto principal con detalles suficientes para superar el umbral
        mínimo requerido y demostrar que el extractor respeta el contenido
        sustantivo de la noticia incluso cuando aparecen elementos ruidosos.
      </p>
      <button>Copiar enlace</button>
      <div>Privacy Manager</div>
      <section>
        <h2>NO TE PIERDAS</h2>
        <ul>
          <li>Animales</li>
          <li>Arequipa</li>
        </ul>
      </section>
      <section>
        <h3>Contenido de</h3>
        <p>Mag.</p>
        <p>Cargando siguiente contenido</p>
      </section>
      <p>
        Final feliz con una clausula descriptiva que también excede el límite
        mínimo y confirma que el extractor conserva los párrafos válidos de la
        noticia original.
      </p>
    </article>
    """

    text = extract_plain_text(html)

    assert "Texto principal con detalles" in text
    assert "Final feliz con una clausula" not in text
    assert "Copiar enlace" not in text
    assert "Privacy Manager" not in text
    assert "NO TE PIERDAS" not in text
    assert "Animales" not in text
    assert "Mag." not in text


def test_extract_plain_text_drops_navigation_lists() -> None:
    html = """
    <div>
      <nav>
        <ul>
          <li>Empresas</li>
          <li>Management &amp; Empleo</li>
          <li>Economía</li>
          <li>Perú</li>
          <li>Únete a El Comercio</li>
        </ul>
      </nav>
      <article>
        <h1>El futuro es hoy</h1>
        <p>
          Llegó el momento de abrazar la transformación digital con una visión
          amplia, práctica y estratégica que permita sostener los cambios.
        </p>
      </article>
    </div>
    """

    text = extract_plain_text(html)

    assert "Empresas" not in text
    assert "Management & Empleo" not in text
    assert "Únete a El Comercio" not in text
    assert "El futuro es hoy" in text
    assert "transformación digital" in text


def test_extract_plain_text_requires_keyword_in_dense_paragraph() -> None:
    html = """
    <div>
      <article>
        <p>Yape.</p>
        <p>
          Esta es una explicación larga que menciona Yape solo una vez pero no alcanza el umbral.
        </p>
      </article>
    </div>
    """

    empty = extract_plain_text(
        html,
        keyword="Yape",
        min_paragraph_chars=120,
        require_keyword=True,
    )
    assert empty == ""

    html += """
    <p>
      En medio de la campaña, el equipo de Yape presentó una nueva funcionalidad que permite a los comercios recibir pagos sin contacto
      y recopilar métricas diarias para sus cajas.
    </p>
    """

    text = extract_plain_text(
        html,
        keyword="Yape",
        min_paragraph_chars=120,
        require_keyword=True,
    )

    assert "nueva funcionalidad" in text
    assert "Yape" in text


def test_extract_plain_text_handles_empty_html() -> None:
    assert extract_plain_text("") == ""
    assert extract_plain_text("   ") == ""
