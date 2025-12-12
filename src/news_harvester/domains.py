"""Lista maestra de medios peruanos para filtrado en GDELT."""

PERUVIAN_MEDIA = {
    # Diarios Nacionales (Grupo El Comercio)
    "elcomercio": "elcomercio.pe",
    "gestion": "gestion.pe",
    "peru21": "peru21.pe",
    "trome": "trome.com",  # A veces trome.pe redirige
    "ojo": "ojo.pe",
    "correo": "diariocorreo.pe",
    "depor": "depor.com",
    "bocon": "elbocon.pe",
    # Diarios Nacionales (Otros)
    "larepublica": "larepublica.pe",
    "libero": "libero.pe",
    "popular": "elpopular.pe",
    "wapa": "wapa.pe",
    "expreso": "expreso.com.pe",
    "larazon": "larazon.pe",
    "diariouno": "diariouno.pe",
    "publimetro": "publimetro.pe",
    "elperuano": "elperuano.pe",  # Diario oficial
    # Agencias de Noticias
    "andina": "andina.pe",
    # Radio y TV
    "rpp": "rpp.pe",
    "exitosa": "exitosanoticias.pe",
    "americatv": "americatv.com.pe",
    "canaln": "canaln.pe",
    "panamericana": "panamericana.pe",
    "latina": "latina.pe",
    "atv": "atv.pe",
    "willax": "willax.pe",  # A veces .tv
    # Medios Digitales / Investigación
    "ojopublico": "ojo-publico.com",
    "idl": "idl-reporteros.pe",
    "convoca": "convoca.pe",
    "saludconlupa": "saludconlupa.com",
    "utero": "utero.pe",
    "lamula": "lamula.pe",
    "wayka": "wayka.pe",
    "sudaca": "sudaca.pe",
    # Regionales (Ejemplos representativos)
    "elbuho": "elbuho.pe",  # Arequipa
    "laindustria": "laindustria.pe",  # Trujillo/Chiclayo
    "eltiempo": "eltiempo.pe",  # Piura
    "diarioahora": "diarioahora.pe",  # Huánuco/Amazonas
    "prensaregional": "prensaregional.pe",  # Moquegua
    "losandes": "losandes.com.pe",  # Puno
    "diariocambio": "diariocambio.com.pe",
    "diariosinfronteras": "diariosinfronteras.com.pe",
}

DOMAIN_SELECTORS = {
    "elcomercio.pe": ["div#contenido", "div[itemprop='articleBody']"],
    "larepublica.pe": ["div.story-content", "div.content-body"],
    "gestion.pe": ["div.story-contents", "section.article-body"],
    "trome.com": ["div.story-content"],   # Trome usa layout similar a Depor/ElComercio moderno a veces
    "peru21.pe": ["div.story-contents", "div.article-body"],
    "diariocorreo.pe": ["div.story-contents", "div[itemprop='articleBody']"],
}

MEDIA_RSS_FEEDS = {
    "elcomercio": "https://elcomercio.pe/arc/outboundfeeds/rss/",
    "larepublica": "https://larepublica.pe/rss/",
    "gestion": "https://gestion.pe/arc/outboundfeeds/rss/",
    "peru21": "https://peru21.pe/arc/outboundfeeds/rss/",
    "rpp": "https://rpp.pe/feed",
    "trome": "https://trome.com/arc/outboundfeeds/rss/",
    "depor": "https://depor.com/arc/outboundfeeds/rss/",
    "correo": "https://diariocorreo.pe/arc/outboundfeeds/rss/",
}
