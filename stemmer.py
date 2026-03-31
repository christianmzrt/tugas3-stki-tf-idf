from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

_factory = StemmerFactory()
_stemmer = _factory.create_stemmer()


def stem(word: str) -> str:
    """Stem satu kata ke bentuk dasarnya menggunakan Sastrawi."""
    return _stemmer.stem(word)