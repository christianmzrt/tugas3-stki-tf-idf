from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ambil daftar stopword dari library
factory = StopWordRemoverFactory()
STOPWORDS = set(factory.get_stop_words())

def remove_stopwords(tokens):
    return [word for word in tokens if word not in STOPWORDS]