import streamlit as st
from documents import DOKUMEN
from preprocessing import preprocess
from tfidf import hitung_tf, hitung_idf, hitung_tfidf


@st.cache_resource(show_spinner=False)
def bangun_indeks():
    """
    Proses semua dokumen sekali saat aplikasi pertama dijalankan:
    1. Preprocessing (tokenisasi → stopword → stemming)
    2. Hitung TF per dokumen
    3. Hitung IDF dari seluruh koleksi
    4. Hitung TF-IDF per dokumen
    """
    docs = [dict(d) for d in DOKUMEN]

    for doc in docs:
        hasil = preprocess(doc["teks"])
        doc["tokens"] = hasil["tokens"]
        doc["nostop"]  = hasil["nostop"]
        doc["stems"]   = hasil["stems"]
        doc["tf"]      = hitung_tf(doc["stems"])

    semua_stems = [doc["stems"] for doc in docs]
    idf = hitung_idf(semua_stems)

    for doc in docs:
        doc["tfidf"] = hitung_tfidf(doc["tf"], idf)

    return docs, idf
