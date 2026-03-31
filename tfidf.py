import math
from collections import Counter


def hitung_tf(stems: list) -> dict:
    """
    TF(t, d) = jumlah kemunculan t dalam d / total kata dalam d
    """
    if not stems:
        return {}
    total = len(stems)
    return {term: count / total for term, count in Counter(stems).items()}


def hitung_idf(semua_stems: list) -> dict:
    """
    IDF(t) = log10(N / DF(t))
    N    = total dokumen
    DF(t) = jumlah dokumen yang mengandung t
    """
    N = len(semua_stems)
    df = {}
    for stems in semua_stems:
        for term in set(stems):
            df[term] = df.get(term, 0) + 1
    return {term: math.log10(N / count) for term, count in df.items()}


def hitung_tfidf(tf: dict, idf: dict) -> dict:
    """
    TF-IDF(t, d) = TF(t, d) x IDF(t)
    """
    return {term: tf_val * idf.get(term, 0) for term, tf_val in tf.items()}


def cosine_similarity(vec_q: dict, vec_d: dict) -> float:
    """
    Cosine Similarity = (q · d) / (|q| x |d|)
    Nilai 0.0 (tidak relevan) sampai 1.0 (sangat relevan)
    """
    dot = sum(vec_q.get(t, 0) * vec_d.get(t, 0) for t in vec_q)
    mag_q = math.sqrt(sum(v ** 2 for v in vec_q.values()))
    mag_d = math.sqrt(sum(v ** 2 for v in vec_d.values()))
    if mag_q == 0 or mag_d == 0:
        return 0.0
    return dot / (mag_q * mag_d)
