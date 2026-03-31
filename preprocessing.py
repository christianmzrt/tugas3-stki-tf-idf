import re
from stopwords import STOPWORDS
from stemmer import stem


def tokenisasi(teks: str) -> list:
    """
    Ubah teks mentah menjadi daftar token:
    - Huruf kecil semua
    - Hapus angka dan tanda baca
    - Buang token kurang dari 3 karakter
    """
    teks   = teks.lower()
    teks   = re.sub(r"[^a-z\s]", " ", teks)
    return [t for t in teks.split() if len(t) >= 3]


def hapus_stopword(tokens: list) -> list:
    """Buang kata-kata yang ada di daftar stopword."""
    return [t for t in tokens if t not in STOPWORDS]


def stemming(tokens: list) -> list:
    """Potong imbuhan setiap token ke bentuk dasar menggunakan PySastrawi."""
    return [stem(t) for t in tokens]


def preprocess(teks: str) -> dict:
    """
    Jalankan seluruh pipeline dan kembalikan hasil tiap tahap.

    Return:
        {
            "tokens": [...],   # hasil tokenisasi
            "nostop": [...],   # setelah hapus stopword
            "stems":  [...],   # setelah stemming
        }
    """
    tokens = tokenisasi(teks)
    nostop = hapus_stopword(tokens)
    stems  = stemming(nostop)
    return {"tokens": tokens, "nostop": nostop, "stems": stems}