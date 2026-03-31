# 🔎 TF-IDF Information Retrieval System

Aplikasi ini merupakan implementasi sederhana **Sistem Temu Kembali Informasi (Information Retrieval)** menggunakan metode **TF-IDF** dan **Cosine Similarity** untuk menemukan dokumen yang paling relevan berdasarkan query pengguna.

Aplikasi dikembangkan menggunakan **Streamlit** sehingga dapat dijalankan secara interaktif melalui browser.

---

## ⚙️ Teknologi yang Digunakan

- Python
- Streamlit
- Pandas
- PySastrawi

---

## 🧠 Cara Kerja Sistem

Proses pencarian dokumen dilakukan melalui tahapan berikut:

1. Pengguna memasukkan query pencarian
2. Sistem melakukan preprocessing teks:
   - Tokenisasi
   - Stopword Removal
   - Stemming Bahasa Indonesia
3. Menghitung Term Frequency (TF)
4. Menghitung Inverse Document Frequency (IDF)
5. Membentuk bobot TF-IDF
6. Menghitung Cosine Similarity
7. Menampilkan dokumen berdasarkan ranking relevansi

---

## 📂 Struktur Project

```
TUGAS3-STKI-TF-IDF/
│── .gitignore
│── app.py
│── documents.py
│── index_builder.py
│── preprocessing.py
│── stemmer.py
│── stopwords.py
│── tfidf.py
│── requirements.txt
```

### Penjelasan File

- **app.py** → Antarmuka utama aplikasi Streamlit  
- **documents.py** → Dataset dokumen  
- **index_builder.py** → Membangun indeks TF-IDF  
- **preprocessing.py** → Tokenisasi dan pembersihan teks  
- **stemmer.py** → Proses stemming menggunakan Sastrawi  
- **stopwords.py** → Stopword Bahasa Indonesia  
- **tfidf.py** → Perhitungan TF, IDF, TF-IDF, dan Similarity  
- **requirements.txt** → Dependency project  

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Install Dependency

```bash
pip install -r requirements.txt
```

### 2. Jalankan Program

```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser.

---

## 🔬 Metode yang Digunakan

### Term Frequency (TF)

Mengukur seberapa sering suatu kata muncul dalam dokumen.

```
TF(t,d) = jumlah kemunculan term / total kata dokumen
```

### Inverse Document Frequency (IDF)

Mengukur tingkat kepentingan kata terhadap seluruh dokumen.

```
IDF(t) = log(N / DF(t))
```

### TF-IDF

Pembobotan kata hasil kombinasi TF dan IDF.

```
TF-IDF = TF × IDF
```

### Cosine Similarity

Digunakan untuk menghitung kemiripan antara query dan dokumen.

Nilai similarity:
- 0 → tidak relevan
- 1 → sangat relevan

---

## 👤 Author

**Richard Christian Mozart Diazoni**  
NIM: 2405551019  
Mata Kuliah: Sistem Temu Kembali Informasi

---

## 🎯 Tujuan Proyek

Proyek ini dibuat untuk memahami implementasi dasar:

- Sistem Temu Kembali Informasi
- Pembobotan TF-IDF
- Natural Language Processing Bahasa Indonesia
- Pengembangan aplikasi pencarian dokumen berbasis web
