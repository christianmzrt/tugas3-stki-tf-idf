import streamlit as st
import pandas as pd
from index_builder import bangun_indeks
from preprocessing import preprocess
from tfidf import hitung_tf, hitung_tfidf, cosine_similarity
from stopwords import STOPWORDS

# ── Konfigurasi Halaman ─────────────────────────────
st.set_page_config(
    page_title="IR Engine | TF-IDF",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stExpander {
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
        background: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ── Load Data ───────────────────────────────────────
docs, idf = bangun_indeks()

# ── Header Utama ────────────────────────────────────
col_head, col_stats = st.columns([2, 1])
with col_head:
    st.title("🔬 Information Retrieval System")
    st.markdown("Implementasi algoritma **TF-IDF** & **Cosine Similarity** untuk pencarian dokumen.")
    st.caption("Tokenisasi → Stopword Removal → Stemming → TF → IDF → TF-IDF → Query")

with col_stats:
    c1, c2 = st.columns(2)
    c1.metric("Total Dokumen", len(docs))
    c2.metric("Vocabulary", f"{len(idf)}")

# ── Navigasi Tabs ───────────────────────────────────
tabs = st.tabs([
    "📄 Koleksi Dokumen",
    "⚙️ Preprocessing",
    "📊 TF",
    "📈 IDF",
    "🔢 TF-IDF",
    "🔍 Mesin Pencari",
])

# ══════════════════════════════════════════════════════
# TAB 1 — DOKUMEN
# ══════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("📚 Koleksi Dokumen")
    st.write(f"Total **{len(docs)} dokumen** Bahasa Indonesia.")
    st.divider()

    for i in range(0, len(docs), 2):
        col1, col2 = st.columns(2)
        for j, col in enumerate([col1, col2]):
            if i + j < len(docs):
                doc = docs[i + j]
                with col.container():
                    st.markdown(f"### {doc['id']}")
                    st.markdown(f"**{doc['judul']}**")
                    st.caption(doc["teks"][:150] + "...")
                    with st.expander("Baca Selengkapnya"):
                        st.write(doc["teks"])
                        ic1, ic2, ic3 = st.columns(3)
                        ic1.metric("Jumlah Token", len(doc["tokens"]))
                        ic2.metric("Setelah Stopword", len(doc["nostop"]))
                        ic3.metric("Term Unik (stem)", len(set(doc["stems"])))
                    st.divider()

# ══════════════════════════════════════════════════════
# TAB 2 — PREPROCESSING
# ══════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("⚙️ Preprocessing")
    st.write("Tiga tahap: **Tokenisasi → Hapus Stopword → Stemming**")
    st.divider()

    pilih = st.selectbox(
        "Pilih dokumen:",
        [f"{d['id']} — {d['judul']}" for d in docs],
        key="pre_select"
    )
    doc = next(d for d in docs if d["id"] == pilih.split(" — ")[0])

    st.subheader("Teks Asli")
    st.info(doc["teks"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader(f"1. Tokenisasi ({len(doc['tokens'])} token)")
        st.caption("Huruf kecil, hapus tanda baca, minimal 3 karakter")
        st.write(doc["tokens"])

    with col2:
        dihapus = len(doc["tokens"]) - len(doc["nostop"])
        st.subheader(f"2. Hapus Stopword ({len(doc['nostop'])} kata)")
        st.caption(f"{dihapus} stopword dihapus")
        st.write(doc["nostop"])

    with col3:
        st.subheader(f"3. Stemming ({len(doc['stems'])} stem)")
        st.caption("Imbuhan dipotong ke bentuk dasar menggunakan PySastrawi")
        st.write(doc["stems"])

    st.divider()
    st.subheader("Perbandingan Sebelum & Sesudah Stemming")

    pasangan      = list(zip(doc["nostop"], doc["stems"]))
    berubah       = [(a, b) for a, b in pasangan if a != b]
    tidak_berubah = [(a, b) for a, b in pasangan if a == b]

    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Kata berubah ({len(berubah)}):**")
        with st.expander("Lihat daftar"):
            for asli, hasil in berubah:
                st.write(f"`{asli}` → :green[**`{hasil}`**]")
    with col_b:
        st.write(f"**Tidak berubah ({len(tidak_berubah)}):**")
        with st.expander("Lihat daftar"):
            for asli, _ in tidak_berubah:
                st.write(f"`{asli}`")

    st.divider()
    st.write("**Visualisasi Pipeline:**")
    st.code(f"ORIGINAL : {doc['teks'][:100]}...", language=None)
    st.code(f"TOKENS   : {', '.join(doc['tokens'][:10])}...", language=None)
    st.code(f"NOSTOP   : {', '.join(doc['nostop'][:10])}...", language=None)
    st.code(f"STEMS    : {', '.join(doc['stems'][:10])}...", language=None)

# ══════════════════════════════════════════════════════
# TAB 3 — TF
# ══════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("📊 Term Frequency (TF)")
    st.latex(
        r"TF(t,d) = \frac{\text{jumlah kemunculan } t \text{ dalam } d}"
        r"{\text{total kata dalam } d}"
    )
    st.divider()

    pilih_tf = st.selectbox(
        "Pilih dokumen:",
        [f"{d['id']} — {d['judul']}" for d in docs],
        key="tf_select"
    )
    doc_tf = next(d for d in docs if d["id"] == pilih_tf.split(" — ")[0])

    st.subheader(f"Nilai TF — {doc_tf['id']}: {doc_tf['judul']}")
    st.caption(
        f"Total kata (setelah stemming): {len(doc_tf['stems'])} | "
        f"Term unik: {len(doc_tf['tf'])}"
    )

    tf_sorted = sorted(doc_tf["tf"].items(), key=lambda x: x[1], reverse=True)
    total     = len(doc_tf["stems"])

    rows = []
    for term, val in tf_sorted:
        jumlah = doc_tf["stems"].count(term)
        rows.append({
            "Term (stem)": term,
            "Muncul":      jumlah,
            "TF":          round(val, 4),
            "Perhitungan": f"{jumlah} / {total} = {val:.4f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════
# TAB 4 — IDF
# ══════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("📈 Inverse Document Frequency (IDF)")
    st.latex(r"IDF(t) = \log_{10}\left(\frac{N}{DF(t)}\right)")
    st.caption(f"N = {len(docs)} dokumen")
    st.divider()

    # Hitung Document Frequency tiap term
    df_count = {}
    for doc in docs:
        for term in set(doc["stems"]):
            df_count[term] = df_count.get(term, 0) + 1

    idf_sorted = sorted(idf.items(), key=lambda x: x[1], reverse=True)

    col_idf1, col_idf2 = st.columns(2)

    with col_idf1:
        st.info("**Term Paling Langka — IDF Tertinggi (spesifik)**")
        rows_high = []
        for term, val in idf_sorted[:15]:
            df_val = df_count.get(term, 1)
            rows_high.append({
                "Term": term,
                "DF": df_val,
                "Perhitungan": f"log({len(docs)}/{df_val})",
                "IDF": round(val, 4),
            })
        st.dataframe(pd.DataFrame(rows_high), use_container_width=True, hide_index=True)

    with col_idf2:
        st.warning("**Term Paling Umum — IDF Terendah (generik)**")
        rows_low = []
        for term, val in list(reversed(idf_sorted))[:15]:
            df_val = df_count.get(term, 1)
            rows_low.append({
                "Term": term,
                "DF": df_val,
                "Perhitungan": f"log({len(docs)}/{df_val})",
                "IDF": round(val, 4),
            })
        st.dataframe(pd.DataFrame(rows_low), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔎 Cari Nilai IDF Satu Kata")
    cari = st.text_input("Masukkan kata (bentuk stem):", key="idf_cari")
    if cari.strip():
        kata = cari.strip().lower()
        if kata in idf:
            df_val = df_count[kata]
            st.success(
                f"IDF(**`{kata}`**) = log({len(docs)}/{df_val}) = **{idf[kata]:.4f}**"
            )
            st.write("📂 Ditemukan di dokumen:")
            for doc in docs:
                if kata in doc["stems"]:
                    st.write(f"- {doc['id']}: {doc['judul']}")
        else:
            st.warning(f"Kata `{kata}` tidak ada dalam vocabulary.")

# ══════════════════════════════════════════════════════
# TAB 5 — TF-IDF
# ══════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("🔢 TF-IDF")
    st.latex(r"TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)")
    st.divider()

    pilih_tfidf = st.selectbox(
        "Pilih dokumen:",
        [f"{d['id']} — {d['judul']}" for d in docs],
        key="tfidf_select"
    )
    doc_tfidf = next(d for d in docs if d["id"] == pilih_tfidf.split(" — ")[0])

    tfidf_sorted = sorted(
        doc_tfidf["tfidf"].items(), key=lambda x: x[1], reverse=True
    )
    st.subheader(f"Bobot TF-IDF — {doc_tfidf['id']}: {doc_tfidf['judul']}")

    rows = []
    for term, val in tfidf_sorted:
        tf_val  = doc_tfidf["tf"].get(term, 0)
        idf_val = idf.get(term, 0)
        rows.append({
            "Term (stem)": term,
            "TF":          round(tf_val, 4),
            "IDF":         round(idf_val, 4),
            "TF-IDF":      round(val, 6),
            "Perhitungan": f"{tf_val:.4f} × {idf_val:.4f} = {val:.6f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════
# TAB 6 — QUERY / PENCARIAN
# ══════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("🔍 Mesin Pencari")
    st.write(
        "Masukkan kata kunci. Query diproses dengan pipeline yang sama, "
        "lalu dihitung **Cosine Similarity** dengan setiap dokumen."
    )
    st.divider()

    query = st.text_input(
        "Apa yang ingin Anda cari?",
        value="pengembangan aplikasi web mobile",
        placeholder="Contoh: pengembangan aplikasi web mobile"
    )

    if query.strip():
        hasil_pre = preprocess(query)
        q_stems   = hasil_pre["stems"]

        with st.expander("🔬 Lihat proses preprocessing query"):
            qc1, qc2, qc3 = st.columns(3)
            qc1.write(f"**Tokenisasi:**\n\n{hasil_pre['tokens']}")
            qc2.write(f"**Setelah Stopword:**\n\n{hasil_pre['nostop']}")
            qc3.write(f"**Setelah Stemming:**\n\n{q_stems}")

        if not q_stems:
            st.error(
                "⚠️ Query tidak menghasilkan kata yang bisa dicari. "
                "Coba kata yang lebih spesifik."
            )
            st.stop()

        q_tf    = hitung_tf(q_stems)
        q_tfidf = hitung_tfidf(q_tf, idf)

        # Hitung Cosine Similarity untuk semua dokumen
        hasil = []
        for doc in docs:
            sim        = cosine_similarity(q_tfidf, doc["tfidf"])
            kata_cocok = [t for t in q_stems if t in doc["stems"]]
            hasil.append((doc, sim, kata_cocok))
        hasil.sort(key=lambda x: x[1], reverse=True)

        n_relevan = sum(1 for _, sim, _ in hasil if sim > 0)
        st.success(f"✅ Ditemukan **{n_relevan} dokumen** relevan.")

        for rank, (doc, sim, kata_cocok) in enumerate(hasil, 1):
            if sim <= 0:
                continue

            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
            with st.container():
                rc1, rc2 = st.columns([0.85, 0.15])
                with rc1:
                    st.markdown(f"#### {medal} {doc['judul']}")
                    st.caption(
                        f"ID: {doc['id']} | "
                        f"Kata Cocok: :blue[{', '.join(kata_cocok) if kata_cocok else '-'}]"
                    )
                    st.write(doc["teks"])
                with rc2:
                    st.metric("Similarity", f"{sim:.4f}")
                st.divider()