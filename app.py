import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. KONFIGURASI HALAMAN WEBSITE
# ==============================================================================
st.set_page_config(
    page_title="Prediksi Rating Ordinal - Naive Bayes & SMOTE",
    page_icon="⭐",
    layout="centered",
)


# ==============================================================================
# 2. MEMUAT ASET MODEL & KAMUS SLANG (.PKL)
# ==============================================================================
@st.cache_resource
def load_assets():
    model = joblib.load("models/model_naive_bayes_skripsi.pkl")
    slang_dict = joblib.load("models/kamus_slang_hybrid.pkl")
    return model, slang_dict


try:
    model_pipeline, kamus_slang = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"Gagal memuat model atau kamus slang. Error: {e}")


# ==============================================================================
# 3. FUNGSI PREPROCESSING
# ==============================================================================
def preprocess_text(text, slang_dictionary):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    normalized_tokens = [slang_dictionary.get(token, token) for token in tokens]
    return " ".join(normalized_tokens)


# ==============================================================================
# 4. TAMPILAN ANTARMUKA (USER INTERFACE)
# ==============================================================================
st.title("⭐ Analisis Klasifikasi Ordinal Rating E-Commerce")
st.write(
    "Prediksi Rating Bintang (1-5) menggunakan **Naive Bayes + Slang Normalization + n-gram + SMOTE + Multinomial Naive Bayes**."
)
st.markdown("---")

if assets_loaded:
    st.subheader("📝 Masukkan Teks Ulasan")

    user_input = st.text_area(
        label="Tulis ulasan produk di sini untuk diprediksi ratingnya:",
        placeholder="Contoh: kualitas produk sangat baik, pengiriman cepat dan kurir ramah bgt",
        height=100,
    )

    if st.button("Prediksi Rating Bintang", type="primary"):
        if user_input.strip() == "":
            st.warning("Mohon masukkan teks ulasan terlebih dahulu!")
        else:
            # A. Preprocessing
            cleaned_text = preprocess_text(user_input, kamus_slang)

            # B. Prediksi Rating Ordinal
            prediction = model_pipeline.predict([cleaned_text])[0]
            proba = model_pipeline.predict_proba([cleaned_text])[0]
            classes = model_pipeline.classes_

            st.markdown("---")
            st.subheader("📊 Hasil Prediksi Model")

            # Menampilkan Bintang interaktif sesuai hasil tebakan (1-5)
            stars_visual = "⭐" * int(prediction)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric(label="Rating yang Diprediksi", value=f"Rating {prediction}")
            with col2:
                st.subheader(f"{stars_visual}")

            # C. MENJELASKAN ALASAN MODEL (Fitur TF-IDF Terkuat dalam Teks)
            st.markdown("---")
            st.subheader("🔍 Penjelasan Keputusan Model (Interpretability)")
            st.write(
                "Mengapa model menebak rating tersebut? Berikut adalah kata kunci penting yang terdeteksi beserta bobot kontribusinya:"
            )

            # Ekstraksi TF-IDF dari teks input untuk melihat kata apa saja yang dominan
            try:
                # Mengambil step 'tfidf' dari pipeline model
                tfidf_vectorizer = model_pipeline.named_steps["tfidf"]
                # Transform teks yang sudah bersih ke vektor TF-IDF
                tfidf_vector = tfidf_vectorizer.transform([cleaned_text])

                # Mendapatkan kata-kata (features) dan nilai TF-IDF-nya
                feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
                non_zero_indices = tfidf_vector.nonzero()[1]
                scores = tfidf_vector.data

                if len(non_zero_indices) > 0:
                    # Membuat DataFrame kata kunci terpenting di teks tersebut
                    importance_df = pd.DataFrame(
                        {
                            "Kata Kunci": feature_names[non_zero_indices],
                            "Nilai Kontribusi (TF-IDF)": scores,
                        }
                    ).sort_values(by="Nilai Kontribusi (TF-IDF)", ascending=False)

                    st.dataframe(
                        importance_df, use_container_width=True, hide_index=True
                    )
                else:
                    st.info(
                        "Tidak ada kata kunci dominan dari kosakata kamus model yang terdeteksi di ulasan ini."
                    )
            except Exception:
                st.write(
                    "*Fitur penjelasan kata saat ini hanya mendukung visualisasi probabilitas kelas.*"
                )

            # Visualisasi Probabilitas Kelas (Keyakinan Model)
            st.write("**Tingkat Keyakinan Model untuk Tiap Rating Bintang:**")
            chart_data = pd.DataFrame(
                {
                    "Rating Target": [f"Rating {c}" for c in classes],
                    "Probabilitas (%)": [p * 100 for p in proba],
                }
            )
            st.bar_chart(
                data=chart_data,
                x="Rating Target",
                y="Probabilitas (%)",
                horizontal=True,
            )

            with st.expander("🛠️ Detail Preprocessing"):
                st.write(f"**Teks Asli:** `{user_input}`")
                st.write(
                    f"**Hasil Preprocessing & Normalisasi Slang:** `{cleaned_text}`"
                )

else:
    st.info("Aset model tidak ditemukan di direktori aktif.")
