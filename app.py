import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. KONFIGURASI HALAMAN WEBSITE
# ==============================================================================
st.set_page_config(
    page_title="Ordinal Rating Prediction - Naive Bayes & SMOTE",
    page_icon="⭐",
    layout="centered",
)

# ==============================================================================
# 2. SELEKSI BAHASA (BILINGUAL DICTIONARY)
# ==============================================================================
# Taruh opsi bahasa di sidebar
lang = st.sidebar.radio("🌐 Language / Bahasa", ["English", "Indonesia"])

# Kamus pelokalan teks UI
text_content = {
    "English": {
        "title": "⭐ Ordinal Rating Classification for E-Commerce Reviews",
        "subtitle": "Predicting 1–5 Star Ratings using Multinomial Naive Bayes and NLP Techniques.",
        "subheader_input": "📝 Enter Customer Review",
        "label_input": "Write a product review below to predict its rating:",
        "input_warning": "**Important Note:** This model is trained specifically on Indonesian text. Please input your review in **Bahasa Indonesia**.",
        "placeholder_input": "Example: kualitas produk sangat baik, pengiriman cepat dan kurir ramah bgt",
        "btn_predict": "Predict Star Rating",
        "warn_empty": "Please enter a review text first!",
        "header_result": "📊 Model Prediction Results",
        "metric_label": "Predicted Rating",
        "header_xai": "🔍 Model Decision Explanation (Interpretability)",
        "desc_xai": "Why did the model predict this rating? Here are the key terms detected in your text along with their contribution weights:",
        "col_word": "Key Term",
        "col_weight": "Contribution Value (TF-IDF)",
        "info_no_words": "No dominant key terms from the model's vocabulary were detected in this review.",
        "fallback_xai": "*Word contribution visualization is currently unavailable for this prediction.*",
        "chart_title": "**Model Confidence Level for Each Star Rating:**",
        "chart_x": "Target Rating",
        "chart_y": "Probability (%)",
        "expander_title": "🛠️ Preprocessing Details",
        "exp_raw": "Original Text",
        "exp_clean": "Normalized & Cleaned Text",
        "err_model": "Failed to load the model or slang dictionary.",
        "info_no_model": "Model assets not found in the active directory.",
    },
    "Indonesia": {
        "title": "⭐ Klasifikasi Ordinal Rating Ulasan E-Commerce",
        "subtitle": "Prediksi Rating Bintang (1–5) menggunakan Multinomial Naive Bayes dan Teknik NLP.",
        "subheader_input": "📝 Masukkan Teks Ulasan",
        "label_input": "Tulis ulasan produk di sini untuk diprediksi ratingnya:",
        "input_warning": "**Catatan Penting:** Model ini dilatih khusus menggunakan data berbahasa Indonesia. Harap masukkan ulasan dalam **Bahasa Indonesia**.",
        "placeholder_input": "Contoh: kualitas produk sangat baik, pengiriman cepat dan kurir ramah bgt",
        "btn_predict": "Prediksi Rating Bintang",
        "warn_empty": "Mohon masukkan teks ulasan terlebih dahulu!",
        "header_result": "📊 Hasil Prediksi Model",
        "metric_label": "Rating yang Diprediksi",
        "header_xai": "🔍 Penjelasan Keputusan Model (Interpretability)",
        "desc_xai": "Mengapa model menebak rating tersebut? Berikut adalah kata kunci penting yang terdeteksi beserta bobot kontribusinya:",
        "col_word": "Kata Kunci",
        "col_weight": "Nilai Kontribusi (TF-IDF)",
        "info_no_words": "Tidak ada kata kunci dominan dari kosakata kamus model yang terdeteksi di ulasan ini.",
        "fallback_xai": "*Fitur penjelasan kata saat ini hanya mendukung visualisasi probabilitas kelas.*",
        "chart_title": "**Tingkat Keyakinan Model untuk Tiap Rating Bintang:**",
        "chart_x": "Rating Target",
        "chart_y": "Probabilitas (%)",
        "expander_title": "🛠️ Detail Preprocessing",
        "exp_raw": "Teks Asli",
        "exp_clean": "Hasil Preprocessing & Normalisasi Slang",
        "err_model": "Gagal memuat model atau kamus slang.",
        "info_no_model": "Aset model tidak ditemukan di direktori aktif.",
    },
}

tx = text_content[lang]

# Ambil bahasa terpilih
tx = text_content[lang]

pipeline_title = tx["title"]


# ==============================================================================
# 3. MEMUAT ASET MODEL & KAMUS SLANG (.PKL)
# ==============================================================================
@st.cache_resource
def load_assets():
    model = joblib.load("models/model_naive_bayes_skripsi.pkl")
    slang_dict = joblib.load("resource/kamus_slang_hybrid.pkl")
    return model, slang_dict


try:
    model_pipeline, kamus_slang = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"{tx['err_model']} Error: {e}")


# ==============================================================================
# 4. FUNGSI PREPROCESSING
# ==============================================================================
def preprocess_text(text, slang_dictionary):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    normalized_tokens = [slang_dictionary.get(token, token) for token in tokens]
    return " ".join(normalized_tokens)


# ==============================================================================
# 5. TAMPILAN ANTARMUKA (USER INTERFACE)
# ==============================================================================
st.title(pipeline_title)
st.write(tx["subtitle"])
st.markdown("---")

if assets_loaded:
    st.subheader(tx["subheader_input"])
    # Menampilkan peringatan batasan bahasa model
    st.info(tx["input_warning"])

    user_input = st.text_area(
        label=tx["label_input"],
        placeholder=tx["placeholder_input"],
        height=100,
    )

    if st.button(tx["btn_predict"], type="primary"):
        if user_input.strip() == "":
            st.warning(tx["warn_empty"])
        else:
            # A. Preprocessing
            cleaned_text = preprocess_text(user_input, kamus_slang)

            # B. Prediksi Rating Ordinal
            prediction = model_pipeline.predict([cleaned_text])[0]
            proba = model_pipeline.predict_proba([cleaned_text])[0]
            classes = model_pipeline.classes_

            st.markdown("---")
            st.subheader(tx["header_result"])

            # Menampilkan Bintang interaktif sesuai hasil tebakan (1-5)
            stars_visual = "⭐" * int(prediction)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric(label=tx["metric_label"], value=f"Rating {prediction}")
            with col2:
                st.subheader(f"{stars_visual}")

            # C. MENJELASKAN ALASAN MODEL (Fitur TF-IDF Terkuat dalam Teks)
            st.markdown("---")
            st.subheader(tx["header_xai"])
            st.write(tx["desc_xai"])

            try:
                tfidf_vectorizer = model_pipeline.named_steps["tfidf"]
                tfidf_vector = tfidf_vectorizer.transform([cleaned_text])

                feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
                non_zero_indices = tfidf_vector.nonzero()[1]
                scores = tfidf_vector.data

                if len(non_zero_indices) > 0:
                    importance_df = pd.DataFrame(
                        {
                            tx["col_word"]: feature_names[non_zero_indices],
                            tx["col_weight"]: scores,
                        }
                    ).sort_values(by=tx["col_weight"], ascending=False)

                    st.dataframe(
                        importance_df, use_container_width=True, hide_index=True
                    )
                else:
                    st.info(tx["info_no_words"])
            except Exception:
                st.write(tx["fallback_xai"])

            # Visualisasi Probabilitas Kelas (Keyakinan Model)
            st.write(tx["chart_title"])
            chart_data = pd.DataFrame(
                {
                    tx["chart_x"]: [f"Rating {c}" for c in classes],
                    tx["chart_y"]: [p * 100 for p in proba],
                }
            )
            st.bar_chart(
                data=chart_data,
                x=tx["chart_x"],
                y=tx["chart_y"],
                horizontal=True,
            )

            with st.expander(tx["expander_title"]):
                st.write(f"**{tx['exp_raw']}:** `{user_input}`")
                st.write(f"**{tx['exp_clean']}:** `{cleaned_text}`")

else:
    st.info(tx["info_no_model"])
