"""Dashboard Streamlit - Klasifikasi Hujan Harian Kolkata (XGBoost).
Jalankan:  streamlit run app.py
Pastikan file kolkata_weather_jan_june_2026.csv dan Nama_Test1_KlasifikasiXGBoost.py berada di folder yang sama.
"""
import pandas as pd
import streamlit as st

import Nama_Test1_KlasifikasiXGBoost as pipe

st.set_page_config(page_title="Klasifikasi Hujan Kolkata - XGBoost", layout="wide")


@st.cache_data
def run_pipeline():
    df = pipe.load_data()
    d = pipe.build_features(df)
    X_train, X_test, y_train, y_test, scaler = pipe.preprocess(d)
    model = pipe.train_model(X_train, y_train)
    metrics, y_pred, y_proba = pipe.evaluate(model, X_test, y_test)
    return d, model, scaler, metrics, y_test, y_pred, y_proba


d, model, scaler, metrics, y_test, y_pred, y_proba = run_pipeline()

# ---- Judul & deskripsi
st.title("Klasifikasi Hujan Harian Kota Kolkata dengan XGBoost")
st.write(
    "Proyek ini memprediksi apakah suatu hari **hujan** (curah hujan harian >= 1 mm) atau "
    "**tidak hujan** berdasarkan suhu, angin, durasi sinar matahari, dan fitur kalender. "
    "Data: Open-Meteo, 30 Juni 2021 - 30 Juni 2026. Kolom curah hujan dan weather_code "
    "tidak dipakai sebagai fitur untuk mencegah data leakage."
)

# ---- Ringkasan dataset
st.header("Ringkasan Dataset")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah data", f"{len(d):,}")
c2.metric("Jumlah fitur", len(pipe.FEATURES))
c3.metric("Hari hujan", f"{(d.hujan == 1).sum():,} ({(d.hujan == 1).mean():.1%})")
c4.metric("Hari tidak hujan", f"{(d.hujan == 0).sum():,} ({(d.hujan == 0).mean():.1%})")
st.dataframe(d[["time"] + pipe.FEATURES + ["hujan"]].head(20), width="stretch")

# ---- EDA
st.header("Eksplorasi Data (EDA)")
e1, e2 = st.columns([1, 2])
e1.pyplot(pipe.fig_class_distribution(d))
e2.pyplot(pipe.fig_heatmap(d))
st.pyplot(pipe.fig_histograms(d))

# ---- Performa model
st.header("Performa Model")
cols = st.columns(5)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")
st.table(pd.DataFrame(metrics, index=["Nilai"]).T.round(4))

# ---- Visualisasi hasil
st.header("Visualisasi Hasil")
v1, v2 = st.columns(2)
v1.pyplot(pipe.fig_confusion(y_test, y_pred))
v2.pyplot(pipe.fig_roc(y_test, y_proba))
st.pyplot(pipe.fig_importance(model))

# ---- Form prediksi manual
st.header("Coba Prediksi Manual")
with st.form("form_prediksi"):
    a, b, c = st.columns(3)
    t_mean = a.number_input("Suhu rata-rata (°C)", 10.0, 45.0, 28.0, 0.1)
    t_max = a.number_input("Suhu maksimum (°C)", 10.0, 50.0, 32.0, 0.1)
    t_min = a.number_input("Suhu minimum (°C)", 5.0, 40.0, 25.0, 0.1)
    w_speed = b.number_input("Kecepatan angin maks (km/jam)", 0.0, 80.0, 14.0, 0.1)
    w_gust = b.number_input("Hembusan angin maks (km/jam)", 0.0, 120.0, 31.0, 0.1)
    w_dir = b.number_input("Arah angin dominan (°)", 0, 360, 190)
    sun = c.number_input("Durasi sinar matahari (jam)", 0.0, 14.0, 9.0, 0.1)
    tanggal = c.date_input("Tanggal", pd.Timestamp("2026-07-15"))
    daylight = c.number_input("Durasi siang (jam)", 9.0, 15.0, 13.0, 0.1)
    submit = st.form_submit_button("Prediksi")

if submit:
    tgl = pd.Timestamp(tanggal)
    row = pd.DataFrame([{
        "temperature_2m_mean": t_mean, "temperature_2m_max": t_max, "temperature_2m_min": t_min,
        "wind_speed_10m_max": w_speed, "wind_gusts_10m_max": w_gust,
        "wind_direction_10m_dominant": w_dir, "sunshine_duration": sun * 3600,
        "bulan": tgl.month, "hari_dalam_tahun": tgl.dayofyear,
        "rentang_suhu": t_max - t_min, "durasi_siang_jam": daylight,
    }])[pipe.FEATURES]
    proba = model.predict_proba(scaler.transform(row))[0, 1]
    if proba >= 0.5:
        st.success(f"Prediksi: **HUJAN** (probabilitas {proba:.1%})")
    else:
        st.info(f"Prediksi: **TIDAK HUJAN** (probabilitas hujan {proba:.1%})")
