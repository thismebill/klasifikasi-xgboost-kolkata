"""
Dashboard Klasifikasi Emosi Wajah (CNN) - Streamlit
Letakkan app.py sejajar dengan folder `output/` hasil notebook.
Jalankan: streamlit run app.py
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image

try:  # opsional: untuk kamera real-time
    import av
    from streamlit_webrtc import webrtc_streamer
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

st.set_page_config(page_title="Emotion CNN Dashboard", page_icon="🙂", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
MODEL_PATH = os.path.join(OUT, "final_emotion_cnn.keras")
IMG_SIZE = (48, 48)


def load_json(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


required = ["final_emotion_cnn.keras", "class_names.json", "metrics.json",
            "history.json", "dataset_info.json"]
missing = [r for r in required if not os.path.exists(os.path.join(OUT, r))]
if missing:
    st.error(f"File ini belum ada di folder `output/`: {', '.join(missing)}. "
             "Jalankan notebook sampai selesai lalu salin folder output ke sini.")
    st.stop()

class_names = load_json("class_names.json")
metrics = load_json("metrics.json")
history = load_json("history.json")
info = load_json("dataset_info.json")


@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


def predict(img: Image.Image):
    arr = np.array(img.convert("L").resize(IMG_SIZE), dtype="float32")[..., None]  # 0-255
    probs = model.predict(arr[None], verbose=0)[0]
    return arr, probs


def gradcam(arr, class_idx):
    grad_model = tf.keras.Model(model.inputs, [model.get_layer("last_conv").output, model.output])
    with tf.GradientTape() as tape:
        conv, preds = grad_model(arr[None], training=False)
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv)
    w = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.maximum(tf.reduce_sum(conv[0] * w, axis=-1), 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return tf.image.resize(cam[..., None], IMG_SIZE).numpy().squeeze()


# ---------------- Deteksi wajah (kamera) ----------------
MAX_FACES = 1
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def predict_face(gray_face):
    """gray_face: crop wajah grayscale (uint8). Mengembalikan probabilitas tiap kelas."""
    face = cv2.resize(gray_face, IMG_SIZE).astype("float32")[..., None]  # skala 0-255, Rescaling ada di model
    return model(face[None], training=False).numpy()[0]


def annotate(bgr):
    """Deteksi wajah di frame BGR, gambar kotak + label emosi. Return (frame, list probabilitas)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Deteksi di versi kecil (lebih cepat untuk video HD), hasilnya dipetakan balik ke resolusi asli
    h0, w0 = gray.shape
    scale = min(1.0, 640 / w0)
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray
    faces = FACE_CASCADE.detectMultiScale(small, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))
    faces = [(int(x / scale), int(y / scale), int(w / scale), int(h / scale)) for (x, y, w, h) in faces]
    # Haar Cascade kadang memberi kotak palsu. Simpan wajah terbesar saja (1 orang di depan kamera).
    # Ubah MAX_FACES kalau mau banyak wajah.
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:MAX_FACES]

    thick = max(2, w0 // 400)
    font_scale = max(0.7, w0 / 1000)
    results = []
    for (x, y, w, h) in faces:
        probs = predict_face(gray[y:y + h, x:x + w])  # crop dari gambar resolusi penuh
        top = int(np.argmax(probs))
        label = f"{class_names[top]} {probs[top] * 100:.0f}%"
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), thick)
        cv2.putText(bgr, label, (x, max(y - 10, 24)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thick)
        results.append(probs)
    return bgr, results


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img, _ = annotate(img)
    return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- Header ----------------
st.title("🙂 Klasifikasi Emosi Wajah dengan CNN")
st.write("Dashboard hasil eksperimen Convolutional Neural Network pada dataset "
         "Facial Emotion Recognition (Kaggle). Model dilatih pada citra grayscale 48×48 piksel.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dataset", "📈 Pelatihan", "✅ Evaluasi", "🔍 Prediksi Gambar", "📷 Kamera"])

# ---------------- Tab 1: Dataset ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah kelas", len(class_names))
    c2.metric("Total gambar", f"{info['total']:,}")
    c3.metric("Data latih", f"{info['n_train']:,}")
    c4.metric("Data validasi", f"{info['n_val']:,}")
    pct_train = info["n_train"] / (info["n_train"] + info["n_val"]) * 100
    st.caption(f"Proporsi latih : validasi = {pct_train:.0f}% : {100 - pct_train:.0f}%  |  "
               f"Rasio kelas terbesar:terkecil = {info['imbalance_ratio']:.1f}×")

    st.subheader("Distribusi gambar per kelas")
    df_counts = pd.DataFrame({"Kelas": list(info["counts"].keys()),
                              "Jumlah": list(info["counts"].values())}).set_index("Kelas")
    st.bar_chart(df_counts)

    sample = os.path.join(OUT, "1_sample_images.png")
    if os.path.exists(sample):
        st.subheader("Contoh gambar tiap kelas")
        st.image(sample)

# ---------------- Tab 2: Pelatihan ----------------
with tab2:
    st.subheader("Grafik akurasi dan loss")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, (m, t) in zip(ax, [("accuracy", "Accuracy"), ("loss", "Loss")]):
        a.plot(history[m], "o-", label=f"Train {t}")
        a.plot(history[f"val_{m}"], "s-", label=f"Val {t}")
        a.set_title(f"Training vs Validation {t}")
        a.set_xlabel("Epoch")
        a.legend()
        a.grid(alpha=.3)
    plt.tight_layout()
    st.pyplot(fig)
    st.write(f"Jumlah epoch berjalan: **{len(history['loss'])}** | "
             f"Val accuracy terbaik: **{max(history['val_accuracy']) * 100:.2f}%**")

# ---------------- Tab 3: Evaluasi ----------------
with tab3:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    m2.metric("Precision (macro)", f"{metrics['macro']['precision']:.3f}")
    m3.metric("Recall (macro)", f"{metrics['macro']['recall']:.3f}")
    m4.metric("F1-score (macro)", f"{metrics['macro']['f1']:.3f}")

    st.subheader("Laporan per kelas")
    rows = []
    for c in class_names:
        r = metrics["per_class"][c]
        rows.append({"Kelas": c, "Precision": r["precision"], "Recall": r["recall"],
                     "F1-score": r["f1-score"], "Support": int(r["support"])})
    df_rep = pd.DataFrame(rows).set_index("Kelas")
    st.dataframe(df_rep.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}),
                 use_container_width=True)
    st.caption(f"Kelas dengan F1 terendah (paling sulit dikenali): **{df_rep['F1-score'].idxmin()}**")

    st.subheader("Confusion matrix")
    cm = np.array(metrics["confusion_matrix"])
    cm_pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, label="Proporsi")
    idx = range(len(class_names))
    ax.set_xticks(list(idx))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(list(idx))
    ax.set_yticklabels(class_names)
    for i in idx:
        for j in idx:
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm_pct[i, j] > .5 else "black")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Kelas sebenarnya")
    plt.tight_layout()
    st.pyplot(fig)

    for fname, title in [("4_sample_predictions.png", "Contoh prediksi pada data validasi"),
                         ("5_gradcam.png", "Feature importance (Grad-CAM)")]:
        p = os.path.join(OUT, fname)
        if os.path.exists(p):
            st.subheader(title)
            st.image(p)

# ---------------- Tab 4: Prediksi ----------------
with tab4:
    st.write("Upload foto wajah (sebaiknya satu wajah, tampak depan, pencahayaan cukup).")
    file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png", "webp"])
    if file is not None:
        image = Image.open(file)
        arr, probs = predict(image)
        order = np.argsort(probs)[::-1]
        top = int(order[0])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Gambar input")
        with col2:
            cam = gradcam(arr, top)
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(arr.squeeze(), cmap="gray")
            ax.imshow(cam, cmap="jet", alpha=.45)
            ax.axis("off")
            st.pyplot(fig)
            st.caption("Grad-CAM: area yang paling memengaruhi prediksi")
        with col3:
            st.success(f"Emosi terdeteksi: **{class_names[top].upper()}** ({probs[top] * 100:.1f}%)")
            if probs[top] < 0.5:
                st.warning("Keyakinan rendah, coba foto yang lebih jelas.")

        st.subheader("Probabilitas semua kelas")
        st.bar_chart(pd.DataFrame({"Probabilitas": probs}, index=class_names))
    else:
        st.info("Belum ada gambar. Upload foto untuk mulai prediksi.")

# ---------------- Tab 5: Kamera ----------------
with tab5:
    st.write("Arahkan wajah ke kamera. Wajah dideteksi otomatis, lalu emosinya diprediksi oleh model CNN.")
    st.caption("Tips: wajah menghadap depan, pencahayaan cukup, dan tidak terlalu jauh dari kamera.")
    mode = st.radio("Mode kamera", ["🎥 Real-time", "📸 Ambil foto"], horizontal=True)

    if mode == "📸 Ambil foto":
        shot = st.camera_input("Ambil foto wajah")
        if shot is not None:
            rgb = np.array(Image.open(shot).convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            annotated, results = annotate(bgr.copy())
            if not results:
                st.warning("Wajah tidak terdeteksi. Coba lebih dekat, hadap depan, dan tambah cahaya.")
                st.image(rgb)
            else:
                st.image(annotated, channels="BGR")
                for k, probs in enumerate(results, 1):
                    top = int(np.argmax(probs))
                    st.success(f"Wajah {k}: **{class_names[top].upper()}** ({probs[top] * 100:.1f}%)")
                    st.bar_chart(pd.DataFrame({"Probabilitas": probs}, index=class_names))
    else:
        if not HAS_WEBRTC:
            st.error("Mode real-time butuh paket tambahan: `pip install streamlit-webrtc av`, "
                     "lalu jalankan ulang dashboard.")
        else:
            st.info("Klik START lalu izinkan akses kamera di browser.")
            webrtc_streamer(
                key="emotion-live",
                video_frame_callback=video_frame_callback,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={
                    "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "frameRate": {"ideal": 30}},
                    "audio": False,
                },
                async_processing=True,
            )

st.divider()
st.caption("Dibuat dengan TensorFlow & Streamlit")