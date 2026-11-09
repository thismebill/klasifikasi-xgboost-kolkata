import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = ["depth", "mag", "sig", "nst", "dmin", "rms", "gap"]
OPTIONAL_FEATURES = ["latitude", "longitude", "tsunami"]
RANDOM_STATE = 42

FEATURE_INFO = {
    "depth": "kedalaman hiposenter (km)",
    "mag": "magnitudo gempa",
    "sig": "skor signifikansi USGS",
    "nst": "jumlah stasiun seismik yang merekam",
    "dmin": "jarak horizontal ke stasiun terdekat (derajat)",
    "rms": "root-mean-square residu waktu tempuh (detik)",
    "gap": "celah azimut stasiun (derajat)",
    "latitude": "lintang episentrum",
    "longitude": "bujur episentrum",
    "tsunami": "flag peringatan tsunami (0/1)",
}


# --------------------------------------------------------------------------
# Fungsi data & model (murni, tanpa UI)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(source):
    return pd.read_csv(source)


def find_local_csv():
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here, os.getcwd()):
        hits = sorted(glob.glob(os.path.join(folder, "global_earthquakes_seismic*.csv")))
        if hits:
            return hits[0]
    return None


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """Hanya event bertipe 'earthquake' dan buang duplikasi."""
    df = raw.copy()
    if "type" in df.columns:
        df = df[df["type"] == "earthquake"]
    return df.drop_duplicates().reset_index(drop=True)


@st.cache_data(show_spinner=False)
def scale(df: pd.DataFrame, feats: tuple):
    Z = StandardScaler().fit_transform(df[list(feats)])
    return Z


@st.cache_data(show_spinner=False)
def evaluate_k(Z: np.ndarray, k_min=2, k_max=10):
    ks, inertia, sil = [], [], []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
        ks.append(k)
        inertia.append(float(km.inertia_))
        sil.append(float(silhouette_score(Z, km.labels_)))
    return ks, inertia, sil


@st.cache_data(show_spinner=False)
def fit_kmeans(Z: np.ndarray, k: int):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
    return km.labels_, float(km.inertia_), float(silhouette_score(Z, km.labels_)), silhouette_samples(Z, km.labels_)


def describe_distribution(s: pd.Series) -> str:
    sk = s.skew()
    if sk > 1:
        shape = "sangat condong ke kanan (ekor panjang di nilai besar)"
    elif sk > 0.5:
        shape = "cukup condong ke kanan"
    elif sk < -0.5:
        shape = "condong ke kiri"
    else:
        shape = "relatif simetris"
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    return (f"Fitur **{s.name}** memiliki rata-rata {s.mean():.2f} dan median {s.median():.2f}, "
            f"skewness {sk:.2f} sehingga distribusinya {shape}. "
            f"Terdapat {n_out} titik di luar batas 1,5×IQR ({n_out / len(s) * 100:.1f}% data).")


def describe_correlation(corr: pd.DataFrame) -> str:
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    if pairs.empty:
        return "Pilih minimal dua fitur untuk melihat korelasi."
    top = pairs.abs().sort_values(ascending=False).head(3)
    lines = [f"**{a}–{b}** (r = {pairs[(a, b)]:.2f})" for a, b in top.index]
    weak = pairs.abs().lt(0.1).sum()
    return ("Pasangan fitur dengan korelasi terkuat: " + "; ".join(lines) + ". "
            f"Sebanyak {weak} dari {len(pairs)} pasangan memiliki |r| < 0,1, artinya sebagian besar fitur membawa "
            "informasi yang saling berbeda. Pasangan berkorelasi sangat tinggi bersifat redundan dan cenderung "
            "memberi bobot ganda pada konsep yang sama dalam K-Means.")


def interpret_silhouette(v: float) -> str:
    if v >= 0.71:
        return "struktur klaster kuat"
    if v >= 0.51:
        return "struktur klaster cukup baik"
    if v >= 0.26:
        return "struktur klaster lemah namun masih terlihat (klaster saling berdekatan/tumpang tindih)"
    return "struktur klaster hampir tidak ada"


def describe_clusters(zmeans: pd.DataFrame, sizes: pd.Series) -> list:
    out = []
    for c in zmeans.index:
        z = zmeans.loc[c]
        top = z.abs().sort_values(ascending=False).head(3).index
        desc = ", ".join(f"{f} {'tinggi' if z[f] > 0 else 'rendah'} (z={z[f]:+.2f})" for f in top)
        out.append(f"**Klaster {c}** ({sizes[c]:,} data, {sizes[c] / sizes.sum() * 100:.1f}%): ciri paling menonjol → {desc}.")
    return out


# --------------------------------------------------------------------------
# Antarmuka
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Clustering K-Means Gempa Bumi", page_icon="🌍", layout="wide")
    st.title("🌍 Analisis Clustering K-Means: Global Earthquakes Seismic Telemetry")
    st.caption("Praktikum Test 3 — Clustering K-Means dengan Dashboard Streamlit")
    st.markdown(
        "Dashboard ini mengelompokkan kejadian gempa bumi global berdasarkan karakteristik seismiknya "
        "(kedalaman, magnitudo, signifikansi, dan kualitas pengukuran) memakai algoritma **K-Means**. "
        "Gunakan sidebar untuk mengubah jumlah klaster (K), fitur yang dipakai, dan fitur yang divisualisasikan."
    )

    # ---------------- Sidebar: data ----------------
    st.sidebar.header("⚙️ Pengaturan")
    path = find_local_csv()
    upload = st.sidebar.file_uploader("Unggah CSV (opsional)", type="csv")
    source = upload if upload is not None else path
    if source is None:
        st.warning("File CSV tidak ditemukan. Unggah file dataset lewat sidebar.")
        return
    raw = load_csv(source)
    df = preprocess(raw)

    numeric_all = [c for c in DEFAULT_FEATURES + OPTIONAL_FEATURES if c in df.columns]
    feats = st.sidebar.multiselect("Fitur untuk clustering", numeric_all,
                                   default=[f for f in DEFAULT_FEATURES if f in df.columns])
    if len(feats) < 2:
        st.error("Pilih minimal 2 fitur untuk clustering.")
        return
    k = st.sidebar.slider("Jumlah klaster (K)", 2, 10, 3)

    st.sidebar.subheader("Visualisasi scatter")
    x_feat = st.sidebar.selectbox("Sumbu X", feats, index=feats.index("mag") if "mag" in feats else 0)
    y_default = feats.index("depth") if "depth" in feats else min(1, len(feats) - 1)
    y_feat = st.sidebar.selectbox("Sumbu Y", feats, index=y_default)
    use_3d = st.sidebar.checkbox("Tampilkan 3D", value=False)
    z_feat = st.sidebar.selectbox("Sumbu Z (untuk 3D)", feats, index=min(2, len(feats) - 1)) if use_3d else None
    dist_feat = st.sidebar.selectbox("Fitur untuk distribusi", feats)

    # ---------------- Komputasi ----------------
    Z = scale(df, tuple(feats))
    with st.spinner("Menghitung Elbow & Silhouette (K=2..10)..."):
        ks, inertia, sil = evaluate_k(Z)
    labels, inertia_k, sil_k, sil_samples = fit_kmeans(Z, k)
    res = df.copy()
    res["cluster"] = labels
    z_df = pd.DataFrame(Z, columns=feats)
    z_df["cluster"] = labels
    palette = sns.color_palette("Set2", k)

    tabs = st.tabs(["📋 Ringkasan Data", "📊 EDA", "📈 Elbow & Silhouette", "🎯 Hasil Clustering", "🧮 Karakteristik Klaster"])

    # ---------------- Tab 1 ----------------
    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah data (setelah filter)", f"{len(df):,}")
        c2.metric("Data mentah", f"{len(raw):,}")
        c3.metric("Fitur clustering", len(feats))
        c4.metric("Missing value", int(df[feats].isna().sum().sum()))
        st.subheader("Contoh data")
        st.dataframe(df[["id", "place", "time"] + feats].head(10) if {"id", "place", "time"} <= set(df.columns) else df[feats].head(10))
        st.subheader("Statistik deskriptif")
        st.dataframe(df[feats].describe().T.round(3))
        st.subheader("Keterangan fitur")
        st.table(pd.DataFrame({"fitur": feats, "arti": [FEATURE_INFO.get(f, "-") for f in feats]}))
        st.info(
            f"**Analisis:** Data mentah berisi {len(raw):,} baris; setelah menyaring event non-gempa dan duplikasi "
            f"tersisa {len(df):,} baris. Seluruh fitur numerik terpilih dinormalisasi dengan StandardScaler "
            "(rata-rata 0, simpangan baku 1) agar fitur berskala besar seperti *sig* dan *depth* tidak mendominasi jarak Euclidean."
        )

    # ---------------- Tab 2 ----------------
    with tabs[1]:
        st.subheader(f"Distribusi fitur: {dist_feat}")
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        sns.histplot(df[dist_feat], bins=40, kde=True, ax=ax, color="#2a6f97")
        ax.set_title("Histogram")
        c1.pyplot(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        sns.boxplot(x=df[dist_feat], ax=ax, color="#a9d6e5")
        ax.set_title("Boxplot")
        c2.pyplot(fig)
        plt.close(fig)
        st.info("**Analisis:** " + describe_distribution(df[dist_feat]))

        st.subheader("Korelasi antar fitur")
        corr = df[feats].corr()
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
        st.info("**Analisis:** " + describe_correlation(corr))

    # ---------------- Tab 3 ----------------
    with tabs[2]:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(ks, inertia, "o-", color="#1d3557")
        ax[0].axvline(k, ls="--", c="gray")
        ax[0].set(title="Elbow Method (Inertia)", xlabel="K", ylabel="Inertia")
        ax[1].plot(ks, sil, "o-", color="#e63946")
        ax[1].axvline(k, ls="--", c="gray")
        ax[1].set(title="Silhouette Score", xlabel="K", ylabel="Silhouette")
        st.pyplot(fig)
        plt.close(fig)
        best_k = ks[int(np.argmax(sil))]
        drops = [(inertia[i - 1] - inertia[i]) / inertia[i - 1] * 100 for i in range(1, len(ks))]
        st.dataframe(pd.DataFrame({"K": ks, "Inertia": np.round(inertia, 1), "Silhouette": np.round(sil, 4),
                                   "Penurunan inertia (%)": [np.nan] + list(np.round(drops, 1))}))
        st.info(
            f"**Analisis:** Inertia selalu turun saat K bertambah (dari {inertia[0]:,.0f} pada K=2 menjadi "
            f"{inertia[-1]:,.0f} pada K=10), sehingga kurva elbow dibaca dari titik di mana penurunannya mulai melandai. "
            f"Silhouette Score tertinggi dicapai pada **K = {best_k}** dengan nilai {max(sil):.3f}. "
            f"K yang kamu pilih saat ini adalah {k} dengan silhouette {sil_k:.3f} → {interpret_silhouette(sil_k)}."
        )

    # ---------------- Tab 4 ----------------
    with tabs[3]:
        st.subheader(f"Scatter plot klaster (K = {k})")
        if use_3d:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            for c in range(k):
                m = res["cluster"] == c
                ax.scatter(res.loc[m, x_feat], res.loc[m, y_feat], res.loc[m, z_feat], s=6, color=palette[c], label=f"Klaster {c}")
            ax.set_xlabel(x_feat); ax.set_ylabel(y_feat); ax.set_zlabel(z_feat)
            ax.legend()
        else:
            fig, ax = plt.subplots(figsize=(8, 5.5))
            sns.scatterplot(data=res, x=x_feat, y=y_feat, hue="cluster", palette=palette, s=14, ax=ax)
            cent = res.groupby("cluster")[[x_feat, y_feat]].mean()
            ax.scatter(cent[x_feat], cent[y_feat], marker="X", s=200, c="black", label="Centroid (rata-rata)")
            ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        sizes = res["cluster"].value_counts().sort_index()
        st.info(
            f"**Analisis:** Dengan K = {k}, ukuran klaster adalah "
            + ", ".join(f"klaster {c}: {n:,}" for c, n in sizes.items())
            + f". Pada sumbu {x_feat} vs {y_feat}, klaster yang terpisah jelas menandakan kedua fitur ini ikut membedakan kelompok; "
            "klaster yang saling tumpang tindih berarti pembeda utamanya ada pada fitur lain yang tidak sedang ditampilkan."
        )

        st.subheader("Proyeksi PCA 2D (semua fitur clustering)")
        P = PCA(2, random_state=RANDOM_STATE)
        pts = P.fit_transform(Z)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=pts[:, 0], y=pts[:, 1], hue=labels, palette=palette, s=10, ax=ax)
        ax.set(xlabel="PC1", ylabel="PC2")
        st.pyplot(fig)
        plt.close(fig)
        st.info(
            f"**Analisis:** Dua komponen utama hanya menjelaskan {P.explained_variance_ratio_.sum() * 100:.1f}% varians, "
            "jadi gambar ini adalah ringkasan, bukan representasi penuh. Pemisahan yang tampak di sini adalah pemisahan "
            "pada gabungan seluruh fitur."
        )

        st.subheader("Silhouette per klaster")
        per = pd.Series(sil_samples).groupby(labels).mean().round(3)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar([f"Klaster {c}" for c in per.index], per.values, color=palette)
        ax.axhline(sil_k, ls="--", c="gray")
        ax.set_ylabel("Rata-rata silhouette")
        st.pyplot(fig)
        plt.close(fig)
        worst = per.idxmin()
        st.info(
            f"**Analisis:** Garis putus-putus adalah silhouette keseluruhan ({sil_k:.3f}). Klaster {worst} memiliki nilai "
            f"terendah ({per.min():.3f}), yaitu klaster dengan batas paling kabur terhadap klaster tetangganya."
        )

        if {"latitude", "longitude"} <= set(res.columns):
            st.subheader("Sebaran geografis klaster")
            fig, ax = plt.subplots(figsize=(10, 4.5))
            sns.scatterplot(data=res, x="longitude", y="latitude", hue="cluster", palette=palette, s=6, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
            st.info("**Analisis:** Lokasi tidak dipakai sebagai fitur clustering (default), sehingga pola spasial di peta "
                    "adalah hasil sampingan: jika satu klaster menumpuk di zona tertentu, berarti karakter seismiknya terkait wilayah tersebut.")

    # ---------------- Tab 5 ----------------
    with tabs[4]:
        agg = st.radio("Ukuran pemusatan", ["Mean", "Median"], horizontal=True)
        table = res.groupby("cluster")[feats].agg("mean" if agg == "Mean" else "median").round(2)
        table.insert(0, "jumlah_data", res["cluster"].value_counts().sort_index())
        st.subheader(f"{agg} fitur per klaster")
        st.dataframe(table)

        zmeans = z_df.groupby("cluster")[feats].mean()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        zmeans.T.plot(kind="bar", ax=ax, color=palette)
        ax.axhline(0, c="k", lw=.8)
        ax.set_ylabel("Rata-rata z-score")
        ax.set_title("Profil klaster (nilai terstandarisasi)")
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("**Interpretasi klaster:**")
        for line in describe_clusters(zmeans, res["cluster"].value_counts().sort_index()):
            st.markdown("- " + line)
        st.info(
            "**Analisis:** Batang di atas nol berarti rata-rata klaster lebih tinggi dari rata-rata seluruh data pada fitur itu, "
            "di bawah nol berarti lebih rendah. Fitur dengan batang paling panjang adalah pembeda utama klaster."
        )

        st.subheader("Metrik evaluasi")
        c1, c2 = st.columns(2)
        c1.metric("Inertia (SSE)", f"{inertia_k:,.1f}")
        c2.metric("Silhouette Score", f"{sil_k:.3f}")
        st.caption("Inertia kecil = klaster lebih kompak; silhouette mendekati 1 = klaster lebih terpisah dengan baik.")

        out = res.copy()
        for f in feats:
            out[f + "_scaled"] = z_df[f].values
        st.download_button("⬇️ Unduh data + label klaster (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="hasil_clustering.csv", mime="text/csv")


if __name__ == "__main__":
    main()
