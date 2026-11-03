 
import json
import os
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_PATH = "kolkata_weather_jan_june_2026.csv"
OUT_DIR = "output"
RAIN_THRESHOLD_MM = 1.0
RANDOM_STATE = 42


BASE_FEATURES = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "sunshine_duration",
]
ENG_FEATURES = ["bulan", "hari_dalam_tahun", "rentang_suhu", "durasi_siang_jam"]
FEATURES = BASE_FEATURES + ENG_FEATURES
LEAKAGE_COLS = ["precipitation_sum", "rain_sum", "precipitation_hours", "weather_code"]


# ---------------------------------------------------------------- Langkah 1
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    return df


def build_features(df):
    """Buat target dan fitur turunan."""
    d = df.copy()
    d["hujan"] = (d["precipitation_sum"] >= RAIN_THRESHOLD_MM).astype(int)
    d["bulan"] = d["time"].dt.month
    d["hari_dalam_tahun"] = d["time"].dt.dayofyear
    d["rentang_suhu"] = d["temperature_2m_max"] - d["temperature_2m_min"]
    sr = pd.to_datetime(d["sunrise"])
    ss = pd.to_datetime(d["sunset"])
    d["durasi_siang_jam"] = (ss - sr).dt.total_seconds() / 3600
    return d


# ---------------------------------------------------------------- Langkah 3
def preprocess(d):
    """Split 80/20 (stratified), standarisasi dengan scaler yang HANYA di-fit pada data latih."""
    X = d[FEATURES]
    y = d["hujan"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(X_train)  
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=FEATURES, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=FEATURES, index=X_test.index)
    return X_train_s, X_test_s, y_train, y_test, scaler


# ---------------------------------------------------------------- Langkah 4
def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=RANDOM_STATE, eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------- Langkah 5
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    return metrics, y_pred, y_proba


# ---------------------------------------------------------------- Visualisasi
def fig_class_distribution(d):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x="hujan", data=d, ax=ax, palette=["#4C78A8", "#E45756"], hue="hujan", legend=False)
    ax.set_xticks([0, 1], ["Tidak hujan (0)", "Hujan (1)"])
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Proporsi Kelas Target")
    ax.set_xlabel("")
    ax.set_ylabel("Jumlah hari")
    fig.tight_layout()
    return fig


def fig_histograms(d):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, col in zip(axes.ravel(), FEATURES):
        sns.histplot(d[col], bins=30, kde=True, ax=ax, color="#4C78A8")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    for ax in axes.ravel()[len(FEATURES):]:
        ax.axis("off")
    fig.suptitle("Distribusi Fitur", fontsize=14)
    fig.tight_layout()
    return fig


def fig_heatmap(d):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = d[FEATURES + ["hujan"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                annot_kws={"size": 8}, cbar_kws={"shrink": .8})
    ax.set_title("Heatmap Korelasi Fitur dan Target")
    fig.tight_layout()
    return fig


def fig_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Tidak hujan", "Hujan"], yticklabels=["Tidak hujan", "Hujan"])
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def fig_roc(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color="#E45756", lw=2, label=f"XGBoost (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Acak (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_importance(model):
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color="#4C78A8")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Feature Importance XGBoost")
    fig.tight_layout()
    return fig



def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Langkah 1: struktur data
    df = load_data()
    print("Jumlah baris, kolom:", df.shape)
    print(df.dtypes)
    print(df.describe().T)

    # Langkah 2: missing values
    print("\nMissing values per kolom:\n", df.isna().sum())
    df = df.dropna(subset=LEAKAGE_COLS + BASE_FEATURES + ["sunrise", "sunset"])

    d = build_features(df)
    print("\nProporsi kelas:\n", d["hujan"].value_counts(normalize=True).round(3))

    # Langkah 2: EDA
    for name, fig in [("01_proporsi_kelas", fig_class_distribution(d)),
                      ("02_distribusi_fitur", fig_histograms(d)),
                      ("03_heatmap_korelasi", fig_heatmap(d))]:
        fig.savefig(f"{OUT_DIR}/{name}.png", dpi=150)
        plt.close(fig)

    # Langkah 3: preprocessing
    X_train, X_test, y_train, y_test, scaler = preprocess(d)
    print(f"\nData latih: {len(X_train)} | Data uji: {len(X_test)}")

    # Dataset ternormalisasi untuk dikumpulkan
    norm = pd.concat([X_train.assign(hujan=y_train, split="train"),
                      X_test.assign(hujan=y_test, split="test")]).sort_index()
    norm.insert(0, "time", d.loc[norm.index, "time"].dt.date)
    norm.to_csv(f"{OUT_DIR}/Nama_Test1_KlasifikasiXGBoost.csv", index=False)

    # Langkah 4-5: model + evaluasi
    model = train_model(X_train, y_train)
    metrics, y_pred, y_proba = evaluate(model, X_test, y_test)
    print("\nMetrik evaluasi:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print("\n", classification_report(y_test, y_pred, target_names=["Tidak hujan", "Hujan"]))

    # Langkah 6: visualisasi hasil
    for name, fig in [("04_confusion_matrix", fig_confusion(y_test, y_pred)),
                      ("05_roc_curve", fig_roc(y_test, y_proba)),
                      ("06_feature_importance", fig_importance(model))]:
        fig.savefig(f"{OUT_DIR}/{name}.png", dpi=150)
        plt.close(fig)

    cm = confusion_matrix(y_test, y_pred)
    imp = dict(zip(FEATURES, map(float, model.feature_importances_)))
    result = {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "feature_importance": imp,
        "n_total": len(d), "n_train": len(X_train), "n_test": len(X_test),
        "class_counts": d["hujan"].value_counts().to_dict(),
    }
    with open(f"{OUT_DIR}/hasil.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    joblib.dump({"model": model, "scaler": scaler}, f"{OUT_DIR}/model_xgb.joblib")
    print(f"\nSelesai. Semua keluaran tersimpan di folder '{OUT_DIR}/'.")


if __name__ == "__main__":
    main()
