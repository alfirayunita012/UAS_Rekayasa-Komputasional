import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

FEATURE_COLS = ["avg_daily_kwh", "peak_hour_kwh", "weekly_std_kwh", "day_night_ratio"]

st.set_page_config(page_title="Segmentasi Konsumsi Listrik", layout="wide")

def load_and_scale(df):
    X_raw = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    return X


def fitness_wcss(centroids, X):
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    return -np.sum((X - centroids[nearest]) ** 2)


def init_population(X, k, pop_size, rng):
    n = X.shape[0]
    return [X[rng.choice(n, size=k, replace=False)].copy() for _ in range(pop_size)]


def tournament_selection(pop, fitnesses, t_size, rng):
    idx = rng.choice(len(pop), size=t_size, replace=False)
    return pop[idx[np.argmax([fitnesses[i] for i in idx])]]


def crossover(p1, p2, rng):
    child = np.empty_like(p1)
    for i in range(p1.shape[0]):
        child[i] = p1[i] if rng.random() < 0.5 else p2[i]
    return child


def mutate(ind, rng, rate, strength):
    m = ind.copy()
    for i in range(m.shape[0]):
        if rng.random() < rate:
            m[i] += rng.normal(0, strength, size=m.shape[1])
    return m


def run_ga(X, k, seed, pop_size, generations, cx_rate, mut_rate, mut_strength, tournament, elitism):
    rng = np.random.default_rng(seed)
    population = init_population(X, k, pop_size, rng)
    best, best_fit = None, -np.inf
    for _ in range(generations):
        fitnesses = [fitness_wcss(ind, X) for ind in population]
        gi = int(np.argmax(fitnesses))
        if fitnesses[gi] > best_fit:
            best_fit, best = fitnesses[gi], population[gi].copy()
        order = np.argsort(fitnesses)[::-1]
        new_pop = [population[i].copy() for i in order[:elitism]]
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitnesses, tournament, rng)
            p2 = tournament_selection(population, fitnesses, tournament, rng)
            child = crossover(p1, p2, rng) if rng.random() < cx_rate else p1.copy()
            new_pop.append(mutate(child, rng, mut_rate, mut_strength))
        population = new_pop
    return best


def compute_metrics(X, labels, wcss):
    if len(set(labels)) < 2:
        return {"silhouette_score": np.nan, "davies_bouldin_index": np.nan, "wcss": wcss}
    return {
        "silhouette_score": silhouette_score(X, labels),
        "davies_bouldin_index": davies_bouldin_score(X, labels),
        "wcss": wcss,
    }


def run_multiple(X, k, n_runs, use_ga, ga_params):
    records = []
    last_labels = None
    for run in range(n_runs):
        seed = 100 + run
        start = time.perf_counter()
        if use_ga:
            centroids = run_ga(X, k, seed, **ga_params)
            model = KMeans(n_clusters=k, init=centroids, n_init=1)
        else:
            model = KMeans(n_clusters=k, init="random", n_init=1, random_state=seed)
        labels = model.fit_predict(X)
        elapsed = time.perf_counter() - start
        metrics = compute_metrics(X, labels, model.inertia_)
        metrics.update({"run": run + 1, "seed": seed, "waktu_detik": elapsed})
        records.append(metrics)
        last_labels = labels
    return pd.DataFrame(records), last_labels

st.sidebar.title("⚙️ Konfigurasi")

uploaded = st.sidebar.file_uploader("Upload daily_features.csv", type="csv")
default_path = "daily_features.csv"

k_value = st.sidebar.slider("Jumlah cluster (k)", 2, 8, 4)
n_runs = st.sidebar.slider("Jumlah run (uji stabilitas)", 3, 20, 10)

st.sidebar.markdown("**Parameter Genetic Algorithm**")
ga_pop = st.sidebar.slider("Ukuran populasi", 10, 100, 30)
ga_gen = st.sidebar.slider("Jumlah generasi", 10, 150, 60)
ga_mut_rate = st.sidebar.slider("Mutation rate", 0.0, 1.0, 0.15)
ga_mut_strength = st.sidebar.slider("Mutation strength", 0.05, 1.0, 0.3)

run_button = st.sidebar.button("🚀 Jalankan Analisis", type="primary", use_container_width=True)

ga_params = dict(
    pop_size=ga_pop, generations=ga_gen, cx_rate=0.8,
    mut_rate=ga_mut_rate, mut_strength=ga_mut_strength,
    tournament=3, elitism=2,
)

st.title("⚡ Segmentasi Pola Konsumsi Listrik Rumah Tangga")
st.caption("K-Means (Baseline) vs K-Means + Genetic Algorithm — Optimasi Centroid Awal")

# --- Load data ---
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    try:
        df = pd.read_csv(default_path)
        st.info(f"Menggunakan file default: `{default_path}`. Bisa upload file lain di sidebar.")
    except FileNotFoundError:
        st.warning("Belum ada file. Silakan upload 'daily_features.csv' di sidebar terlebih dahulu.")
        st.stop()

missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    st.error(f"Kolom berikut tidak ditemukan di file: {missing_cols}")
    st.stop()

with st.expander("📄 Lihat data (5 baris pertama)"):
    st.dataframe(df.head())

st.markdown(f"**Jumlah observasi:** {len(df)} hari &nbsp;|&nbsp; **Fitur:** {', '.join(FEATURE_COLS)}")

X = load_and_scale(df)

tab2, tab3 = st.tabs(["🔬 Perbandingan Baseline vs GA", "📥 Hasil & Download"])

# --- TAB 1: PERBANDINGAN ---
with tab2:
    st.subheader(f"Perbandingan K-Means Baseline vs K-Means + GA (k={k_value}, {n_runs} run)")

    if run_button:
        with st.spinner(f"Menjalankan {n_runs}x Baseline..."):
            df_base, _ = run_multiple(X, k_value, n_runs, use_ga=False, ga_params=ga_params)
        with st.spinner(f"Menjalankan {n_runs}x K-Means + GA..."):
            df_ga, labels_ga = run_multiple(X, k_value, n_runs, use_ga=True, ga_params=ga_params)

        st.session_state["df_base"] = df_base
        st.session_state["df_ga"] = df_ga
        st.session_state["labels_ga"] = labels_ga
        st.session_state["k_used"] = k_value

    if "df_base" in st.session_state:
        df_base = st.session_state["df_base"]
        df_ga = st.session_state["df_ga"]

        # Ringkasan metrik
        m1, m2, m3 = st.columns(3)
        sil_delta = df_ga["silhouette_score"].mean() - df_base["silhouette_score"].mean()
        dbi_delta = df_base["davies_bouldin_index"].mean() - df_ga["davies_bouldin_index"].mean()
        wcss_delta = df_base["wcss"].mean() - df_ga["wcss"].mean()

        m1.metric("Silhouette Score (rata-rata)",
                  f"{df_ga['silhouette_score'].mean():.4f}",
                  f"{sil_delta:+.4f} vs baseline")
        m2.metric("Davies-Bouldin Index (rata-rata)",
                  f"{df_ga['davies_bouldin_index'].mean():.4f}",
                  f"{dbi_delta:+.4f} vs baseline (turun = baik)")
        m3.metric("WCSS (rata-rata)",
                  f"{df_ga['wcss'].mean():.2f}",
                  f"{wcss_delta:+.2f} vs baseline (turun = baik)")

        # Win-rate
        paired = df_base.merge(df_ga, on="seed", suffixes=("_base", "_ga"))
        sil_win = (paired["silhouette_score_ga"] > paired["silhouette_score_base"]).mean() * 100
        dbi_win = (paired["davies_bouldin_index_ga"] < paired["davies_bouldin_index_base"]).mean() * 100
        wcss_win = (paired["wcss_ga"] < paired["wcss_base"]).mean() * 100

        st.markdown(f"**Win-rate GA vs Baseline** (dari {len(paired)} run berpasangan):")
        w1, w2, w3 = st.columns(3)
        w1.metric("Silhouette lebih tinggi", f"{sil_win:.0f}%")
        w2.metric("Davies-Bouldin lebih rendah", f"{dbi_win:.0f}%")
        w3.metric("WCSS lebih rendah", f"{wcss_win:.0f}%")

        # Boxplot
        st.markdown("**Distribusi Metrik dari Seluruh Run**")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, col, title in zip(
            axes,
            ["silhouette_score", "davies_bouldin_index", "wcss"],
            ["Silhouette Score", "Davies-Bouldin Index", "WCSS"],
        ):
            ax.boxplot([df_base[col].dropna(), df_ga[col].dropna()])
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Baseline", "K-Means+GA"])
            ax.set_title(title, fontsize=10)
        st.pyplot(fig)

        with st.expander("Lihat detail tiap run"):
            st.markdown("**Baseline**")
            st.dataframe(df_base.round(4), use_container_width=True)
            st.markdown("**K-Means + GA**")
            st.dataframe(df_ga.round(4), use_container_width=True)
    else:
        st.info("Klik tombol **'🚀 Jalankan Analisis'** di sidebar untuk memulai.")

with tab3:
    st.subheader("Unduh Hasil Analisis")
    if "df_base" in st.session_state:
        df_out = df.copy()
        df_out["cluster_ga_final"] = st.session_state["labels_ga"]

        st.dataframe(df_out.head(20), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download Cluster Assignments (CSV)",
                df_out.to_csv(index=False).encode("utf-8"),
                "cluster_assignments_final.csv",
                "text/csv",
                use_container_width=True,
            )
        with col2:
            comparison_summary = pd.DataFrame({
                "Metrik": ["Silhouette Score", "Davies-Bouldin Index", "WCSS", "Waktu (detik)"],
                "Baseline_mean": [
                    st.session_state["df_base"]["silhouette_score"].mean(),
                    st.session_state["df_base"]["davies_bouldin_index"].mean(),
                    st.session_state["df_base"]["wcss"].mean(),
                    st.session_state["df_base"]["waktu_detik"].mean(),
                ],
                "GA_mean": [
                    st.session_state["df_ga"]["silhouette_score"].mean(),
                    st.session_state["df_ga"]["davies_bouldin_index"].mean(),
                    st.session_state["df_ga"]["wcss"].mean(),
                    st.session_state["df_ga"]["waktu_detik"].mean(),
                ],
            })
            st.download_button(
                "⬇️ Download Ringkasan Perbandingan (CSV)",
                comparison_summary.to_csv(index=False).encode("utf-8"),
                "comparison_summary.csv",
                "text/csv",
                use_container_width=True,
            )
    else:
        st.info("Jalankan analisis di tab sebelumnya terlebih dahulu.")

st.markdown("---")
st.caption("Project Final Komputasional — Segmentasi Pola Konsumsi Listrik Rumah Tangga "
           "menggunakan K-Means yang dioptimasi dengan Genetic Algorithm.")
