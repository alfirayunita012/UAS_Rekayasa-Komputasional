import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = "household_power_consumption.txt"   # ganti sesuai lokasi file
OUTPUT_PATH = "daily_features.csv"

PEAK_HOUR_START = 17
PEAK_HOUR_END = 22   


DAY_HOUR_START = 6
DAY_HOUR_END = 18   

def load_raw_data(path: str) -> pd.DataFrame:
    """Load file mentah UCI (.txt, separator ';', missing value = '?')."""
    print(f"[1/5] Membaca file mentah: {path}")

    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce"
    )
    df = df.drop(columns=["Date", "Time"])

    print(f"      Jumlah baris mentah : {len(df):,}")
    print(f"      Rentang waktu       : {df['datetime'].min()} s.d. {df['datetime'].max()}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Tangani missing values dan pastikan tipe data numerik benar."""
    print("[2/5] Membersihkan data...")

    numeric_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_before = df[numeric_cols].isna().sum().sum()
    print(f"      Total nilai kosong terdeteksi: {missing_before:,}")

    # Interpolasi linear untuk mengisi celah singkat (lebih baik daripada
    # menghapus baris, karena data adalah time-series per menit).
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    missing_after = df[numeric_cols].isna().sum().sum()
    print(f"      Nilai kosong setelah interpolasi: {missing_after:,}")

    return df


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agregasi data per-menit menjadi fitur harian untuk klastering:
      - avg_daily_kwh       : rata-rata konsumsi harian (proxy dari Global_active_power)
      - peak_hour_kwh       : rata-rata konsumsi pada jam beban puncak
      - weekly_std_kwh      : variasi konsumsi mingguan (dihitung lalu di-broadcast ke tiap hari
                              dalam minggu yang sama)
      - day_night_ratio     : rasio konsumsi siang vs malam
    """
    print("[3/5] Membangun fitur harian...")

    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["week"] = df["datetime"].dt.isocalendar().week
    df["year"] = df["datetime"].dt.year

    daily_avg = (
        df.groupby("date")["Global_active_power"]
        .mean()
        .rename("avg_daily_kwh")
    )

    peak_mask = (df["hour"] >= PEAK_HOUR_START) & (df["hour"] < PEAK_HOUR_END)
    peak_avg = (
        df[peak_mask]
        .groupby("date")["Global_active_power"]
        .mean()
        .rename("peak_hour_kwh")
    )


    day_mask = (df["hour"] >= DAY_HOUR_START) & (df["hour"] < DAY_HOUR_END)
    day_avg = df[day_mask].groupby("date")["Global_active_power"].mean().rename("day_kwh")
    night_avg = df[~day_mask].groupby("date")["Global_active_power"].mean().rename("night_kwh")

    # --- Variasi konsumsi mingguan (std per minggu, di-broadcast ke tiap hari) ---
    weekly_std = (
        df.groupby(["year", "week"])["Global_active_power"]
        .std()
        .rename("weekly_std_kwh")
    )
    date_to_week = df.groupby("date")[["year", "week"]].first()

    features = pd.concat([daily_avg, peak_avg, day_avg, night_avg], axis=1)
    features = features.join(date_to_week)
    features = features.join(weekly_std, on=["year", "week"])

    features["day_night_ratio"] = features["day_kwh"] / features["night_kwh"].replace(0, np.nan)

    features = features.drop(columns=["day_kwh", "night_kwh", "year", "week"])
    features = features.reset_index().rename(columns={"index": "date"})

    print(f"      Jumlah hari (baris fitur): {len(features):,}")
    return features


def finalize(features: pd.DataFrame) -> pd.DataFrame:
    """Tangani missing values sisa hasil agregasi dan pastikan siap dipakai."""
    print("[4/5] Finalisasi & validasi fitur...")

    before = len(features)
    features = features.dropna(
        subset=["avg_daily_kwh", "peak_hour_kwh", "weekly_std_kwh", "day_night_ratio"]
    )
    after = len(features)
    print(f"      Baris dibuang karena fitur tidak lengkap: {before - after}")

    # Ringkasan statistik cepat untuk sanity check sebelum ke tahap klastering
    print("\n      Ringkasan statistik fitur:")
    print(features[["avg_daily_kwh", "peak_hour_kwh", "weekly_std_kwh", "day_night_ratio"]]
          .describe().round(3).to_string())

    return features


def main():
    if not Path(RAW_PATH).exists():
        print(f"ERROR: File tidak ditemukan di '{RAW_PATH}'.")
        print("Silakan download dataset dari UCI/Kaggle dan letakkan di folder ini,")
        print("atau ubah variabel RAW_PATH sesuai lokasi filenya.")
        return

    df = load_raw_data(RAW_PATH)
    df = clean_data(df)
    features = build_daily_features(df)
    features = finalize(features)

    print(f"[5/5] Menyimpan hasil ke: {OUTPUT_PATH}")
    features.to_csv(OUTPUT_PATH, index=False)
    print("Selesai. File 'daily_features.csv' siap digunakan untuk tahap")
    print("K-Means (baseline) dan K-Means + Genetic Algorithm (optimasi).")


if __name__ == "__main__":
    main()
