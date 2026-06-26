from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Prediksi Sampah Kota Bogor", layout="wide")

DATA_PATH = Path("bogor_waste_transactions_2025.csv")
GROUND_TRUTH_PATH = Path("bogor_daily_waste_ground_truth_2025.csv")
MODEL_NAME = "Gradient Boosting Regressor"
RANDOM_STATE = 42

# --- KONFIGURASI WARNA KONSISTEN ---
WASTE_COLORS = {
    "Total": "#7c3aed",  # Ungu
    "Organik": "#10b981",  # Hijau Emerald
    "Anorganik": "#eab308"  # Kuning
}

# Palet aman untuk 6 kecamatan
KECAMATAN_COLORS = ["#10b981", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#ea580c"]

EVENT_MULTIPLIERS = {
    "normal": 1.00,
    "ramadan": 1.05,
    "eid_period": 1.16,
    "school_holiday": 1.06,
    "public_event": 1.10,
    "year_end": 1.11,
}

EVENT_LABELS = {
    "normal": "Normal",
    "ramadan": "Ramadan",
    "eid_period": "Periode Idulfitri",
    "school_holiday": "Libur sekolah",
    "public_event": "Acara publik",
    "year_end": "Akhir tahun",
}

DAY_LABELS = {
    "Monday": "Senin",
    "Tuesday": "Selasa",
    "Wednesday": "Rabu",
    "Thursday": "Kamis",
    "Friday": "Jumat",
    "Saturday": "Sabtu",
    "Sunday": "Minggu",
}

DISTRICT_COVERAGE = {
    "Bogor Barat": {"latitude": -6.575630, "longitude": 106.764570, "radius_m": 4374},
    "Bogor Selatan": {"latitude": -6.634152, "longitude": 106.803028, "radius_m": 1829},
    "Bogor Tengah": {"latitude": -6.596542, "longitude": 106.796306, "radius_m": 1186},
    "Bogor Timur": {"latitude": -6.606462, "longitude": 106.828527, "radius_m": 4728},
    "Bogor Utara": {"latitude": -6.570646, "longitude": 106.813101, "radius_m": 2803},
    "Tanah Sareal": {"latitude": -6.544125, "longitude": 106.792078, "radius_m": 2410},
}

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "day_num",
    "is_weekend",
    "event_multiplier",
    "sin_dow",
    "cos_dow",
    "sin_year",
    "cos_year",
    "lag1",
    "lag7",
    "lag14",
    "ma7",
    "ma14",
]
CATEGORICAL_FEATURES = ["kecamatan", "synthetic_event"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

WASTE_VIEW_OPTIONS = {
    "Total": {
        "history_column": "total_tons",
        "forecast_column": "prediksi_total_ton",
        "label": "Total sampah",
        "color": WASTE_COLORS["Total"]
    },
    "Organik": {
        "history_column": "organic_tons",
        "forecast_column": "prediksi_organik_ton",
        "label": "Sampah organik",
        "color": WASTE_COLORS["Organik"]
    },
    "Anorganik": {
        "history_column": "inorganic_tons",
        "forecast_column": "prediksi_anorganik_ton",
        "label": "Sampah anorganik",
        "color": WASTE_COLORS["Anorganik"]
    },
}

st.markdown(
    """
    <style>
    .block-container {max-width: 1440px; padding-top: 1.35rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {border-top: 2px solid #7c3aed; padding-top: .65rem;}
    [data-testid="stMetricLabel"] {font-size: .82rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 1.25rem;}
    .stTabs [data-baseweb="tab"] {padding-left: 0; padding-right: 0;}
    h1, h2, h3 {letter-spacing: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


def validate_daily_data(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "kecamatan",
        "latitude",
        "longitude",
        "synthetic_event",
        "organic_tons",
        "inorganic_tons",
        "total_tons",
        "is_weekend",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(sorted(missing))}")

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["latitude", "longitude", "organic_tons", "inorganic_tons", "total_tons", "is_weekend"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=list(required)).drop_duplicates(["date", "kecamatan"])
    frame = frame.sort_values(["kecamatan", "date"]).reset_index(drop=True)
    split_error = (frame["organic_tons"] + frame["inorganic_tons"] - frame["total_tons"]).abs().max()
    if split_error > 1e-4:
        raise ValueError("Jumlah sampah organik dan anorganik tidak sesuai dengan total sampah.")
    return frame


def transactions_to_daily(transactions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "transaction_id",
        "date",
        "year",
        "month",
        "day_of_week",
        "is_weekend",
        "kecamatan",
        "latitude",
        "longitude",
        "waste_type",
        "synthetic_event",
        "tonnage",
    }
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"Kolom transaksi wajib tidak ditemukan: {', '.join(sorted(missing))}")

    frame = transactions.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["tonnage"] = pd.to_numeric(frame["tonnage"], errors="coerce")
    for column in ["latitude", "longitude", "is_weekend", "year", "month"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "kecamatan", "waste_type", "tonnage"])

    base = frame.groupby(["date", "kecamatan"], as_index=False).agg(
        year=("year", "first"),
        month=("month", "first"),
        day_of_week=("day_of_week", "first"),
        is_weekend=("is_weekend", "first"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        synthetic_event=("synthetic_event", "first"),
    )
    pivot = (
        frame.pivot_table(
            index=["date", "kecamatan"],
            columns="waste_type",
            values="tonnage",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={"organik": "organic_tons", "anorganik": "inorganic_tons"})
    )
    if "organic_tons" not in pivot:
        pivot["organic_tons"] = 0
    if "inorganic_tons" not in pivot:
        pivot["inorganic_tons"] = 0
    daily = base.merge(pivot, on=["date", "kecamatan"], how="left")
    daily["total_tons"] = daily["organic_tons"] + daily["inorganic_tons"]
    return validate_daily_data(daily)


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    transactions = pd.read_csv(DATA_PATH)
    return transactions_to_daily(transactions)


@st.cache_data(show_spinner=False)
def load_transactions() -> pd.DataFrame:
    transactions = pd.read_csv(DATA_PATH)
    transactions["date"] = pd.to_datetime(transactions["date"], errors="coerce")
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], errors="coerce")
    transactions["tonnage"] = pd.to_numeric(transactions["tonnage"], errors="coerce")
    return transactions.dropna(subset=["date", "timestamp", "tonnage"])


def add_features(frame: pd.DataFrame, target_column: str = "total_tons") -> pd.DataFrame:
    model_df = frame.sort_values(["kecamatan", "date"]).copy()
    model_df["event_multiplier"] = model_df["synthetic_event"].map(EVENT_MULTIPLIERS).fillna(1.0)
    model_df["day_num"] = (model_df["date"] - model_df["date"].min()).dt.days
    dow = model_df["date"].dt.dayofweek
    model_df["sin_dow"] = np.sin(2 * np.pi * dow / 7)
    model_df["cos_dow"] = np.cos(2 * np.pi * dow / 7)
    model_df["sin_year"] = np.sin(2 * np.pi * model_df["day_num"] / 365)
    model_df["cos_year"] = np.cos(2 * np.pi * model_df["day_num"] / 365)

    grouped = model_df.groupby("kecamatan")[target_column]
    model_df["lag1"] = grouped.shift(1)
    model_df["lag7"] = grouped.shift(7)
    model_df["lag14"] = grouped.shift(14)
    model_df["ma7"] = grouped.transform(lambda values: values.shift(1).rolling(7).mean())
    model_df["ma14"] = grouped.transform(lambda values: values.shift(1).rolling(14).mean())
    return model_df.dropna().reset_index(drop=True)


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    estimator = GradientBoostingRegressor(
        n_estimators=180,
        learning_rate=0.04,
        max_depth=3,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def regression_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    nonzero = np.abs(actual) > 1e-9

    mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100 if np.sum(
        nonzero) > 0 else 0.0
    r2 = r2_score(actual, predicted) if len(actual) >= 2 else np.nan

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
        "MAPE (%)": mape,
        "R2": r2,
    }


@st.cache_data(show_spinner=False)
def run_final_model(frame: pd.DataFrame):
    model_df = add_features(frame)
    train = model_df[model_df["date"] <= "2025-11-30"]
    test = model_df[model_df["date"] >= "2025-12-01"].sort_values(["date", "kecamatan"])

    model = build_model()
    model.fit(train[FEATURES], train["total_tons"])
    prediction = np.maximum(0, model.predict(test[FEATURES]))

    backtest = test[["date", "kecamatan", "total_tons"]].copy()
    backtest["prediction"] = prediction
    return backtest, regression_metrics(backtest["total_tons"], backtest["prediction"])


def event_name(date: pd.Timestamp) -> str:
    text = date.strftime("%Y-%m-%d")
    periods = [
        ("2025-03-01", "2025-03-23", "ramadan"),
        ("2025-03-24", "2025-04-03", "eid_period"),
        ("2025-06-23", "2025-07-13", "school_holiday"),
        ("2025-12-20", "2025-12-31", "year_end"),
    ]
    for start, end, name in periods:
        if start <= text <= end:
            return name
    return "normal"


def forecast_target(frame: pd.DataFrame, days: int, target_column: str, output_column: str) -> pd.DataFrame:
    model_df = add_features(frame, target_column)
    model = build_model()
    model.fit(model_df[FEATURES], model_df[target_column])

    hist_df = model_df[["date", "kecamatan", target_column, "synthetic_event"]].copy()
    hist_df[output_column] = np.maximum(0, model.predict(model_df[FEATURES]))
    hist_df["tipe_data"] = "Historis"
    hist_df["kejadian"] = hist_df["synthetic_event"].map(EVENT_LABELS)
    hist_df = hist_df.drop(columns=["synthetic_event"])

    last_date = frame["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days)
    history = {
        district: list(model_df.loc[model_df["kecamatan"] == district, target_column].tail(14))
        for district in model_df["kecamatan"].unique()
    }
    coordinates = model_df.groupby("kecamatan")[["latitude", "longitude"]].first()

    rows = []
    for date in future_dates:
        for district in sorted(history):
            values = history[district]
            day_num = (date - model_df["date"].min()).days
            event = event_name(date)
            row = {
                "latitude": coordinates.loc[district, "latitude"],
                "longitude": coordinates.loc[district, "longitude"],
                "day_num": day_num,
                "is_weekend": int(date.dayofweek >= 5),
                "event_multiplier": EVENT_MULTIPLIERS.get(event, 1.0),
                "sin_dow": np.sin(2 * np.pi * date.dayofweek / 7),
                "cos_dow": np.cos(2 * np.pi * date.dayofweek / 7),
                "sin_year": np.sin(2 * np.pi * day_num / 365),
                "cos_year": np.cos(2 * np.pi * day_num / 365),
                "lag1": values[-1],
                "lag7": values[-7],
                "lag14": values[-14],
                "ma7": np.mean(values[-7:]),
                "ma14": np.mean(values[-14:]),
                "kecamatan": district,
                "synthetic_event": event,
            }
            prediction = max(0, float(model.predict(pd.DataFrame([row])[FEATURES])[0]))
            history[district].append(prediction)
            history[district] = history[district][-14:]
            rows.append(
                {
                    "date": date,
                    "kecamatan": district,
                    target_column: np.nan,
                    output_column: prediction,
                    "tipe_data": "Prediksi",
                    "kejadian": EVENT_LABELS.get(event, event),
                }
            )

    future_df = pd.DataFrame(rows)
    return pd.concat([hist_df, future_df], ignore_index=True)


@st.cache_data(show_spinner=False)
def forecast_future(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    total = forecast_target(frame, days, "total_tons", "prediksi_total_ton")
    organic = forecast_target(frame, days, "organic_tons", "prediksi_organik_ton")
    inorganic = forecast_target(frame, days, "inorganic_tons", "prediksi_anorganik_ton")

    merge_cols = ["date", "kecamatan", "tipe_data", "kejadian"]
    forecast = total.merge(
        organic[merge_cols + ["organic_tons", "prediksi_organik_ton"]], on=merge_cols, how="left"
    ).merge(
        inorganic[merge_cols + ["inorganic_tons", "prediksi_anorganik_ton"]], on=merge_cols, how="left"
    )
    return forecast


def line_chart(data, x, y, color_col=None, title=None, height=360, color_scale=None, single_color="#7c3aed"):
    encoding = {
        "x": alt.X(f"{x}:T", title=None),
        "y": alt.Y(f"{y}:Q", title="Ton"),
        "tooltip": [alt.Tooltip(f"{x}:T", format="%Y-%m-%d"), alt.Tooltip(f"{y}:Q", format=",.2f")],
    }
    if color_col:
        if color_scale:
            encoding["color"] = alt.Color(f"{color_col}:N", title=None, scale=color_scale)
        else:
            encoding["color"] = alt.Color(f"{color_col}:N", title=None)
        encoding["tooltip"].insert(0, alt.Tooltip(f"{color_col}:N"))
        chart = alt.Chart(data).mark_line(strokeWidth=2)
    else:
        chart = alt.Chart(data).mark_line(strokeWidth=2, color=single_color)

    return chart.encode(**encoding).properties(title=title, height=height).interactive()


def circle_polygon(latitude, longitude, radius_m, points=72):
    earth_radius = 6_371_000
    angular_distance = radius_m / earth_radius
    latitude_rad = np.radians(latitude)
    longitude_rad = np.radians(longitude)
    polygon = []
    for bearing in np.linspace(0, 2 * np.pi, points, endpoint=False):
        point_latitude = np.arcsin(
            np.sin(latitude_rad) * np.cos(angular_distance)
            + np.cos(latitude_rad) * np.sin(angular_distance) * np.cos(bearing)
        )
        point_longitude = longitude_rad + np.arctan2(
            np.sin(bearing) * np.sin(angular_distance) * np.cos(latitude_rad),
            np.cos(angular_distance) - np.sin(latitude_rad) * np.sin(point_latitude),
        )
        polygon.append([float(np.degrees(point_longitude)), float(np.degrees(point_latitude))])
    return polygon


def radius_map(frame: pd.DataFrame):
    map_data = frame.groupby("kecamatan", as_index=False).agg(total_tons=("total_tons", "sum"))
    map_data["latitude"] = map_data["kecamatan"].map(lambda name: DISTRICT_COVERAGE[name]["latitude"])
    map_data["longitude"] = map_data["kecamatan"].map(lambda name: DISTRICT_COVERAGE[name]["longitude"])
    map_data["radius_m"] = map_data["kecamatan"].map(lambda name: DISTRICT_COVERAGE[name]["radius_m"])
    map_data["polygon"] = map_data.apply(
        lambda row: circle_polygon(row["latitude"], row["longitude"], row["radius_m"]), axis=1
    )
    map_data["total_label"] = map_data["total_tons"].map(lambda value: f"{value:,.0f} ton")
    map_data["radius_label"] = map_data["radius_m"].map(lambda value: f"{value / 1000:.2f} km")

    layer = pdk.Layer(
        "PolygonLayer",
        data=map_data,
        get_polygon="polygon",
        get_fill_color=[124, 58, 237, 55],
        get_line_color=[124, 58, 237, 220],
        line_width_min_pixels=2,
        stroked=True,
        filled=True,
        pickable=True,
    )
    view_state = pdk.ViewState(
        latitude=float(map_data["latitude"].mean()),
        longitude=float(map_data["longitude"].mean()),
        zoom=10.25,
        pitch=0,
    )
    return pdk.Deck(
        map_style="light",
        initial_view_state=view_state,
        layers=[layer],
        tooltip={
            "html": "<b>{kecamatan}</b><br/>Total: {total_label}<br/>Radius: {radius_label}",
            "style": {"backgroundColor": "#143d31", "color": "white"},
        },
    )


def mape_status(value: float) -> tuple[str, str]:
    if value <= 7:
        return "Baik", "normal"
    if value <= 12:
        return "Perlu dipantau", "off"
    return "Tinggi", "inverse"


st.title("Prediksi Sampah Kota Bogor")
st.caption("Dashboard ringkas untuk memantau volume, komposisi, peta cakupan, dan prediksi sampah")

with st.sidebar:
    st.header("Pengaturan Tampilan")
    horizon = st.slider("Rentang prediksi", 7, 90, 30, 1, format="%d hari")

try:
    data = load_default_data()
    transactions = load_transactions()
except Exception as exc:
    st.error(f"Dataset tidak dapat dimuat: {exc}")
    st.stop()

districts = sorted(data["kecamatan"].unique())
with st.sidebar:
    selected_districts = st.multiselect("Kecamatan", districts, default=districts)
    st.divider()
    st.markdown(f"**Model final:** {MODEL_NAME}")
    st.caption("Bagian riset dan pemilihan model berada di notebook. Dashboard ini menampilkan hasil akhir.")

filtered = data[data["kecamatan"].isin(selected_districts)]
if filtered.empty:
    st.warning("Pilih setidaknya satu kecamatan.")
    st.stop()

with st.spinner("Menyiapkan hasil akhir model..."):
    backtest, test_metrics = run_final_model(filtered)
    future = forecast_future(filtered, horizon)
    future_filtered = future[future["kecamatan"].isin(selected_districts)]

ringkasan_tab, komposisi_tab, prediksi_tab, kinerja_tab, detail_tab, data_tab = st.tabs(
    ["Ringkasan", "Komposisi", "Prediksi", "Kinerja", "Detail Harian", "Data"]
)

with ringkasan_tab:
    total = filtered["total_tons"].sum()
    organic_share = filtered["organic_tons"].sum() / total * 100
    daily_city = filtered.groupby("date", as_index=False)[["total_tons", "organic_tons", "inorganic_tons"]].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total tahunan", f"{total:,.0f} ton")
    c2.metric("Rata-rata harian", f"{daily_city.total_tons.mean():,.1f} ton")
    c3.metric("Proporsi organik", f"{organic_share:.1f}%")
    c4.metric("Kecamatan", f"{filtered.kecamatan.nunique()}")

    summary_view = st.selectbox(
        "Jenis sampah pada grafik tren",
        list(WASTE_VIEW_OPTIONS),
        index=0,
        key="summary_waste_view",
    )
    summary_config = WASTE_VIEW_OPTIONS[summary_view]
    daily_plot = daily_city[["date", summary_config["history_column"]]].rename(
        columns={summary_config["history_column"]: "tons"}
    )
    st.altair_chart(
        line_chart(daily_plot, "date", "tons", None, f"Tren harian {summary_config['label'].lower()}", 370,
                   single_color=summary_config["color"]),
        width="stretch",
    )

    left, right = st.columns([1.2, 1])
    district_summary = filtered.groupby("kecamatan", as_index=False)["total_tons"].sum().sort_values("total_tons")
    district_chart = (
        alt.Chart(district_summary)
        .mark_bar(color=WASTE_COLORS["Total"])
        .encode(
            x=alt.X("total_tons:Q", title="Total tahunan (ton)"),
            y=alt.Y("kecamatan:N", sort=None, title=None),
            tooltip=["kecamatan", alt.Tooltip("total_tons:Q", format=",.0f")],
        )
        .properties(title="Total sampah tahunan per kecamatan", height=310)
    )
    left.altair_chart(district_chart, width="stretch")
    right.subheader("Radius cakupan kecamatan")
    right.pydeck_chart(radius_map(filtered), height=310, width="stretch")
    right.caption(
        "Radius dihitung dari rentang koordinat bank sampah pada tiap kecamatan, bukan batas administratif resmi.")

with komposisi_tab:
    monthly = (
        filtered.assign(bulan=filtered.date.dt.to_period("M").astype(str))
        .groupby("bulan", as_index=False)[["organic_tons", "inorganic_tons", "total_tons"]]
        .sum()
    )
    monthly_long = monthly.melt("bulan", ["organic_tons", "inorganic_tons"], var_name="jenis", value_name="tons")
    monthly_long["jenis"] = monthly_long["jenis"].map({"organic_tons": "Organik", "inorganic_tons": "Anorganik"})
    monthly_chart = (
        alt.Chart(monthly_long)
        .mark_bar()
        .encode(
            x=alt.X("bulan:N", title=None),
            y=alt.Y("tons:Q", title="Ton"),
            color=alt.Color("jenis:N", title=None, scale=alt.Scale(domain=["Organik", "Anorganik"],
                                                                   range=[WASTE_COLORS["Organik"],
                                                                          WASTE_COLORS["Anorganik"]])),
            tooltip=["bulan", "jenis", alt.Tooltip("tons:Q", format=",.0f")],
        )
        .properties(title="Komposisi sampah bulanan", height=360)
    )
    st.altair_chart(monthly_chart, width="stretch")

    c1, c2 = st.columns(2)
    weekday = (
        filtered.assign(hari=filtered.date.dt.day_name())
        .groupby("hari", as_index=False)["total_tons"]
        .mean()
    )
    order_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    order = [DAY_LABELS[day] for day in order_en]
    weekday["hari"] = weekday["hari"].map(DAY_LABELS)
    weekday["hari"] = pd.Categorical(weekday["hari"], order, ordered=True)
    weekday = weekday.sort_values("hari")
    c1.altair_chart(
        alt.Chart(weekday)
        .mark_bar(color=WASTE_COLORS["Total"])
        .encode(
            x=alt.X("hari:N", sort=order, title=None),
            y=alt.Y("total_tons:Q", title="Rata-rata ton"),
            tooltip=["hari", alt.Tooltip("total_tons:Q", format=",.2f")],
        )
        .properties(title="Rata-rata sampah menurut hari", height=320),
        width="stretch",
    )

    event_average = (
        filtered.groupby("synthetic_event", as_index=False)["total_tons"]
        .mean()
        .sort_values("total_tons")
    )
    event_average["synthetic_event"] = event_average["synthetic_event"].map(EVENT_LABELS)
    c2.altair_chart(
        alt.Chart(event_average)
        .mark_bar(color=WASTE_COLORS["Total"])
        .encode(
            x=alt.X("total_tons:Q", title="Rata-rata ton"),
            y=alt.Y("synthetic_event:N", sort=None, title=None),
            tooltip=["synthetic_event", alt.Tooltip("total_tons:Q", format=",.2f")],
        )
        .properties(title="Rata-rata sampah menurut kejadian", height=320),
        width="stretch",
    )

with prediksi_tab:
    prediction_view = st.selectbox(
        "Jenis sampah untuk prediksi",
        list(WASTE_VIEW_OPTIONS),
        index=0,
        key="prediction_waste_view",
    )
    prediction_config = WASTE_VIEW_OPTIONS[prediction_view]
    st.caption(
        "Prediksi total, organik, dan anorganik dihitung oleh tiga model terpisah, bukan dari pembagian proporsi total.")

    future_pred_only = future_filtered[future_filtered["tipe_data"] == "Prediksi"]

    st.subheader("Prediksi Agregat")
    city_future = future_pred_only.groupby("date", as_index=False)[prediction_config["forecast_column"]].sum()
    history = (
        filtered.groupby("date", as_index=False)[prediction_config["history_column"]]
        .sum()
        .tail(45)
        .rename(columns={prediction_config["history_column"]: "tons"})
    )
    history["seri"] = "Historis"
    future_plot = city_future.rename(columns={prediction_config["forecast_column"]: "tons"})
    future_plot["seri"] = "Prediksi"
    combined = pd.concat([history, future_plot], ignore_index=True)
    st.altair_chart(
        line_chart(combined, "date", "tons", "seri",
                   f"Prediksi {prediction_config['label'].lower()} untuk {horizon} hari ke depan", 350,
                   color_scale=alt.Scale(domain=["Historis", "Prediksi"],
                                         range=[prediction_config["color"], "#a8a29e"])),
        width="stretch",
    )

    st.subheader("Prediksi per Kecamatan")
    cutoff_date = filtered["date"].max() - pd.Timedelta(days=21)

    dist_history = filtered[filtered["date"] > cutoff_date][
        ["date", "kecamatan", prediction_config["history_column"]]].copy()
    dist_history.rename(columns={prediction_config["history_column"]: "tons"}, inplace=True)
    dist_history["Keterangan"] = "Historis"

    dist_future = future_pred_only[["date", "kecamatan", prediction_config["forecast_column"]]].copy()
    dist_future.rename(columns={prediction_config["forecast_column"]: "tons"}, inplace=True)
    dist_future["Keterangan"] = "Prediksi"

    combined_dist = pd.concat([dist_history, dist_future], ignore_index=True)

    dist_chart = (
        alt.Chart(combined_dist)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Tanggal"),
            y=alt.Y("tons:Q", title="Tonase"),
            color=alt.Color("kecamatan:N", title="Kecamatan", scale=alt.Scale(range=KECAMATAN_COLORS)),
            strokeDash=alt.StrokeDash("Keterangan:N", title="Data"),
            tooltip=[
                alt.Tooltip("date:T", title="Tanggal", format="%Y-%m-%d"),
                alt.Tooltip("kecamatan:N", title="Kecamatan"),
                alt.Tooltip("Keterangan:N", title="Keterangan"),
                alt.Tooltip("tons:Q", title="Ton", format=",.2f")
            ]
        )
        .properties(
            title=f"Tren Prediksi {prediction_config['label']} per Kecamatan",
            height=400
        )
        .interactive()
    )
    st.altair_chart(dist_chart, width="stretch")

    st.subheader("Ringkasan Prediksi")
    summary_columns = list(dict.fromkeys([
        prediction_config["forecast_column"],
        "prediksi_total_ton",
        "prediksi_organik_ton",
        "prediksi_anorganik_ton",
    ]))
    forecast_summary = (
        future_pred_only.groupby("kecamatan")[summary_columns]
        .sum()
        .sort_values(prediction_config["forecast_column"], ascending=False)
        .rename(
            columns={
                "prediksi_total_ton": "Total prediksi",
                "prediksi_organik_ton": "Organik",
                "prediksi_anorganik_ton": "Anorganik",
            }
        )
    )
    st.dataframe(forecast_summary.style.format("{:.2f}"), width="stretch")
    st.download_button(
        "Unduh hasil prediksi",
        future_pred_only.to_csv(index=False).encode("utf-8"),
        file_name=f"prediksi_sampah_bogor_{horizon}_hari.csv",
        mime="text/csv",
    )

with kinerja_tab:
    status_label, status_delta = mape_status(test_metrics["MAPE (%)"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE pengujian", f"{test_metrics['MAE']:.2f} ton")
    m2.metric("RMSE pengujian", f"{test_metrics['RMSE']:.2f} ton")
    m3.metric("MAPE pengujian", f"{test_metrics['MAPE (%)']:.2f}%", delta=status_label, delta_color=status_delta)
    m4.metric("R2 pengujian", f"{test_metrics['R2']:.3f}")

    st.caption("Pengujian dilakukan pada data Desember. Model dilatih menggunakan data sampai akhir November.")
    city_backtest = backtest.groupby("date", as_index=False).agg(
        aktual=("total_tons", "sum"), prediksi=("prediction", "sum")
    )
    backtest_long = city_backtest.melt("date", ["aktual", "prediksi"], var_name="seri", value_name="tons")
    backtest_long["seri"] = backtest_long["seri"].map({"aktual": "Aktual", "prediksi": "Prediksi"})
    st.altair_chart(
        line_chart(backtest_long, "date", "tons", "seri", "Pengujian balik total sampah Kota Bogor", 370,
                   color_scale=alt.Scale(domain=["Aktual", "Prediksi"], range=[WASTE_COLORS["Total"], "#ec4899"])),
        width="stretch",
    )

    district_backtest = backtest.copy()
    district_error = (
        district_backtest.assign(abs_error=(district_backtest["total_tons"] - district_backtest["prediction"]).abs())
        .groupby("kecamatan", as_index=False)
        .agg(
            aktual=("total_tons", "sum"),
            prediksi=("prediction", "sum"),
            galat_absolut=("abs_error", "mean"),
        )
    )
    district_error["selisih_total"] = district_error["prediksi"] - district_error["aktual"]
    error_chart = (
        alt.Chart(district_error.sort_values("galat_absolut"))
        .mark_bar(color=WASTE_COLORS["Total"])
        .encode(
            x=alt.X("galat_absolut:Q", title="Rata-rata galat absolut (ton)"),
            y=alt.Y("kecamatan:N", sort=None, title=None),
            tooltip=[
                "kecamatan",
                alt.Tooltip("aktual:Q", format=",.2f"),
                alt.Tooltip("prediksi:Q", format=",.2f"),
                alt.Tooltip("galat_absolut:Q", format=",.2f"),
            ],
        )
        .properties(title="Galat prediksi rata-rata per kecamatan", height=320)
    )
    st.altair_chart(error_chart, width="stretch")

    if GROUND_TRUTH_PATH.exists():
        truth = pd.read_csv(GROUND_TRUTH_PATH, parse_dates=["date"])
        december = truth[truth.date.between("2025-12-01", "2025-12-31")]
        floor = regression_metrics(december.total_tons, december.clean_signal_tons)
        st.info(
            f"Catatan simulasi: batas galat noise sintetis pada Desember sekitar "
            f"{floor['MAPE (%)']:.2f}% MAPE, bahkan terhadap sinyal bersih tersembunyi."
        )

with detail_tab:
    st.header("Pencarian & Analisis Harian")
    st.caption("Pilih spesifik kecamatan, jenis sampah, dan tanggal untuk membedah data secara lebih detail.")

    c1, c2, c3 = st.columns(3)
    detail_view = c1.selectbox(
        "Jenis Sampah",
        list(WASTE_VIEW_OPTIONS),
        index=0,
        key="detail_waste_view"
    )
    detail_config = WASTE_VIEW_OPTIONS[detail_view]

    detail_districts = c2.multiselect(
        "Filter Kecamatan",
        options=selected_districts,
        default=selected_districts,
        key="detail_districts"
    )

    min_date = future_filtered["date"].min().date()
    max_date = future_filtered["date"].max().date()
    last_actual_date = filtered["date"].max().date()

    selected_date = c3.date_input(
        "Pilih Tanggal (Masa lalu / Depan)",
        value=last_actual_date,
        min_value=min_date,
        max_value=max_date
    )

    sel_ts = pd.to_datetime(selected_date)
    detail_df = future_filtered[
        (future_filtered["date"] == sel_ts) &
        (future_filtered["kecamatan"].isin(detail_districts))
        ].copy()

    if detail_df.empty:
        st.warning(
            "Tidak ada data untuk kombinasi filter yang dipilih (Catatan: 14 hari pertama digunakan sebagai riwayat awal (Lag) model).")
    else:
        tipe_data = detail_df["tipe_data"].iloc[0]
        actual_col = detail_config["history_column"]
        pred_col = detail_config["forecast_column"]
        label_nama = detail_config["label"]

        st.subheader(f"Statistik {tipe_data} - {selected_date.strftime('%d %B %Y')}")

        if tipe_data == "Historis":
            y_true = detail_df[actual_col]
            y_pred = detail_df[pred_col]

            if len(detail_df) >= 2 and y_true.sum() > 0:
                metrics = regression_metrics(y_true, y_pred)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("MAE Harian", f"{metrics['MAE']:.2f} ton")
                m2.metric("RMSE Harian", f"{metrics['RMSE']:.2f} ton")
                st_lab, st_col = mape_status(metrics['MAPE (%)'])
                m3.metric("MAPE Harian", f"{metrics['MAPE (%)']:.2f}%", delta=st_lab, delta_color=st_col)
                m4.metric("R2 Harian", f"{metrics['R2']:.3f}")
            else:
                mae = np.abs(y_true.iloc[0] - y_pred.iloc[0])
                m1, m2, m3 = st.columns(3)
                m1.metric("Aktual", f"{y_true.iloc[0]:.2f} ton")
                m2.metric("Prediksi", f"{y_pred.iloc[0]:.2f} ton")
                m3.metric("Selisih Absolut", f"{mae:.2f} ton")
                st.info("Pilih lebih dari 1 kecamatan untuk melihat agregat statistik akurasi penuh (R2 Harian).")

            chart_df = detail_df[["kecamatan", actual_col, pred_col]].rename(
                columns={actual_col: "Aktual", pred_col: "Prediksi"})
            chart_df_melted = chart_df.melt("kecamatan", var_name="Keterangan", value_name="Tonase")

        else:
            st.info("Anda memilih hari di masa depan. Hanya menampilkan data prediksi (tanpa metrik evaluasi aktual).")
            chart_df = detail_df[["kecamatan", pred_col]].rename(columns={pred_col: "Prediksi"})
            chart_df_melted = chart_df.melt("kecamatan", var_name="Keterangan", value_name="Tonase")

        st.markdown("### Perbandingan Per Kecamatan")

        bar_chart = (
            alt.Chart(chart_df_melted)
            .mark_bar()
            .encode(
                x=alt.X("kecamatan:N", title="Kecamatan", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Tonase:Q", title="Tonase (Ton)"),
                xOffset="Keterangan:N",
                color=alt.Color("Keterangan:N", title="Legenda", scale=alt.Scale(domain=["Aktual", "Prediksi"],
                                                                                 range=[detail_config["color"],
                                                                                        "#a8a29e"])),
                tooltip=["kecamatan", "Keterangan", alt.Tooltip("Tonase:Q", format=",.2f")]
            )
            .properties(height=350)
            .interactive()
        )
        st.altair_chart(bar_chart, use_container_width=True)

        tabel_kolom = ["kecamatan", "kejadian", actual_col, pred_col] if tipe_data == "Historis" else ["kecamatan",
                                                                                                       "kejadian",
                                                                                                       pred_col]
        tabel_df = detail_df[tabel_kolom].rename(
            columns={actual_col: "Aktual (Ton)", pred_col: "Prediksi (Ton)", "kecamatan": "Kecamatan",
                     "kejadian": "Event/Kejadian"}
        )
        st.dataframe(tabel_df, hide_index=True, width="stretch")

with data_tab:
    st.subheader("Data yang digunakan")
    filtered_transactions = transactions[transactions["kecamatan"].isin(selected_districts)]

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Transaksi", f"{len(filtered_transactions):,}")
    q2.metric("Agregat harian", f"{len(filtered):,}")
    q3.metric("Kecamatan", f"{filtered.kecamatan.nunique():,}")
    q4.metric("Cakupan tanggal", f"{filtered.date.nunique()} hari")

    display_data = filtered.sort_values(["date", "kecamatan"]).rename(
        columns={
            "date": "tanggal",
            "year": "tahun",
            "month": "bulan",
            "day_of_week": "hari",
            "is_weekend": "akhir_pekan",
            "latitude": "lintang",
            "longitude": "bujur",
            "synthetic_event": "kejadian_simulasi",
            "organic_tons": "sampah_organik_ton",
            "inorganic_tons": "sampah_anorganik_ton",
            "total_tons": "total_sampah_ton",
        }
    )
    display_data["hari"] = display_data["hari"].map(DAY_LABELS).fillna(display_data["hari"])
    display_data["kejadian_simulasi"] = display_data["kejadian_simulasi"].map(EVENT_LABELS).fillna(
        display_data["kejadian_simulasi"]
    )
    st.dataframe(display_data, width="stretch", hide_index=True)
    st.download_button(
        "Unduh data terfilter",
        display_data.to_csv(index=False).encode("utf-8"),
        file_name="data_sampah_bogor_terfilter.csv",
        mime="text/csv",
    )

    st.subheader("Contoh data transaksi")
    transaction_display = filtered_transactions.sort_values("timestamp").head(500)
    transaction_display = transaction_display.rename(
        columns={
            "transaction_id": "id_transaksi",
            "timestamp": "waktu_transaksi",
            "date": "tanggal",
            "year": "tahun",
            "month": "bulan",
            "day_of_week": "hari",
            "is_weekend": "akhir_pekan",
            "latitude": "lintang",
            "longitude": "bujur",
            "waste_type": "jenis_sampah",
            "source_type": "sumber_sampah",
            "vehicle_id": "id_kendaraan",
            "route_id": "id_rute",
            "synthetic_event": "kejadian_simulasi",
            "load_factor": "faktor_muatan",
            "tonnage": "tonase",
        }
    )
    if "hari" in transaction_display:
        transaction_display["hari"] = transaction_display["hari"].map(DAY_LABELS).fillna(transaction_display["hari"])
    if "kejadian_simulasi" in transaction_display:
        transaction_display["kejadian_simulasi"] = transaction_display["kejadian_simulasi"].map(EVENT_LABELS).fillna(
            transaction_display["kejadian_simulasi"]
        )
    st.dataframe(transaction_display, width="stretch", hide_index=True)
    st.download_button(
        "Unduh data transaksi",
        filtered_transactions.to_csv(index=False).encode("utf-8"),
        file_name="data_transaksi_sampah_bogor.csv",
        mime="text/csv",
    )