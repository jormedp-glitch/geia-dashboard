import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path

def standardize_prices(df: pd.DataFrame, price_columns=None) -> pd.DataFrame:
    price_columns = price_columns or ["price", "price_per_night", "precio", "price_m2"]
    cleaned = df.copy()
    for column in price_columns:
        if column in cleaned.columns:
            cleaned[column] = (
                cleaned[column].astype(str)
                .str.replace("$", "", regex=False)
                .str.replace("€", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for col in normalized.columns:
        if "date" in col or "fecha" in col:
            normalized[col] = pd.to_datetime(normalized[col], errors="coerce")
        if normalized[col].dtype == object:
            normalized[col] = pd.to_numeric(normalized[col], errors="ignore")
    return normalized


def add_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if "price" in enriched.columns:
        enriched["precio_noche"] = pd.to_numeric(enriched["price"], errors="coerce")
    if "availability_365" in enriched.columns:
        enriched["ocupacion_estimada"] = 365 - pd.to_numeric(
            enriched["availability_365"], errors="coerce"
        )
    return enriched


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = standardize_prices(df)
    cleaned = normalize_types(cleaned)
    cleaned = add_basic_columns(cleaned)
    cleaned = cleaned.drop_duplicates()
    return cleaned


def _prepare_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    if "listing_id" not in reviews.columns:
        return pd.DataFrame()
    aggregations = {}
    if "rating" in reviews.columns:
        aggregations["rating"] = "mean"
    if not aggregations:
        return pd.DataFrame()
    grouped = reviews.groupby("listing_id").agg(aggregations).reset_index()
    return grouped


def _normalize_neighbourhoods(listings: pd.DataFrame, neighbourhoods: pd.DataFrame):
    listings_norm = listings.copy()
    if "neighbourhood_cleansed" in listings_norm.columns:
        listings_norm = listings_norm.rename(columns={"neighbourhood_cleansed": "neighbourhood"})
    neighbourhoods_norm = neighbourhoods.copy()
    if "neighbourhood_group" in neighbourhoods_norm.columns:
        neighbourhoods_norm = neighbourhoods_norm.rename(columns={"neighbourhood_group": "district"})
    return listings_norm, neighbourhoods_norm


def unify_data(listings: pd.DataFrame, reviews: pd.DataFrame, neighbourhoods: pd.DataFrame, idealista: pd.DataFrame) -> pd.DataFrame:
    listings_norm, neighbourhoods_norm = _normalize_neighbourhoods(listings, neighbourhoods)
    reviews_summary = _prepare_reviews(reviews)
    if not reviews_summary.empty and "id" in listings_norm.columns:
        listings_norm = listings_norm.merge(
            reviews_summary,
            left_on="id",
            right_on="listing_id",
            how="left",
        )
    if "neighbourhood" in listings_norm.columns and "neighbourhood" in neighbourhoods_norm.columns:
        listings_norm = listings_norm.merge(
            neighbourhoods_norm,
            on="neighbourhood",
            how="left",
            suffixes=("", "_ref"),
        )
    # Idealista no se usa porque requiere un mapeo más elaborado
    return listings_norm


def price_by_area(df, level="district"):
    if level not in {"district", "neighbourhood"}:
        raise ValueError("level debe ser 'district' o 'neighbourhood'")
    if level not in df.columns:
        return pd.DataFrame()
    return (
        df[[level, "precio_noche"]]
        .dropna()
        .groupby(level)
        .agg(precio_medio=("precio_noche", "mean"))
        .reset_index()
        .sort_values("precio_medio", ascending=False)
    )


def occupancy_by_area(df, level="district"):
    if level not in {"district", "neighbourhood"}:
        raise ValueError("level debe ser 'district' o 'neighbourhood'")
    if level not in df.columns:
        return pd.DataFrame()
    return (
        df[[level, "ocupacion_estimada"]]
        .dropna()
        .groupby(level)
        .agg(ocupacion_media=("ocupacion_estimada", "mean"))
        .reset_index()
        .sort_values("ocupacion_media", ascending=False)
    )


st.set_page_config(page_title="GEIA Madrid – Dashboard", layout="wide")

st.title("🏙️ GEIA Madrid – Dashboard de Alquiler Turístico")

idealista = pd.read_csv("data/idealista.csv")
listings = pd.read_csv("data/listings.csv")
neighbourhoods = pd.read_csv("data/neighbourhoods.csv")
reviews = pd.read_csv("data/reviews.csv")

combined = unify_data(listings, reviews, neighbourhoods, idealista)
cleaned = clean_dataset(combined)

st.subheader("📊 KPIs Globales")
col1, col2 = st.columns(2)
if "precio_noche" in cleaned.columns:
    col1.metric("Precio medio por noche", f"{cleaned['precio_noche'].mean():.2f} €")
if "ocupacion_estimada" in cleaned.columns:
    col2.metric("Ocupación media (días/año)", f"{cleaned['ocupacion_estimada'].mean():.1f}")

st.subheader("🏆 Ranking por distrito")
top_price = price_by_area(cleaned, "district")
top_occupancy = occupancy_by_area(cleaned, "district")

tab1, tab2 = st.tabs(["Precio", "Ocupación"])

with tab1:
    st.write("Distritos por precio medio por noche")
    st.dataframe(top_price)
    if not top_price.empty:
        st.bar_chart(top_price.set_index("district")["precio_medio"])

with tab2:
    st.write("Distritos por ocupación media (días/año)")
    st.dataframe(top_occupancy)
    if not top_occupancy.empty:
        st.bar_chart(top_occupancy.set_index("district")["ocupacion_media"])

st.subheader("🔍 Muestra de registros")
st.dataframe(cleaned.head(100))
