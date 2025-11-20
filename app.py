import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(
    page_title="BMW Sales Dashboard",
    layout="wide"
)


@st.cache_data
def load_data(path="data/BMW sales data (2010-2024) (1).csv"):
    df = pd.read_csv(path)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Price_USD'] = pd.to_numeric(df['Price_USD'], errors='coerce')
    df['Mileage_KM'] = pd.to_numeric(df['Mileage_KM'], errors='coerce')
    df['Engine_Size_L'] = pd.to_numeric(df['Engine_Size_L'], errors='coerce')
    df['Sales_Volume'] = pd.to_numeric(df['Sales_Volume'], errors='coerce')
    df = df.dropna(subset=['Year', 'Price_USD', 'Sales_Volume'])
    return df

df = load_data()


with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/44/BMW.svg", width=110)
    st.markdown("## 🎛️ Filters")
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    year_range = st.slider("Select Year Range", min_year, max_year, (2015, max_year))

    regions = sorted(df['Region'].dropna().unique().tolist())
    selected_regions = st.multiselect("Select Regions", regions, default=regions)

    top_models = df['Model'].value_counts().nlargest(10).index.tolist()
    selected_models = st.multiselect("Select Models (top 10)", top_models, default=top_models[:4])

    fuel_types = sorted(df['Fuel_Type'].dropna().unique().tolist())
    selected_fuels = st.multiselect("Fuel Types", fuel_types, default=fuel_types)

st.markdown("<h1 style='text-align:center;'>BMW Sales Dashboard (2010–2024)</h1>", unsafe_allow_html=True)

st.markdown("## 📊 Key Performance Indicators")

k1, k2, k3 = st.columns(3)

with k1:
    st.metric("Total Sales", f"{df['Sales_Volume'].sum():,}")

with k2:
    avg_price = df["Price_USD"].mean()
    st.metric("Average Price", f"${avg_price:,.0f}")

with k3:
    median_mileage = df["Mileage_KM"].median()
    st.metric("Median Mileage", f"{median_mileage:,.0f} KM")


st.markdown("## 📈 Visualizations")

st.markdown("### 1. Yearly Sales Trend")
st.info("➡️ Grafik buraya eklenecek.")

st.markdown("### 2. Region → Model Treemap")
st.info("➡️ Grafik buraya eklenecek.")

st.markdown("### 3. Price Distribution by Fuel Type")
st.info("➡️ Grafik buraya eklenecek.")

st.markdown("### 4. Scatter: Price vs Mileage")
st.info("➡️ Grafik buraya eklenecek.")

st.markdown("### 5. Top Models Bar Chart")
st.info("➡️ Grafik buraya eklenecek.")

st.markdown("### 6. Parallel Coordinates")
st.info("➡️ Grafik buraya eklenecek.")

st.markdown("### 7. Global Sales Map")
st.info("➡️ Grafik buraya eklenecek.")

st.markdown("### 8. Fuel → Transmission → Sankey")
st.info("➡️ Grafik buraya eklenecek.")

st.markdown("### 9. Price Histogram")
st.info("➡️ Grafik buraya eklenecek.")


st.markdown("---")
st.header("🤖 Machine Learning Section")
st.info("➡️ ML modelleri buraya eklenecek.")
