
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from PIL import Image


st.set_page_config(
    page_title="BMW Sales Dashboard",
    layout="wide"
)

#Data Yükleme ve Ön işleme kısmı
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

#Sidebar Kısmı
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
    
#Filtre uygulama kısmı
mask = (
    (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
    (df['Region'].isin(selected_regions)) &
    (df['Fuel_Type'].isin(selected_fuels)) &
    (df['Model'].isin(selected_models))
)
df_filtered = df[mask].copy()

with st.sidebar:
    st.markdown(f"<p style='font-size:15px; color:#cbd5e1;'>📊 Filtered Rows: "
                f"<b style='color:#ffffff;'>{len(df_filtered):,}</b></p>", unsafe_allow_html=True)

def df_to_csv_bytes(df_in):
    buf = io.StringIO()
    df_in.to_csv(buf, index=False)
    return buf.getvalue().encode('utf-8')

csv_bytes = df_to_csv_bytes(df_filtered)
size_kb = len(csv_bytes) / 1024




#Sayfa Başlığı kısmı
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

# Melisa Charts

st.markdown("### 🎯 4. Price vs Mileage Scatter")
scatter_df = df_filtered.dropna(subset=['Price_USD','Mileage_KM'])
fig_scatter = px.scatter(scatter_df, x='Mileage_KM', y='Price_USD', color='Fuel_Type', size='Sales_Volume',
                         hover_data=['Model','Year','Region'], title="Price vs Mileage (size=Sales volume)")
fig_scatter.update_layout(xaxis_title="Mileage (KM)", yaxis_title="Price (USD)")
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("### 📊 5. Top-Selling Models (Horizontal Bar)")
model_sales = df_filtered.groupby('Model', as_index=False)['Sales_Volume'].sum().sort_values(by='Sales_Volume', ascending=False)
top_n = st.slider("Number of top models to show", 5, 25, 10, key="topn_slider")
fig_bar = px.bar(model_sales.head(top_n), x='Sales_Volume', y='Model', orientation='h', text='Sales_Volume')
fig_bar.update_traces(texttemplate='%{text:,}', textposition='outside')
fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Sales Volume")
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("### 🧭 6. Parallel Coordinates: Engine, Price, Mileage")
pc_df = df_filtered[['Engine_Size_L','Price_USD','Mileage_KM','Sales_Volume']].dropna()
sample_size = min(len(pc_df), 1000)
if sample_size > 0:
    pc_sample = pc_df.sample(sample_size, random_state=1)
    fig_pc = px.parallel_coordinates(pc_sample, color='Sales_Volume',
                                     labels={'Engine_Size_L':'Engine (L)','Price_USD':'Price (USD)','Mileage_KM':'Mileage (KM)'},
                                     title="Parallel Coordinates (sampled)")
    st.plotly_chart(fig_pc, use_container_width=True)
else:
    st.info("No data for parallel coordinates with current filters.")

st.markdown("### 7. Global Sales Map")
st.info("➡️ Grafik buraya eklenecek.")

st.markdown("### 8. Fuel → Transmission → Sankey")
st.info("➡️ Grafik buraya eklenecek.")

st.markdown("### 9. Price Histogram")
st.info("➡️ Grafik buraya eklenecek.")


st.markdown("---")
st.header("🤖 Machine Learning Section")
st.info("➡️ ML modelleri buraya eklenecek.")