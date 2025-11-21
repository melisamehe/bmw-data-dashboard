
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats as scipy_stats



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

    st.markdown("---")
    st.markdown("### ℹ️ Dataset & Export")
    st.markdown("<p class='muted'>Use filters above then download the filtered CSV below.</p>", unsafe_allow_html=True)

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

with st.sidebar:
    st.markdown(f"<p style='font-size:14px; color:#cbd5e1;'>💾 Filtered CSV size: <b style='color:#ffffff;'>{size_kb:,.1f} KB</b></p>", unsafe_allow_html=True)
    st.download_button("💾 Download Filtered CSV", data=csv_bytes, file_name="filtered_bmw_data.csv", mime="text/csv")



#Sayfa Başlığı kısmı
st.markdown("<h1 style='text-align:center;'>BMW Sales Dashboard (2010–2024)</h1>", unsafe_allow_html=True)


# Gösterge kartları
total_sales = int(df_filtered['Sales_Volume'].sum()) if len(df_filtered) > 0 else 0
avg_price = df_filtered['Price_USD'].mean() if len(df_filtered) > 0 else 0
median_mileage = df_filtered['Mileage_KM'].median() if len(df_filtered) > 0 else 0

k1, k2, k3 = st.columns([1,1,1])
with k1:
    st.markdown(f"""<div class="kpi-card"><h3>Total Sales</h3><h2>{total_sales:,}</h2><div class="muted">Filtered total sales</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card"><h3>Average Price</h3><h2>${avg_price:,.0f}</h2><div class="muted">Filtered average</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card"><h3>Median Mileage</h3><h2>{median_mileage:,.0f} KM</h2><div class="muted">Filtered median</div></div>""", unsafe_allow_html=True)


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
                                     labels={
                                         'Engine_Size_L':'Engine (L)','Price_USD':'Price (USD)','Mileage_KM':'Mileage (KM)'},
                                     title="Parallel Coordinates (sampled)")
    st.plotly_chart(fig_pc, use_container_width=True)
else:
    st.info("No data for parallel coordinates with current filters.")

# Zeynep Charts

st.markdown("### 7.🌍 Global Sales Map")

col1, col2 = st.columns(2)
with col1:
    projection_type = st.selectbox("Map Projection", ["orthographic", "natural earth", "equirectangular"], index=0)
with col2:
    bubble_size = st.slider("Bubble Size", 20, 60, 45)

region_coords = {
    'Europe': {'lat': 54.5260, 'lon': 15.2551},
    'Asia': {'lat': 34.0479, 'lon': 100.6197},
    'North America': {'lat': 54.5260, 'lon': -105.2551},
    'South America': {'lat': -8.7832, 'lon': -55.4915},
    'Middle East': {'lat': 25.0, 'lon': 45.0},
    'Africa': {'lat': 1.6508, 'lon': 17.6791},
    'Oceania': {'lat': -22.7359, 'lon': 140.0188}
}

map_df = df_filtered.groupby("Region", as_index=False).agg({
    "Sales_Volume": "sum",
    "Price_USD": "mean",
    "Model": "nunique"
}).rename(columns={"Model": "Unique_Models", "Price_USD": "Avg_Price"})

if len(map_df) > 0:
    map_df['Year_Range'] = f"{int(df_filtered['Year'].min())}-{int(df_filtered['Year'].max())}"
    map_df["lat"] = map_df["Region"].map(lambda x: region_coords.get(x, {}).get("lat", 0))
    map_df["lon"] = map_df["Region"].map(lambda x: region_coords.get(x, {}).get("lon", 0))
    map_df["Market_Share_%"] = (map_df["Sales_Volume"] / map_df["Sales_Volume"].sum() * 100).round(2)
    map_df["Size"] = map_df["Sales_Volume"].apply(lambda x: max(8, x / 1_000_000))
    
    fig_map = px.scatter_geo(
        map_df, lat="lat", lon="lon", size="Size", size_max=bubble_size,
        color="Sales_Volume", hover_name="Region",
        hover_data={"Sales_Volume": ":,", "Avg_Price": ":,.0f", "Unique_Models": True,
                    "Market_Share_%": ":.2f", "Year_Range": True, "lat": False, "lon": False, "Size": False},
        color_continuous_scale="OrRd", projection=projection_type
    )
    
    fig_map.update_layout(
        geo=dict(showland=True, landcolor="#e8e8e8", showocean=True, oceancolor="#b3d9ff",
                 showcountries=True, countrycolor="#666666", showlakes=True, lakecolor="#b3d9ff",
                 projection_type=projection_type, bgcolor="#1e1e1e"),
        height=650, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="#0e1117", font=dict(color="white"), showlegend=False
    )
    fig_map.update_coloraxes(colorbar=dict(title="Sales Volume", tickformat=",", len=0.7))
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("⚠️ No data available for map.")

st.markdown("---")


st.markdown("### 8.🔄 Sankey Diagram: Fuel → Transmission → Sales Classification")

col1, col2 = st.columns(2)
with col1:
    flow_direction = st.selectbox("Flow Direction", ["Left to Right", "Right to Left"], index=0)
with col2:
    show_percentages = st.checkbox("Show Percentages", value=True)

sankey_df = df_filtered.groupby(['Fuel_Type', 'Transmission', 'Sales_Classification'], 
                                 as_index=False)['Sales_Volume'].sum()

if len(sankey_df) > 0:
    total_sales = sankey_df['Sales_Volume'].sum()
    fuels = sorted(sankey_df['Fuel_Type'].unique().tolist())
    trans = sorted(sankey_df['Transmission'].unique().tolist())
    classes = sorted(sankey_df['Sales_Classification'].unique().tolist())
    
    nodes = (classes + trans + fuels) if flow_direction == "Right to Left" else (fuels + trans + classes)
    node_indices = {n: i for i, n in enumerate(nodes)}
    
    node_colors = {
        "Petrol": "#FF6B6B", "Diesel": "#4ECDC4", "Hybrid": "#FFD93D", "Electric": "#5DA9E9",
        "Automatic": "#A78BFA", "Manual": "#C084FC",
        "High": "#FB923C", "Low": "#FDE047"
    }
    
    def normalize(values, new_min=10, new_max=30):
        if not values: return []
        old_min, old_max = min(values), max(values)
        if old_max == old_min: return [new_min] * len(values)
        return [((v - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min for v in values]
    
    source, target, value, real_values, percentages, link_colors, link_labels = [], [], [], [], [], [], []
    
    if flow_direction == "Right to Left":
        tc = sankey_df.groupby(["Sales_Classification", "Transmission"])["Sales_Volume"].sum().reset_index()
        for i, row in tc.iterrows():
            source.append(node_indices[row["Sales_Classification"]])
            target.append(node_indices[row["Transmission"]])
            value.append(normalize(tc["Sales_Volume"].tolist())[i])
            real_values.append(row["Sales_Volume"])
            percentages.append(row["Sales_Volume"] / total_sales * 100)
            link_colors.append(node_colors.get(row["Sales_Classification"], "#CCCCCC"))
            link_labels.append(f"{row['Sales_Classification']} → {row['Transmission']}")
        
        ft = sankey_df.groupby(["Transmission", "Fuel_Type"])["Sales_Volume"].sum().reset_index()
        for i, row in ft.iterrows():
            source.append(node_indices[row["Transmission"]])
            target.append(node_indices[row["Fuel_Type"]])
            value.append(normalize(ft["Sales_Volume"].tolist())[i])
            real_values.append(row["Sales_Volume"])
            percentages.append(row["Sales_Volume"] / total_sales * 100)
            link_colors.append(node_colors.get(row["Transmission"], "#CCCCCC"))
            link_labels.append(f"{row['Transmission']} → {row['Fuel_Type']}")
    else:
        ft = sankey_df.groupby(["Fuel_Type", "Transmission"])["Sales_Volume"].sum().reset_index()
        for i, row in ft.iterrows():
            source.append(node_indices[row["Fuel_Type"]])
            target.append(node_indices[row["Transmission"]])
            value.append(normalize(ft["Sales_Volume"].tolist())[i])
            real_values.append(row["Sales_Volume"])
            percentages.append(row["Sales_Volume"] / total_sales * 100)
            link_colors.append(node_colors.get(row["Fuel_Type"], "#CCCCCC"))
            link_labels.append(f"{row['Fuel_Type']} → {row['Transmission']}")
        
        tc = sankey_df.groupby(["Transmission", "Sales_Classification"])["Sales_Volume"].sum().reset_index()
        for i, row in tc.iterrows():
            source.append(node_indices[row["Transmission"]])
            target.append(node_indices[row["Sales_Classification"]])
            value.append(normalize(tc["Sales_Volume"].tolist())[i])
            real_values.append(row["Sales_Volume"])
            percentages.append(row["Sales_Volume"] / total_sales * 100)
            link_colors.append(node_colors.get(row["Transmission"], "#CCCCCC"))
            link_labels.append(f"{row['Transmission']} → {row['Sales_Classification']}")
    
    customdata = [[rv, pct, lbl] for rv, pct, lbl in zip(real_values, percentages, link_labels)] if show_percentages else [[rv, lbl] for rv, lbl in zip(real_values, link_labels)]
    hover_template = "Sales: %{customdata[0]:,}<br>Percentage: %{customdata[1]:.2f}%<br>Flow: %{customdata[2]}<extra></extra>" if show_percentages else "Sales: %{customdata[0]:,}<br>Flow: %{customdata[1]}<extra></extra>"
    
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(label=nodes, pad=20, thickness=20, 
                  color=[node_colors.get(n, "#CCCCCC") for n in nodes],
                  line=dict(color="black", width=0.5)),
        link=dict(source=source, target=target, value=value,
                  color=[f"rgba{tuple(list(int(c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.4])}" for c in link_colors],
                  customdata=customdata, hovertemplate=hover_template)
    )])
    
    fig_sankey.update_layout(height=600, font=dict(size=12, color="white"),
                             paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                             margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_sankey, use_container_width=True)
else:
    st.warning("⚠️ No data available for Sankey diagram.")

st.markdown("---")



st.markdown("### 9.📊 Price Distribution Analysis")

col1, col2, col3 = st.columns(3)
with col1:
    viz_type = st.selectbox("Visualization Type", ["Histogram", "Histogram + KDE"], index=0)
with col2:
    year_select = st.selectbox("Select Year", ['All'] + sorted(df_filtered['Year'].dropna().unique().tolist()), index=0)
with col3:
    num_bins = st.slider("Number of Bins", 10, 100, 40, 5)

hist_df = df_filtered[df_filtered['Year'] == int(year_select)].copy() if year_select != 'All' else df_filtered.copy()
title_suffix = f"Year {year_select}" if year_select != 'All' else f"Years {int(df_filtered['Year'].min())}-{int(df_filtered['Year'].max())}"

if len(hist_df) > 0:
    stats = {
        'mean': hist_df['Price_USD'].mean(), 'median': hist_df['Price_USD'].median(),
        'std': hist_df['Price_USD'].std(), 'min': hist_df['Price_USD'].min(),
        'max': hist_df['Price_USD'].max(), 'q1': hist_df['Price_USD'].quantile(0.25),
        'q3': hist_df['Price_USD'].quantile(0.75), 'count': len(hist_df)
    }
    
    iqr = stats['q3'] - stats['q1']
    lower_bound, upper_bound = stats['q1'] - 1.5 * iqr, stats['q3'] + 1.5 * iqr
    stats['outliers'] = len(hist_df[(hist_df['Price_USD'] < lower_bound) | (hist_df['Price_USD'] > upper_bound)])
    
    if viz_type == "Histogram":
        fig = px.histogram(hist_df, x='Price_USD', nbins=num_bins,
                          title=f"Price Distribution - {title_suffix}",
                          color_discrete_sequence=['#60A5FA'])
        fig.update_traces(marker_line_width=1, marker_line_color='white')
        fig.add_vline(x=stats['mean'], line_dash="dash", line_color="red",
                     annotation_text=f"Mean: ${stats['mean']:,.0f}", annotation_position="top")
        fig.add_vline(x=stats['median'], line_dash="dash", line_color="green",
                     annotation_text=f"Median: ${stats['median']:,.0f}", annotation_position="bottom")
    else:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=hist_df['Price_USD'], nbinsx=num_bins, name='Frequency',
                                   marker_color='#60A5FA', opacity=0.7))
        
        density = scipy_stats.gaussian_kde(hist_df['Price_USD'].dropna())
        x_range = np.linspace(hist_df['Price_USD'].min(), hist_df['Price_USD'].max(), 200)
        y_density = density(x_range) * len(hist_df) * (hist_df['Price_USD'].max() - hist_df['Price_USD'].min()) / num_bins
        
        fig.add_trace(go.Scatter(x=x_range, y=y_density, mode='lines', name='Density',
                                line=dict(color='#EF4444', width=3)))
        fig.add_vline(x=stats['mean'], line_dash="dash", line_color="red",
                     annotation_text=f"Mean: ${stats['mean']:,.0f}", annotation_position="top")
        fig.add_vline(x=stats['median'], line_dash="dash", line_color="green",
                     annotation_text=f"Median: ${stats['median']:,.0f}", annotation_position="bottom")
        fig.update_layout(title=f"Price Distribution with KDE - {title_suffix}")
    
    fig.update_layout(xaxis_title="Price (USD)", yaxis_title="Frequency", height=500,
                     paper_bgcolor="#0e1117", plot_bgcolor="#1e1e1e",
                     font=dict(color="white"), showlegend=(viz_type == "Histogram + KDE"))
    st.plotly_chart(fig, use_container_width=True)
    
else:
    st.warning("⚠️ No data available.")

st.markdown("---")




# Machine Learning Bölümü

st.header("🔮 Machine Learning Analysis")

st.subheader("1️⃣ K-Means Clustering (Customer Segmentation)")

col1, col2 = st.columns(2)
with col1:
    num_clusters = st.slider("Number of Clusters", 2, 8, 4)
with col2:
    show_2d = st.checkbox("Show 2D Projection", False)

ml_df = df[['Price_USD', 'Mileage_KM', 'Engine_Size_L']].dropna().copy()

if len(ml_df) > 0:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(ml_df)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    ml_df['Cluster'] = kmeans.fit_predict(scaled_data)
    
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(scaled_data, ml_df['Cluster'])
    
    
    fig_3d = px.scatter_3d(ml_df, x='Mileage_KM', y='Engine_Size_L', z='Price_USD',
                           color='Cluster', title=f"BMW Segmentation ({num_clusters} Clusters)",
                           color_discrete_sequence=px.colors.qualitative.Set2)
    fig_3d.update_layout(height=600, paper_bgcolor="#0e1117",
                        scene=dict(xaxis_title="Mileage (KM)", yaxis_title="Engine Size (L)",
                                  zaxis_title="Price (USD)", bgcolor="#1e1e1e"),
                        font=dict(color="white"))
    st.plotly_chart(fig_3d, use_container_width=True)
    
    if show_2d:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        ml_df['PCA1'], ml_df['PCA2'] = pca_data[:, 0], pca_data[:, 1]
        
        fig_2d = px.scatter(ml_df, x='PCA1', y='PCA2', color='Cluster',
                           title="2D Projection (PCA)",
                           color_discrete_sequence=px.colors.qualitative.Set2)
        fig_2d.update_layout(height=500, paper_bgcolor="#0e1117",
                            plot_bgcolor="#1e1e1e", font=dict(color="white"))
        st.plotly_chart(fig_2d, use_container_width=True)
    
    cluster_stats = ml_df.groupby('Cluster').agg({
        'Price_USD': 'mean', 'Mileage_KM': 'mean',
        'Engine_Size_L': 'mean', 'Cluster': 'count'
    }).round(0)
    cluster_stats.columns = ['Avg_Price', 'Avg_Mileage', 'Avg_Engine', 'Count']
    cluster_stats = cluster_stats.reset_index()
    cluster_stats['Percentage'] = (cluster_stats['Count'] / len(ml_df) * 100).round(1)
    
    def get_profile(row):
        if row['Avg_Price'] > 80000 and row['Avg_Mileage'] < 50000: return "Luxury/Premium"
        elif row['Avg_Price'] < 50000 and row['Avg_Mileage'] > 100000: return "Economy/High Mileage"
        elif row['Avg_Price'] < 50000 and row['Avg_Mileage'] < 50000: return "Value/Opportunity"
        else: return "Mid-Range"
    
    cluster_stats['Profile'] = cluster_stats.apply(get_profile, axis=1)
    
    for _, row in cluster_stats.iterrows():
        with st.expander(f"Cluster {int(row['Cluster'])}: {row['Profile']} ({int(row['Count'])} vehicles - {row['Percentage']:.1f}%)"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Price", f"${row['Avg_Price']:,.0f}")
            c2.metric("Avg Mileage", f"{row['Avg_Mileage']:,.0f} km")
            c3.metric("Avg Engine", f"{row['Avg_Engine']:.1f} L")
    
else:
    st.warning("⚠️ Not enough data.")
