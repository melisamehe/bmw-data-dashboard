st.markdown("### 🎯 4. Price vs Mileage Scatter")

scatter_df = df_filtered.dropna(subset=['Price_USD','Mileage_KM'])
fig_scatter = px.scatter(
    scatter_df,
    x='Mileage_KM',
    y='Price_USD',
    color='Fuel_Type',
    size='Sales_Volume',
    hover_data=['Model','Year','Region', 'Price_USD', 'Mileage_KM', 'Sales_Volume'],
    title="Price vs Mileage (size = Sales volume)",
    color_discrete_sequence=px.colors.qualitative.Set2,
    size_max=40
)
fig_scatter.update_traces(marker=dict(line=dict(width=1, color='black'), opacity=0.8))
fig_scatter.update_layout(
    xaxis_title="Mileage (KM)",
    yaxis_title="Price (USD)",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#1e1e1e",
    font=dict(color="white"),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial", font_color="black")  
)
st.plotly_chart(fig_scatter, use_container_width=True)



st.markdown("### 📊 5. Top-Selling Models (Horizontal Bar)")

model_sales = df_filtered.groupby('Model', as_index=False)['Sales_Volume'].sum().sort_values(by='Sales_Volume', ascending=False)
top_n = st.slider("Number of top models to show", 5, 25, 10, key="topn_slider")
fig_bar = px.bar(
    model_sales.head(top_n),
    x='Sales_Volume',
    y='Model',
    orientation='h',
    text='Sales_Volume',
    color='Sales_Volume',
    color_continuous_scale='Blues'
)
fig_bar.update_traces(texttemplate='%{text:,}', textposition='outside')
fig_bar.update_layout(
    yaxis={'categoryorder':'total ascending'},
    xaxis_title="Sales Volume",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#1e1e1e",
    font=dict(color="white"),
    coloraxis_showscale=False
)
st.plotly_chart(fig_bar, use_container_width=True)



st.markdown("### 🧭 6. Parallel Coordinates: Engine, Price, Mileage")

pc_df = df_filtered[['Engine_Size_L','Price_USD','Mileage_KM','Sales_Volume']].dropna()
sample_size = min(len(pc_df), 1000)
if sample_size > 0:
    pc_sample = pc_df.sample(sample_size, random_state=1)
    fig_pc = px.parallel_coordinates(
        pc_sample,
        color='Sales_Volume',
        dimensions=['Engine_Size_L','Price_USD','Mileage_KM'],
        labels={'Engine_Size_L':'Engine (L)','Price_USD':'Price (USD)','Mileage_KM':'Mileage (KM)'},
        color_continuous_scale=px.colors.sequential.Viridis,
        range_color=[pc_sample['Sales_Volume'].min(), pc_sample['Sales_Volume'].max()]
    )
    fig_pc.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1e1e1e",
        font=dict(color="white")
    )
    st.plotly_chart(fig_pc, use_container_width=True)
else:
    st.info("No data for parallel coordinates with current filters.")
