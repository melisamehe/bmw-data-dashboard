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