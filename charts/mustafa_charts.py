## 1. Yearly Total Sales Trend
sales_by_year = df_filtered.groupby('Year', as_index=False)['Sales_Volume'].sum().sort_values('Year')
fig_line = px.line(sales_by_year, x='Year', y='Sales_Volume', markers=True)
fig_line.update_traces(hovertemplate='Year: %{x}<br>Sales: %{y:,}')
st.plotly_chart(fig_line, use_container_width=True)

## 🗂️ 2. Region → Model Sales Treemap
treemap_df = df_filtered.groupby(['Region', 'Model'], as_index=False)['Sales_Volume'].sum()
fig_treemap = px.treemap(treemap_df, path=['Region', 'Model'], values='Sales_Volume')
st.plotly_chart(fig_treemap, use_container_width=True)

## 💰 3. Price Distribution by Fuel Type
box_df = df_filtered[['Fuel_Type', 'Price_USD']].dropna()
fig_box = px.box(box_df, x='Fuel_Type', y='Price_USD', points="outliers")
st.plotly_chart(fig_box, use_container_width=True)


# ----------------------------------------------------------
# MACHINE LEARNING 
# ----------------------------------------------------------
st.markdown("---")
st.header("Random Forest Learning")

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

st.subheader("Price Prediction Model")

df_ml = df_filtered.copy()

required = ['Price_USD', 'Mileage_KM', 'Engine_Size_L', 'Year', 'Transmission']

if not all(col in df_ml.columns for col in required):
    st.warning("Dataset missing required columns.")
else:

    df_ml = df_ml.dropna(subset=required)

    if len(df_ml) < 50:
        st.warning("Need more data for model training.")
    else:
        le = LabelEncoder()
        df_ml['Transmission_enc'] = le.fit_transform(df_ml['Transmission'].astype(str))

        feature_cols = ['Year', 'Mileage_KM', 'Engine_Size_L', 'Transmission_enc']

        X = df_ml[feature_cols]
        y = df_ml['Price_USD']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{r2:.3f}")
        col2.metric("MSE", f"{mse:,.0f}")
        col3.metric("MAE", f"{mae:,.0f}")

       
        fig_reg = px.scatter(
            x=y_test,
            y=y_pred,
            labels={'x': 'Actual Price', 'y': 'Predicted Price'},
            opacity=0.6
        )
        fig_reg.add_shape(
            type="line",
            x0=y_test.min(), y0=y_test.min(),
            x1=y_test.max(), y1=y_test.max(),
            line=dict(color="red", dash="dash")
        )
        st.plotly_chart(fig_reg, use_container_width=True)
        
        # ----------------------------------------------------------
        # CORRELATION ANALYSIS 
        # ----------------------------------------------------------
        import seaborn as sns
        import matplotlib.pyplot as plt

        st.subheader("📉 Correlation Matrix")

        corr_df = df_ml.copy()
        corr_df['Transmission_enc'] = corr_df['Transmission'].astype('category').cat.codes
        numeric_cols = corr_df.select_dtypes(include=['int64', 'float64', 'Int64']).columns

        corr_matrix = corr_df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdBu",
            center=0,
            linewidths=0.5,
            cbar=True
        )
        plt.title("Correlation Matrix", fontsize=14)
        st.pyplot(fig)

        st.info("""
        **Interpretation:**  
        If all correlations with *Price_USD* remain close to zero (|r| < 0.05),  
        the dataset does not contain meaningful patterns for price prediction.  
        In such cases, low or negative R² scores are expected and normal.
        """)