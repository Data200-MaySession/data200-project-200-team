import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Load Excel file
file_path = r"C:\Users\dell\.cache\kagglehub\datasets\wellkilo\supermarket-dataset\versions\1\supermarket.xlsx"
df = pd.read_excel(file_path)

# ✅ Clean 'Sales' column: remove currency symbols and convert to float
df['Sales'] = df['Sales'].replace(r'[\$,]', '', regex=True).astype(float)

st.title("💰 Supermarket Profit Prediction")

# Sidebar Inputs
st.sidebar.header("Order Details")

category = st.sidebar.selectbox("Category", sorted(df['Category'].unique()))
region = st.sidebar.selectbox("Region", sorted(df['Region'].unique()))
segment = st.sidebar.selectbox("Segment", sorted(df['Segment'].unique()))
ship_mode = st.sidebar.selectbox("Ship Mode", sorted(df['Ship Mode'].unique()))
discount = st.sidebar.slider("Discount", float(df['Discount'].min()), float(df['Discount'].max()), float(df['Discount'].mean()))
quantity = st.sidebar.number_input("Quantity", min_value=int(df['Quantity'].min()), max_value=int(df['Quantity'].max()), value=int(df['Quantity'].mean()))
sales = st.sidebar.number_input("Sales", min_value=float(df['Sales'].min()), max_value=float(df['Sales'].max()), value=float(df['Sales'].mean()))

# Create input DataFrame
input_df = pd.DataFrame({
    'Category': [category],
    'Region': [region],
    'Segment': [segment],
    'Ship Mode': [ship_mode],
    'Discount': [discount],
    'Quantity': [quantity],
    'Sales': [sales]
})

model_path = "supermarket_profit_pipeline.pkl"
model_path = os.path.normpath(model_path)

# Load the trained model pipeline
model_pipeline = None

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model_pipeline = pickle.load(f)
else:
    st.error(f"❌ Model file '{model_path}' not found. Please ensure the model is trained and the file exists.")

# Prediction button
if st.sidebar.button("Predict Profit"):
    if model_pipeline is not None:
        try:
            profit_pred = model_pipeline.predict(input_df)[0]
            st.subheader("Prediction Result")
            st.info(f"💵 Predicted Profit: {profit_pred:.2f}")
            # --- Add Graphs Below ---
            st.subheader("📊 Data Visualizations")

            # 1. Sales vs Profit (if 'Profit' exists in df)
            if 'Profit' in df.columns:
                fig1, ax1 = plt.subplots()

                # Downsample for plotting if too many points
                plot_df = df.sample(n=min(500, len(df)), random_state=42) if len(df) > 500 else df

                sns.scatterplot(
                    data=plot_df, x='Sales', y='Profit', hue='Category', ax=ax1, alpha=0.5, s=25, edgecolor=None
                )

                ax1.set_title('Sales vs Profit by Category')
                ax1.grid(True, linestyle='--', alpha=0.3)

                # Format y-axis as currency with fewer ticks
                ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
                ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))

                fig1.tight_layout()
                st.pyplot(fig1)

            # 2. Profit by Region (if 'Profit' exists)
            if 'Profit' in df.columns:
                fig2, ax2 = plt.subplots()

                sns.boxplot(
                    data=df, x='Region', y='Profit', ax=ax2, showfliers=False
                )

                # Downsample for stripplot to reduce clutter
                strip_df = df.sample(n=min(200, len(df)), random_state=42) if len(df) > 200 else df

                sns.stripplot(
                    data=strip_df, x='Region', y='Profit', ax=ax2, color='gray', alpha=0.2, jitter=True, size=2
                )

                ax2.set_title('Profit Distribution by Region')
                ax2.grid(True, linestyle='--', alpha=0.3)

                # Format y-axis as currency with fewer ticks
                ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
                ax2.yaxis.set_major_locator(MaxNLocator(nbins=8))

                fig2.tight_layout()
                st.pyplot(fig2)

            # 3. Sales Distribution
            fig3, ax3 = plt.subplots()
            sns.histplot(df['Sales'], bins=30, kde=True, ax=ax3)
            ax3.set_title('Sales Distribution')
            st.pyplot(fig3)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
    else:
        st.error("❌ Model is not loaded. Cannot make predictions.")