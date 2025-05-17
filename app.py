import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import logging
import traceback
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    .stApp {
        background-color: #f5f8ff;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .stMetric label {
        color: #4B5563;
        font-weight: 600;
    }
    .stMetric .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #1E3A8A;
    }
    .chart-container {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ————— DATABASE CONNECTION —————
@st.cache_resource
def get_connection():
    try:
        warehouse = "postgresql://user1:IgiGkzZBqdpyGfr3F5aclDLHdzwU4iX9@dpg-d0k14ct6ubrc73asal2g-a.oregon-postgres.render.com/datawarehouse_j4rf"
        # Add connection pooling and timeout parameters
        engine = create_engine(
            warehouse,
            client_encoding='utf8',
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"connect_timeout": 30}
        )
        return engine
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

@st.cache_data(ttl=3600)  
def load_data():
    try:
        engine = get_connection()
        if engine is None:
            st.error("Failed to connect to database. Please check your connection string.")
            return pd.DataFrame()
        
        # Create a new connection for each query
        with engine.connect() as conn:
            # Execute query with error handling
            query = "SELECT * FROM cleaned_sales_data;"
            result = conn.execute(text(query))
            
            # Convert to DataFrame
            df = pd.DataFrame(result.mappings().all())
            
        if df.empty:
            logger.warning("Query returned no data.")
            return pd.DataFrame()
        
        logger.info(f"Successfully loaded {len(df)} rows of data.")
        
        logger.info(f"Columns before processing: {df.columns.tolist()}")
        logger.info(f"Sample date values: {df['Order Date'].head().tolist() if 'Order Date' in df.columns else 'No Order Date column'}")
        
        logger.info(f"Sample 'Price Each' values: {df['Price Each'].head().tolist() if 'Price Each' in df.columns else 'No Price Each column'}")
        
        # ——— Parsing & cleanup ———
        # Handle the date column
        df['Order Date'] = pd.to_datetime(
            df['Order Date'],
            errors='coerce'
        )
        
        # For Quantity Ordered
        df['Quantity Ordered'] = df['Quantity Ordered'].astype(str).str.replace(',', '').str.strip()
        df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce')
        
        # For Price Each
        if 'Price Each' in df.columns:
            # If price is a string with currency symbols or commas
            if df['Price Each'].dtype == object:
                df['Price Each'] = df['Price Each'].astype(str).str.replace('$', '', regex=False).str.replace(',', '').str.strip()
            df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce')
        
        # Count NaN values after conversion
        date_na_count = df['Order Date'].isna().sum()
        qty_na_count = df['Quantity Ordered'].isna().sum()
        price_na_count = df['Price Each'].isna().sum() if 'Price Each' in df.columns else 0
        
        logger.info(f"NaN counts - Order Date: {date_na_count}, Quantity: {qty_na_count}, Price: {price_na_count}")
        
        # If Price Each is all NaN, show sample raw values and add fallback
        if 'Price Each' in df.columns and df['Price Each'].isna().all():
            logger.warning("All Price Each values are NaN. Using fallback value of 1.0")
            # For demonstration purposes, use a fallback value
            df['Price Each'] = 1.0
        
        # Drop rows with NaN in critical columns
        df.dropna(subset=['Order Date', 'Quantity Ordered'], inplace=True)
        
        # Derived columns
        df['Revenue'] = df['Quantity Ordered'] * df['Price Each']
        df['Month'] = df['Order Date'].dt.strftime('%Y-%m')  # More standard month format
        
        # Extract city from address
        def extract_city(address):
            if not isinstance(address, str):
                return 'Unknown'
            parts = address.split(',')
            if len(parts) >= 2:
                return parts[1].strip()
            return 'Unknown'
        
        df['City'] = df['Purchase Address'].apply(extract_city)
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

# ————— LOAD DATA —————
with st.spinner("Loading sales data..."):
    df = load_data()

# ————— HANDLE EMPTY DATA —————
if df.empty:
    st.error("🚨 No data was loaded from the database.")
    st.info("Possible causes:")
    st.markdown("""
    1. Database connection issues
    2. The query returned no results
    3. Date parsing errors removed all valid records
    """)
    st.stop()

# Check for valid dates after parsing
valid_dates = df['Order Date'].dropna()
if valid_dates.empty:
    st.error("🚨 No valid Order Date values available after parsing.")
    st.info("Check the format of your date values in the database.")
    st.write("Sample of raw data:")
    st.write(df.head())
    st.stop()

# ————— SIDEBAR CONTROLS —————
with st.sidebar:
    st.title("Sales Dashboard")
    
    # Display data summary in sidebar with custom styling
    st.sidebar.markdown("""
    <div style="background-color: #e0f7fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        <h3 style="color: #006064; margin-top: 0;">Data Summary</h3>
        <ul style="list-style-type: none; padding-left: 0;">
            <li>📊 <b>Total Records:</b> {}</li>
            <li>📅 <b>Date Range:</b> {} to {}</li>
            <li>🛒 <b>Products:</b> {}</li>
            <li>🏙️ <b>Cities:</b> {}</li>
        </ul>
    </div>
    """.format(
        len(df),
        valid_dates.min().date(),
        valid_dates.max().date(),
        len(df['Product'].unique()),
        len(df['City'].unique())
    ), unsafe_allow_html=True)
    
    st.header("Filters")
    
    # Date range selector
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()
    
    date_range = st.date_input(
        "Order Date range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Ensure we have both start and end dates
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range[0]  
        end_date = date_range[0]
    
    # Product selector
    products = st.multiselect(
        "Select Product(s)",
        options=sorted(df['Product'].unique()),
        default=None,  # Start with none selected
        help="Leave empty to include all products"
    )
    
    # If nothing selected, include all products
    if not products:
        products = df['Product'].unique()
    
    # City selector  
    cities = st.multiselect(
        "Select City(ies)",
        options=sorted(df['City'].unique()),
        default=None,  # Start with none selected
        help="Leave empty to include all cities"
    )
    
    # If nothing selected, include all cities
    if not cities:
        cities = df['City'].unique()

    
    color_theme = st.selectbox(
        "Chart Color Theme",
        options=["Viridis", "Blues", "Plasma", "Inferno", "Magma", "Cividis", "Rainbow"],
        index=0
    )

# ————— APPLY FILTERS —————
mask = (
    (df['Order Date'].dt.date >= start_date) &
    (df['Order Date'].dt.date <= end_date) &
    (df['Product'].isin(products)) &
    (df['City'].isin(cities))
)
filtered = df[mask]

# Handle empty filtered data
if filtered.empty:
    st.warning("⚠️ No data matches your filter criteria. Please adjust your filters.")
    st.stop()

# ————— DASHBOARD LAYOUT —————
st.markdown(f"""
<h1 style="text-align: center; color: #1E3A8A; margin-bottom: 30px;">
    📊 Sales Dashboard
</h1>
""", unsafe_allow_html=True)

# Display filtered data summary
st.markdown(
    f"""<p style="text-align: center; font-size: 16px; color: #4B5563; margin-bottom: 30px;">
    Showing <span style="font-weight: bold; color: #1E40AF;">{len(filtered):,}</span> orders 
    from <span style="font-weight: bold; color: #1E40AF;">{start_date}</span> to 
    <span style="font-weight: bold; color: #1E40AF;">{end_date}</span>
    </p>""",
    unsafe_allow_html=True
)

# Create metric summary
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_revenue = filtered['Revenue'].sum()
    st.markdown(f"""
    <div class="stMetric">
        <label>Total Revenue</label>
        <p class="metric-value" style="color: #047857;">${total_revenue:,.2f}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_orders = filtered['Order ID'].nunique()
    st.markdown(f"""
    <div class="stMetric">
        <label>Total Orders</label>
        <p class="metric-value" style="color: #1D4ED8;">{total_orders:,}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    st.markdown(f"""
    <div class="stMetric">
        <label>Avg Order Value</label>
        <p class="metric-value" style="color: #7E22CE;">${avg_order_value:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    total_units = filtered['Quantity Ordered'].sum()
    st.markdown(f"""
    <div class="stMetric">
        <label>Units Sold</label>
        <p class="metric-value" style="color: #B91C1C;">{total_units:,}</p>
    </div>
    """, unsafe_allow_html=True)

# Create two columns for better space usage
col1, col2 = st.columns(2)

# 1. Top Products by Quantity 
with col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Top Products by Quantity Sold")
    
    qty_by_prod = (
        filtered
        .groupby('Product')['Quantity Ordered']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    
    fig = px.bar(
        qty_by_prod,
        orientation='h',
        color=qty_by_prod.values,
        color_continuous_scale=color_theme,
        labels={'value': 'Quantity Sold', 'index': 'Product'},
        title=''
    )
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Quantity Sold",
        yaxis_title="",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 2. Revenue by Product
with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Revenue by Product")
    
    rev_by_prod = (
        filtered
        .groupby('Product')['Revenue']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    
    fig = px.bar(
        rev_by_prod,
        orientation='h',
        color=rev_by_prod.values,
        color_continuous_scale=color_theme,
        labels={'value': 'Revenue ($)', 'index': 'Product'},
        title=''
    )
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Revenue ($)",
        yaxis_title="",
        coloraxis_showscale=False
    )
    fig.update_traces(hovertemplate='$%{x:,.2f}')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 3. Monthly Revenue Trend with Plotly
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.subheader("Monthly Revenue Trend")

if not filtered.empty:
    # Create a dataframe with months for plotting
    monthly_data = (
        filtered
        .groupby(filtered['Order Date'].dt.strftime('%Y-%m'))['Revenue']
        .sum()
        .reset_index()
    )
    monthly_data.columns = ['Month', 'Revenue']
    
    fig = px.line(
        monthly_data, 
        x='Month', 
        y='Revenue',
        markers=True,
        line_shape='spline',
        color_discrete_sequence=[px.colors.sequential.Viridis[7]]  # Use direct color reference
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_data['Month'],
            y=monthly_data['Revenue'],
            fill='tozeroy',
            fillcolor='rgba(64, 145, 108, 0.2)',  # Use direct RGBA color
            line=dict(width=0),
            showlegend=False
        )
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
        hovermode="x unified"
    )
    fig.update_traces(hovertemplate='$%{y:,.2f}')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No monthly data available for the selected filters.")
st.markdown('</div>', unsafe_allow_html=True)

# 4. Orders Over Time
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.subheader("Orders Over Time")

if not filtered.empty:
    try:
        orders_data = (
            filtered
            .groupby(filtered['Order Date'].dt.date)['Order ID']
            .nunique()
            .reset_index()
        )
        orders_data.columns = ['Date', 'Order Count']
        
        fig = px.line(
            orders_data, 
            x='Date', 
            y='Order Count',
            markers=True,
            color_discrete_sequence=[px.colors.sequential.Viridis[5]]  # Use direct color reference
        )
        
        fig.add_trace(
            go.Scatter(
                x=orders_data['Date'],
                y=orders_data['Order Count'],
                fill='tozeroy',
                fillcolor='rgba(72, 108, 164, 0.2)',  # Use direct RGBA color
                line=dict(width=0),
                showlegend=False
            )
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Date",
            yaxis_title="Number of Orders",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logger.error(f"Error creating Orders Over Time chart: {e}")
        st.warning("Could not generate Orders Over Time chart due to data issue.")
else:
    st.info("No order date data available for the selected filters.")
st.markdown('</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

# 5. Revenue by City
with col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Revenue by City")
    
    if not filtered.empty:
        rev_by_city = (
            filtered
            .groupby('City')['Revenue']
            .sum()
            .sort_values(ascending=True)
            .reset_index()
        )
        
        fig = px.bar(
            rev_by_city,
            y='City',
            x='Revenue',
            orientation='h',
            color='Revenue',
            color_continuous_scale=color_theme,
            labels={'Revenue': 'Revenue ($)', 'City': ''},
            title=''
        )
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Revenue ($)",
            coloraxis_showscale=False
        )
        fig.update_traces(hovertemplate='$%{x:,.2f}')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No city data available for the selected filters.")
    st.markdown('</div>', unsafe_allow_html=True)

# 6. Revenue by Day of Week
with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Revenue by Day of Week")
    
    try:
        weekday_mapping = {
            0: 'Monday',
            1: 'Tuesday',
            2: 'Wednesday',
            3: 'Thursday',
            4: 'Friday',
            5: 'Saturday',
            6: 'Sunday'
        }
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        day_revenue = filtered.copy()
        day_revenue['Weekday'] = day_revenue['Order Date'].dt.dayofweek.map(weekday_mapping)
        
        weekday_data = (
            day_revenue
            .groupby('Weekday')['Revenue']
            .sum()
            .reindex(weekday_order)
            .reset_index()
        )
        
        fig = px.bar(
            weekday_data,
            x='Weekday',
            y='Revenue',
            color='Revenue',
            color_continuous_scale=color_theme,
            category_orders={"Weekday": weekday_order},
            labels={'Revenue': 'Revenue ($)', 'Weekday': ''},
            title=''
        )
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Revenue ($)",
            coloraxis_showscale=False
        )
        fig.update_traces(hovertemplate='$%{y:,.2f}')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logger.error(f"Error creating day of week chart: {e}")
        st.info("Could not generate Revenue by Day of Week chart.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.subheader("Data Diagnostics")
with st.expander("Show Data Samples and Diagnostics"):
    st.write("Column Information:")
    st.write(filtered.dtypes)
    
    st.write("Data Sample (first 10 rows):")
    st.dataframe(filtered.head(10))
    
    if 'Price Each' in filtered.columns:
        st.write("Price Each Statistics:")
        st.write(filtered['Price Each'].describe())
    
    if 'Quantity Ordered' in filtered.columns:
        st.write("Quantity Ordered Statistics:")
        st.write(filtered['Quantity Ordered'].describe())
st.markdown('</div>', unsafe_allow_html=True)

# Add footer with timestamp
st.markdown("---")
st.markdown(
    f"""<div style="text-align: center; color: #6B7280; font-size: 12px;">
    Dashboard last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>""",
    unsafe_allow_html=True
)
