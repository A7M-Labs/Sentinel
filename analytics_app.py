import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Sentinel Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mock data - in a real scenario, this would be fetched from your partner's backend
def generate_mock_data():
    # Create date range for the last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Event types
    event_types = ['Unauthorized Entry', 'Unattended Object', 'Suspicious Behavior', 'Motion Detected']
    
    # Camera locations
    cameras = ['Front Door', 'Lobby', 'Parking Lot', 'Back Entrance', 'Storage Room']
    
    # Generate random events
    data = []
    for date in date_range:
        # Random number of events per hour (0-5)
        num_events = np.random.randint(0, 6)
        for _ in range(num_events):
            event_type = np.random.choice(event_types)
            camera = np.random.choice(cameras)
            confidence = round(np.random.uniform(0.6, 0.99), 2)
            
            data.append({
                'timestamp': date,
                'event_type': event_type,
                'camera': camera,
                'confidence': confidence
            })
    
    return pd.DataFrame(data)

# Function to create analytics dashboard
def create_analytics_dashboard():
    df = generate_mock_data()
    
    # Add date column for easier filtering
    df['date'] = df['timestamp'].dt.date
    
    # Sidebar for filtering
    st.sidebar.title('Sentinel Analytics')
    
    # Date filter
    date_range = st.sidebar.date_input(
        "Date Range",
        [df['date'].min(), df['date'].max()]
    )
    
    # Event type filter
    event_types = df['event_type'].unique().tolist()
    selected_event_types = st.sidebar.multiselect(
        "Event Types",
        event_types,
        default=event_types
    )
    
    # Camera filter
    cameras = df['camera'].unique().tolist()
    selected_cameras = st.sidebar.multiselect(
        "Cameras",
        cameras,
        default=cameras
    )
    
    # Apply filters
    filtered_df = df[
        (df['date'] >= date_range[0]) & 
        (df['date'] <= date_range[1]) &
        (df['event_type'].isin(selected_event_types)) &
        (df['camera'].isin(selected_cameras))
    ]
    
    # Main area
    st.title("Security Event Analytics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", len(filtered_df))
    with col2:
        st.metric("Avg. Confidence", f"{filtered_df['confidence'].mean():.2f}")
    with col3:
        st.metric("Unique Cameras", len(filtered_df['camera'].unique()))
    with col4:
        st.metric("Event Types", len(filtered_df['event_type'].unique()))
    
    # Charts
    st.subheader("Events Over Time")
    events_by_time = filtered_df.groupby(filtered_df['timestamp'].dt.date).size().reset_index(name='count')
    fig1 = px.line(events_by_time, x='timestamp', y='count', title='Events by Day')
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Events by Type")
        events_by_type = filtered_df['event_type'].value_counts().reset_index()
        events_by_type.columns = ['event_type', 'count']
        fig2 = px.pie(events_by_type, values='count', names='event_type', title='Distribution of Event Types')
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("Events by Camera")
        events_by_camera = filtered_df['camera'].value_counts().reset_index()
        events_by_camera.columns = ['camera', 'count']
        fig3 = px.bar(events_by_camera, x='camera', y='count', title='Events by Camera Location')
        st.plotly_chart(fig3, use_container_width=True)
    
    # Confidence scores distribution
    st.subheader("Confidence Score Distribution")
    fig4 = px.histogram(filtered_df, x='confidence', nbins=20, title='Distribution of Confidence Scores')
    st.plotly_chart(fig4, use_container_width=True)
    
    # Display raw data
    st.subheader("Raw Event Data")
    st.dataframe(filtered_df.sort_values('timestamp', ascending=False), height=300)
    
    # Export option
    if st.button('Export Data (CSV)'):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='sentinel_analytics.csv',
            mime='text/csv',
        )

# Run the dashboard
create_analytics_dashboard() 