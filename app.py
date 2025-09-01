#!/usr/bin/env python3
"""
Fraud Detection Streamlit Dashboard

Run Instructions:
- streamlit run app.py
- Upload new_transactions.csv in the dashboard
- View predictions and interactive charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .suspicious-row {
        background-color: #ffebee;
    }
    .normal-row {
        background-color: #e8f5e8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    """Load trained model and preprocessing objects"""
    try:
        with open('fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
            
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        # Try to load implemented features summary
        implemented_features = None
        try:
            with open('implemented_features.pkl', 'rb') as f:
                implemented_features = pickle.load(f)
        except FileNotFoundError:
            pass  # It's okay if this file doesn't exist for backward compatibility
        
        return model, scaler, encoder, feature_columns, implemented_features
    except FileNotFoundError as e:
        st.error(f"Model files not found! Please run 'python train_model.py' first.")
        st.error(f"Missing file: {e.filename}")
        return None, None, None, None, None

def create_enhanced_features_for_prediction(df):
    """Create enhanced features for new transaction data (simplified for prediction)"""
    # Since we don't have historical data for new transactions, we'll use simplified versions
    processed_df = df.copy()
    
    # Create user aggregates as if we have some historical data (simplified approach)
    # In a real system, these would be looked up from a user profile database
    
    # Transaction frequency features (simplified)
    processed_df['txn_count'] = 1  # Default for new transactions
    processed_df['avg_amount'] = processed_df['amount']  # Use current amount as baseline
    processed_df['std_amount'] = 0  # Default to 0 for single transaction
    processed_df['active_days'] = 1  # Default to 1 day
    processed_df['ratio_current_to_avg'] = 1  # Default to 1 for single transaction
    
    # Merchant/Geographic features
    processed_df['is_merchant_dest'] = processed_df['nameDest'].str.startswith('M').astype(int)
    processed_df['unique_destinations'] = 1  # Default to 1 for new transactions
    processed_df['destination_diversity_ratio'] = 1  # Default to 1 for single transaction
    processed_df['new_destination_flag'] = 1  # Assume new destination for prediction
    
    # Simulate geographic coordinates for destinations (consistent per destination)
    dest_to_coords = {}
    for dest in processed_df['nameDest'].unique():
        # Use destination name hash for consistent coordinates
        seed_val = hash(dest) % (2**32)
        np.random.seed(seed_val)
        dest_to_coords[dest] = (np.random.uniform(-90, 90), np.random.uniform(-180, 180))
    
    processed_df['dest_lat'] = processed_df['nameDest'].map(lambda x: dest_to_coords[x][0])
    processed_df['dest_lon'] = processed_df['nameDest'].map(lambda x: dest_to_coords[x][1])
    processed_df['distance_from_common'] = 0  # Default to 0 for single transaction
    
    return processed_df

def preprocess_data(df, encoder, feature_columns):
    """Preprocess uploaded data to match training format with enhanced features"""
    
    # Create a copy to avoid modifying original
    processed_df = df.copy()
    
    # Original feature engineering
    processed_df['balance_diff_orig'] = processed_df['oldbalanceOrg'] - processed_df['newbalanceOrig']
    processed_df['balance_diff_dest'] = processed_df['newbalanceDest'] - processed_df['oldbalanceDest']
    processed_df['amount_to_balance_ratio'] = processed_df['amount'] / (processed_df['oldbalanceOrg'] + 1)
    
    # Handle zero balances
    processed_df['orig_zero_balance'] = (processed_df['oldbalanceOrg'] == 0).astype(int)
    processed_df['dest_zero_balance'] = (processed_df['oldbalanceDest'] == 0).astype(int)
    
    # Encode transaction type
    processed_df['type_encoded'] = encoder.transform(processed_df['type'])
    
    # Create enhanced features
    processed_df = create_enhanced_features_for_prediction(processed_df)
    
    # Select only the features used in training
    X = processed_df[feature_columns].copy()
    
    # Handle any NaN values
    X = X.fillna(0)
    
    return X

def create_visualizations(df_results):
    """Create interactive visualizations"""
    
    # Count of Fraud vs Normal
    fraud_counts = df_results['Prediction'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Fraud Detection Summary")
        fig_pie = px.pie(
            values=fraud_counts.values, 
            names=fraud_counts.index,
            title="Transaction Classification",
            color_discrete_map={'Suspicious': '#ff4444', 'Normal': '#44ff44'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("💰 Transaction Amounts")
        fig_hist = px.histogram(
            df_results, 
            x='amount', 
            color='Prediction',
            nbins=20,
            title="Amount Distribution by Classification",
            color_discrete_map={'Suspicious': '#ff4444', 'Normal': '#44ff44'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Additional visualizations
    if 'Suspicious' in fraud_counts.index:
        suspicious_data = df_results[df_results['Prediction'] == 'Suspicious']
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🚨 Fraud Probability Distribution")
            fig_prob = px.histogram(
                suspicious_data, 
                x='Fraud Probability (%)',
                nbins=10,
                title="Fraud Probability for Suspicious Transactions"
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        
        with col4:
            st.subheader("🏪 Transaction Types")
            type_counts = df_results['type'].value_counts()
            fig_types = px.bar(
                x=type_counts.index,
                y=type_counts.values,
                title="Transaction Types Distribution"
            )
            fig_types.update_xaxes(title_text="Transaction Type")
            fig_types.update_yaxes(title_text="Count")
            st.plotly_chart(fig_types, use_container_width=True)


def style_dataframe(df):
    """Apply conditional styling to dataframe"""
    def highlight_predictions(row):
        if row['Prediction'] == 'Suspicious':
            return ['background-color: #ffebee'] * len(row)
        else:
            return ['background-color: #e8f5e8'] * len(row)
    
    return df.style.apply(highlight_predictions, axis=1)

def main():
    """Main Streamlit app"""
    
    st.markdown('<h1 class="main-header">🔍 Fraud Detection System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Upload transaction data to detect potential fraudulent activities using machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    model, scaler, encoder, feature_columns, implemented_features = load_models()
    
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("📊 Enhanced Model Information")
    st.sidebar.success("✅ Enhanced model loaded successfully!")
    st.sidebar.info("Upload a CSV file with transaction data to get fraud predictions using advanced features.")
    
    st.sidebar.markdown("""
    **Required CSV columns:**
    - step, type, amount
    - nameOrig, nameDest
    - oldbalanceOrg, newbalanceOrig
    - oldbalanceDest, newbalanceDest
    """)
    
    # Show implemented features if available
    if implemented_features:
        st.sidebar.header("🚀 Enhanced Features")
        
        # Map technical feature names to display names for consistency
        feature_display_mapping = {
            'txn_count': 'Txn Count (1h)',
            'avg_amount': 'Avg Amount',
            'std_amount': 'Std Amount', 
            'active_days': 'Active Days',
            'ratio_current_to_avg': 'Amount Ratio',
            'is_merchant_dest': 'Is Merchant',
            'unique_destinations': 'Unique Destinations',
            'destination_diversity_ratio': 'Destination Diversity',
            'new_destination_flag': 'New Destination',
            'dest_lat': 'Dest Latitude',
            'dest_lon': 'Dest Longitude',
            'distance_from_common': 'Distance From Common'
        }
        
        for category, features in implemented_features.items():
            with st.sidebar.expander(category):
                for feature in features:
                    display_name = feature_display_mapping.get(feature, feature)
                    st.write(f"• {display_name}")
                    if feature in feature_display_mapping:
                        st.caption(f"  ({feature})")  # Show technical name as caption
    
    # File uploader
    st.header("📁 Upload Transaction Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload new_transactions.csv or any CSV with the required columns"
    )
    
    if uploaded_file is not None:
        try:
            # Load uploaded data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Show sample of uploaded data
            with st.expander("👀 Preview Uploaded Data"):
                st.dataframe(df.head(10))
            
            # Check required columns
            required_columns = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                              'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
                st.stop()
            
            # Preprocess data
            with st.spinner("🔄 Processing data with enhanced features..."):
                X_processed = preprocess_data(df, encoder, feature_columns)
                X_scaled = scaler.transform(X_processed)
                
                # Make predictions
                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)[:, 1]
            
            # Create results dataframe
            df_results = df.copy()
            df_results['Fraud Probability (%)'] = (probabilities * 100).round(2)
            
            # Mark as suspicious if prediction == 1 OR fraud probability > 20%
            df_results['Prediction'] = [
                'Suspicious' if (pred == 1 or prob > 0.20) else 'Normal' 
                for pred, prob in zip(predictions, probabilities)
            ]
            
            # Add all enhanced features to results for transparency
            enhanced_feature_mapping = {
                'txn_count': 'Txn Count (1h)',
                'avg_amount': 'Avg Amount',
                'std_amount': 'Std Amount',
                'active_days': 'Active Days',
                'ratio_current_to_avg': 'Amount Ratio',
                'is_merchant_dest': 'Is Merchant',
                'unique_destinations': 'Unique Destinations',
                'destination_diversity_ratio': 'Destination Diversity',
                'new_destination_flag': 'New Destination',
                'dest_lat': 'Dest Latitude',
                'dest_lon': 'Dest Longitude',
                'distance_from_common': 'Distance From Common'
            }
            
            # Add enhanced features to results dataframe
            for original_name, display_name in enhanced_feature_mapping.items():
                if original_name in X_processed.columns:
                    if original_name == 'is_merchant_dest':
                        df_results[display_name] = X_processed[original_name].map({0: 'Customer', 1: 'Merchant'})
                    elif original_name == 'new_destination_flag':
                        df_results[display_name] = X_processed[original_name].map({0: 'No', 1: 'Yes'})
                    elif original_name in ['dest_lat', 'dest_lon', 'distance_from_common']:
                        df_results[display_name] = X_processed[original_name].round(2)
                    elif original_name in ['avg_amount', 'std_amount']:
                        df_results[display_name] = X_processed[original_name].round(2)
                    elif original_name == 'ratio_current_to_avg':
                        df_results[display_name] = X_processed[original_name].round(2)
                    elif original_name == 'destination_diversity_ratio':
                        df_results[display_name] = X_processed[original_name].round(3)
                    else:
                        df_results[display_name] = X_processed[original_name]
            
            # Display metrics
            st.header("📈 Results Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_transactions = len(df_results)
                st.metric("Total Transactions", total_transactions)
            
            with col2:
                suspicious_count = (df_results['Prediction'] == 'Suspicious').sum()
                st.metric("Suspicious Transactions", suspicious_count, 
                         delta=f"{(suspicious_count/total_transactions*100):.1f}%")
            
            with col3:
                avg_fraud_prob = df_results['Fraud Probability (%)'].mean()
                st.metric("Avg Fraud Probability", f"{avg_fraud_prob:.2f}%")
            
            with col4:
                max_fraud_prob = df_results['Fraud Probability (%)'].max()
                st.metric("Max Fraud Probability", f"{max_fraud_prob:.2f}%")
            
            # Display results table
            st.header("📋 Detailed Results")
            
            # Create base display columns
            base_columns = ['nameOrig', 'nameDest', 'type', 'amount', 
                           'Fraud Probability (%)', 'Prediction']
            
            # Add transaction ID if not present
            if 'transaction_id' not in df_results.columns:
                df_results['transaction_id'] = ['TXN_' + str(i+1).zfill(4) for i in range(len(df_results))]
                base_columns.insert(0, 'transaction_id')
            
            # Define enhanced feature columns that are available
            available_enhanced_features = []
            for display_name in enhanced_feature_mapping.values():
                if display_name in df_results.columns:
                    available_enhanced_features.append(display_name)
            
            # Feature visibility toggle
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Table Display Options:**")
            with col2:
                show_enhanced = st.checkbox("Show Enhanced Features", value=True, key="show_enhanced_features")
            
            # Create final display columns based on toggle
            if show_enhanced and available_enhanced_features:
                display_columns = base_columns + available_enhanced_features
                st.info(f"📊 Displaying {len(available_enhanced_features)} enhanced features alongside base transaction data")
            else:
                display_columns = base_columns
                if available_enhanced_features:
                    st.info("💡 Enhanced features are available - check the box above to display them")
            
            # Create column selection for advanced users
            with st.expander("🔧 Advanced: Custom Column Selection"):
                st.markdown("**Select specific columns to display:**")
                
                # Group columns by category
                col_categories = {
                    "Transaction Basics": ['transaction_id', 'nameOrig', 'nameDest', 'type', 'amount'],
                    "Predictions": ['Fraud Probability (%)', 'Prediction'],
                    "Transaction Frequency": ['Txn Count (1h)', 'Avg Amount', 'Std Amount', 'Active Days', 'Amount Ratio'],
                    "Merchant & Destination": ['Is Merchant', 'Unique Destinations', 'Destination Diversity', 'New Destination'],
                    "Geographic": ['Dest Latitude', 'Dest Longitude', 'Distance From Common']
                }
                
                selected_columns = []
                for category, cols in col_categories.items():
                    available_cols = [col for col in cols if col in df_results.columns]
                    if available_cols:
                        st.markdown(f"**{category}:**")
                        cols_row = st.columns(min(4, len(available_cols)))
                        for idx, col in enumerate(available_cols):
                            with cols_row[idx % len(cols_row)]:
                                if st.checkbox(col, value=col in display_columns, key=f"col_{col}"):
                                    selected_columns.append(col)
                
                if st.button("Apply Custom Selection", key="apply_custom"):
                    if selected_columns:
                        display_columns = selected_columns
                        st.success(f"✅ Applied custom selection: {len(selected_columns)} columns")
                    else:
                        st.warning("⚠️ Please select at least one column")
            
            # Style and display dataframe
            styled_df = style_dataframe(df_results[display_columns])
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Show feature summary
            if show_enhanced and available_enhanced_features:
                st.markdown("**Enhanced Features in Results:**")
                feature_cols = st.columns(3)
                for idx, feature in enumerate(available_enhanced_features[:9]):  # Show first 9 features
                    with feature_cols[idx % 3]:
                        st.markdown(f"• {feature}")
            
            # Show enhanced features summary
            if implemented_features:
                with st.expander("🔍 Enhanced Features Used in Prediction"):
                    st.info("This model uses advanced features for better fraud detection:")
                    for category, features in implemented_features.items():
                        st.write(f"**{category}:**")
                        for feature in features:
                            display_name = enhanced_feature_mapping.get(feature, feature)
                            st.write(f"  • {display_name}")
                            if feature in enhanced_feature_mapping:
                                st.caption(f"    Technical name: {feature}")
            
            # Create visualizations
            st.header("📊 Interactive Visualizations")
            create_visualizations(df_results)
            
            # Enhanced feature visualizations
            if show_enhanced and available_enhanced_features:
                st.header("🚀 Enhanced Feature Analysis")
                
                # Create enhanced visualizations based on available features
                vis_cols = st.columns(2)
                
                with vis_cols[0]:
                    if 'Is Merchant' in df_results.columns:
                        st.subheader("🏪 Merchant vs Customer Transactions")
                        merchant_fraud = df_results.groupby(['Is Merchant', 'Prediction']).size().reset_index(name='Count')
                        fig_merchant = px.bar(
                            merchant_fraud, 
                            x='Is Merchant', 
                            y='Count', 
                            color='Prediction',
                            title="Fraud Detection by Destination Type",
                            color_discrete_map={'Suspicious': '#ff4444', 'Normal': '#44ff44'}
                        )
                        st.plotly_chart(fig_merchant, use_container_width=True)
                
                with vis_cols[1]:
                    if 'Amount Ratio' in df_results.columns:
                        st.subheader("💰 Amount Ratio Distribution")
                        fig_ratio = px.histogram(
                            df_results, 
                            x='Amount Ratio', 
                            color='Prediction',
                            nbins=20,
                            title="Amount Ratio vs Fraud Detection",
                            color_discrete_map={'Suspicious': '#ff4444', 'Normal': '#44ff44'}
                        )
                        st.plotly_chart(fig_ratio, use_container_width=True)
                
                # Geographic visualization if available
                if 'Dest Latitude' in df_results.columns and 'Dest Longitude' in df_results.columns:
                    st.subheader("🗺️ Geographic Distribution of Transactions")
                    fig_map = px.scatter_mapbox(
                        df_results.head(100),  # Limit to first 100 for performance
                        lat='Dest Latitude',
                        lon='Dest Longitude',
                        color='Prediction',
                        size='amount',
                        hover_data=['transaction_id', 'Fraud Probability (%)'],
                        mapbox_style='open-street-map',
                        title="Transaction Locations (First 100 transactions)",
                        color_discrete_map={'Suspicious': '#ff4444', 'Normal': '#44ff44'},
                        height=500
                    )
                    fig_map.update_layout(mapbox_zoom=1)
                    st.plotly_chart(fig_map, use_container_width=True)
            
            # Top suspicious transactions
            if suspicious_count > 0:
                st.header("🚨 Top Suspicious Transactions")
                top_suspicious = df_results[df_results['Prediction'] == 'Suspicious'].nlargest(5, 'Fraud Probability (%)')
                
                for idx, row in top_suspicious.iterrows():
                    with st.expander(f"⚠️ Transaction {row.get('transaction_id', idx)} - {row['Fraud Probability (%)']}% probability"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Basic Info:**")
                            st.write(f"**From:** {row['nameOrig']}")
                            st.write(f"**To:** {row['nameDest']}")
                            st.write(f"**Type:** {row['type']}")
                            st.write(f"**Amount:** ${row['amount']:,.2f}")
                        
                        with col2:
                            st.markdown("**Balance Info:**")
                            st.write(f"**Orig Balance:** ${row['oldbalanceOrg']:,.2f}")
                            st.write(f"**New Orig Balance:** ${row['newbalanceOrig']:,.2f}")
                            st.write(f"**Dest Balance:** ${row['oldbalanceDest']:,.2f}")
                            st.write(f"**New Dest Balance:** ${row['newbalanceDest']:,.2f}")
                        
                        with col3:
                            st.markdown("**Enhanced Features:**")
                            # Display available enhanced features for this transaction
                            if 'Is Merchant' in row:
                                st.write(f"**Destination Type:** {row['Is Merchant']}")
                            if 'Amount Ratio' in row:
                                st.write(f"**Amount Ratio:** {row['Amount Ratio']}")
                            if 'Unique Destinations' in row:
                                st.write(f"**Unique Destinations:** {row['Unique Destinations']}")
                            if 'New Destination' in row:
                                st.write(f"**New Destination:** {row['New Destination']}")
                            if 'Txn Count (1h)' in row:
                                st.write(f"**Transaction Count:** {row['Txn Count (1h)']}")
                            if 'Distance From Common' in row:
                                st.write(f"**Distance Anomaly:** {row['Distance From Common']:.2f}")
            
            # Download results
            st.header("💾 Download Results")
            csv_results = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_results,
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.error("Please check that your CSV file has the correct format and required columns.")
    
    else:
        st.info("👆 Please upload a CSV file to get started.")
        
        # Show sample format
        st.header("📋 Sample Data Format")
        sample_data = {
            'step': [1, 1, 1],
            'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT'],
            'amount': [9839.64, 181.0, 181.0],
            'nameOrig': ['C1231006815', 'C1305486145', 'C840083671'],
            'oldbalanceOrg': [170136.0, 181.0, 181.0],
            'newbalanceOrig': [160296.36, 0.0, 0.0],
            'nameDest': ['M1979787155', 'C553264065', 'C38997010'],
            'oldbalanceDest': [0.0, 0.0, 21182.0],
            'newbalanceDest': [0.0, 0.0, 0.0]
        }
        st.dataframe(pd.DataFrame(sample_data))

if __name__ == "__main__":
    main()