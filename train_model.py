#!/usr/bin/env python3
"""
Fraud Detection Model Training Script

Run Instructions:
- Train: python train_model.py
- This will create fraud_model.pkl, scaler.pkl, encoder.pkl files
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

def create_transaction_frequency_features(df):
    """Create transaction frequency features per user"""
    print("Creating transaction frequency features...")
    
    # Sort by user and step for time-based calculations
    df_sorted = df.sort_values(['nameOrig', 'step']).copy()
    
    # Calculate rolling features per user
    user_features = []
    
    for user in df_sorted['nameOrig'].unique():
        user_data = df_sorted[df_sorted['nameOrig'] == user].copy()
        
        # Transaction count in last 24 steps (representing 24 hours)
        user_data['txn_count_last_24h'] = user_data.rolling(window=24, min_periods=1)['step'].count()
        
        # Average amount in last 7 steps (representing 7 days)
        user_data['avg_amount_last_7d'] = user_data.rolling(window=7, min_periods=1)['amount'].mean()
        
        # Standard deviation of amount in last 7 steps
        user_data['std_dev_amount_last_7d'] = user_data.rolling(window=7, min_periods=1)['amount'].std().fillna(0)
        
        # Ratio of current transaction to average
        user_data['ratio_current_to_avg'] = user_data['amount'] / (user_data['avg_amount_last_7d'] + 1)
        
        user_features.append(user_data)
    
    # Combine all user features
    enhanced_df = pd.concat(user_features, ignore_index=True)
    
    return enhanced_df

def create_merchant_geographic_features(df):
    """Create merchant and geographic pattern features"""
    print("Creating merchant/geographic pattern features...")
    
    # Identify merchants (destinations starting with 'M') vs customers (starting with 'C')
    df['is_merchant_dest'] = df['nameDest'].str.startswith('M').astype(int)
    
    # Sort by user and step for time-based merchant analysis
    df_sorted = df.sort_values(['nameOrig', 'step']).copy()
    
    # Calculate merchant pattern features per user
    user_features = []
    
    for user in df_sorted['nameOrig'].unique():
        user_data = df_sorted[df_sorted['nameOrig'] == user].copy()
        
        # Unique merchants/destinations in last 7 steps
        user_data['unique_merchants_last_7d'] = user_data.rolling(window=7, min_periods=1)['nameDest'].apply(
            lambda x: len(set(x)), raw=False
        )
        
        # Most common merchant ratio (frequency of most used destination)
        def calc_most_common_ratio(dest_series):
            if len(dest_series) == 0:
                return 0
            return dest_series.value_counts().iloc[0] / len(dest_series) if len(dest_series) > 0 else 0
        
        user_data['most_common_merchant_ratio'] = user_data.rolling(window=7, min_periods=1)['nameDest'].apply(
            calc_most_common_ratio, raw=False
        )
        
        # New destination flag (1 if destination hasn't been used before)
        seen_destinations = set()
        new_dest_flags = []
        for dest in user_data['nameDest']:
            if dest in seen_destinations:
                new_dest_flags.append(0)
            else:
                new_dest_flags.append(1)
                seen_destinations.add(dest)
        user_data['new_destination_flag'] = new_dest_flags
        
        user_features.append(user_data)
    
    # Combine all user features
    enhanced_df = pd.concat(user_features, ignore_index=True)
    
    # Simulate geographic features since we don't have real location data
    print("Simulating geographic features for demonstration...")
    np.random.seed(42)  # For reproducible simulation
    
    # Simulate location coordinates based on destination patterns
    unique_destinations = enhanced_df['nameDest'].unique()
    dest_to_location = {}
    for dest in unique_destinations:
        # Merchants tend to have more stable locations
        if dest.startswith('M'):
            dest_to_location[dest] = (np.random.uniform(-90, 90), np.random.uniform(-180, 180))
        else:
            dest_to_location[dest] = (np.random.uniform(-90, 90), np.random.uniform(-180, 180))
    
    # Add simulated location features
    enhanced_df['dest_lat'] = enhanced_df['nameDest'].map(lambda x: dest_to_location[x][0])
    enhanced_df['dest_lon'] = enhanced_df['nameDest'].map(lambda x: dest_to_location[x][1])
    
    # Calculate distance from last location per user
    user_features_with_distance = []
    
    for user in enhanced_df['nameOrig'].unique():
        user_data = enhanced_df[enhanced_df['nameOrig'] == user].copy()
        
        distances = [0]  # First transaction has 0 distance
        for i in range(1, len(user_data)):
            lat1, lon1 = user_data.iloc[i-1]['dest_lat'], user_data.iloc[i-1]['dest_lon']
            lat2, lon2 = user_data.iloc[i]['dest_lat'], user_data.iloc[i]['dest_lon']
            
            # Simple distance calculation (not true geodesic, but sufficient for ML features)
            distance = np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2)
            distances.append(distance)
        
        user_data['distance_from_last_location'] = distances
        user_features_with_distance.append(user_data)
    
    # Combine all user features with distances
    final_df = pd.concat(user_features_with_distance, ignore_index=True)
    
    return final_df

def load_and_preprocess_data():
    """Load PaySim dataset and perform preprocessing"""
    print("Loading PaySim dataset...")
    
    # Load the dataset
    df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Basic info about fraud distribution
    fraud_count = df['isFraud'].sum()
    print(f"Total fraud cases: {fraud_count} ({fraud_count/len(df)*100:.2f}%)")
    
    # For demonstration, use a smaller sample of the data due to computational complexity
    print("Using a sample of 50,000 transactions for enhanced feature engineering...")
    df_sample = df.sample(n=50000, random_state=42).copy()
    
    # Feature engineering
    print("Performing enhanced feature engineering...")
    
    # Original features
    df_sample['balance_diff_orig'] = df_sample['oldbalanceOrg'] - df_sample['newbalanceOrig']
    df_sample['balance_diff_dest'] = df_sample['newbalanceDest'] - df_sample['oldbalanceDest']
    df_sample['amount_to_balance_ratio'] = df_sample['amount'] / (df_sample['oldbalanceOrg'] + 1)  # +1 to avoid division by zero
    
    # Handle zero balances (suspicious patterns)
    df_sample['orig_zero_balance'] = (df_sample['oldbalanceOrg'] == 0).astype(int)
    df_sample['dest_zero_balance'] = (df_sample['oldbalanceDest'] == 0).astype(int)
    
    # Transaction type encoding
    type_encoder = LabelEncoder()
    df_sample['type_encoded'] = type_encoder.fit_transform(df_sample['type'])
    
    # Create enhanced features
    print("Adding transaction frequency features...")
    df_enhanced = create_transaction_frequency_features(df_sample)
    
    print("Adding merchant/geographic features...")
    df_final = create_merchant_geographic_features(df_enhanced)
    
    # Select features for model (including new features)
    feature_columns = [
        'step', 'type_encoded', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'balance_diff_orig', 'balance_diff_dest',
        'amount_to_balance_ratio', 'orig_zero_balance', 'dest_zero_balance',
        # New transaction frequency features
        'txn_count_last_24h', 'avg_amount_last_7d', 'std_dev_amount_last_7d', 'ratio_current_to_avg',
        # New merchant/geographic features
        'is_merchant_dest', 'unique_merchants_last_7d', 'most_common_merchant_ratio',
        'new_destination_flag', 'dest_lat', 'dest_lon', 'distance_from_last_location'
    ]
    
    X = df_final[feature_columns].copy()
    y = df_final['isFraud'].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    print(f"\nEnhanced feature set created with {len(feature_columns)} features:")
    implemented_features = {
        'Transaction Frequency Features (Real)': [
            'txn_count_last_24h', 'avg_amount_last_7d', 
            'std_dev_amount_last_7d', 'ratio_current_to_avg'
        ],
        'Merchant Pattern Features (Real)': [
            'is_merchant_dest', 'unique_merchants_last_7d', 
            'most_common_merchant_ratio', 'new_destination_flag'
        ],
        'Geographic Features (Simulated)': [
            'dest_lat', 'dest_lon', 'distance_from_last_location'
        ]
    }
    
    for category, features in implemented_features.items():
        print(f"  {category}: {features}")
    
    return X, y, type_encoder, implemented_features

def train_model():
    """Train fraud detection model with enhanced features"""
    print("Starting enhanced model training...")
    
    # Load and preprocess data with new features
    X, y, type_encoder, implemented_features = load_and_preprocess_data()
    
    # Split data
    print("Splitting data into train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("Training Random Forest model with enhanced feature set...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,  # Increased depth for more complex features
        min_samples_split=30,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle imbalanced dataset
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    print("Evaluating enhanced model...")
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Analyze new features' importance
    print("\nImportance of New Features:")
    new_features = ['txn_count_last_24h', 'avg_amount_last_7d', 'std_dev_amount_last_7d', 'ratio_current_to_avg',
                   'is_merchant_dest', 'unique_merchants_last_7d', 'most_common_merchant_ratio',
                   'new_destination_flag', 'dest_lat', 'dest_lon', 'distance_from_last_location']
    
    new_feature_importance = feature_importance[feature_importance['feature'].isin(new_features)]
    print(new_feature_importance)
    
    # Save model and preprocessing objects
    print("\nSaving enhanced model and preprocessing objects...")
    
    with open('fraud_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(type_encoder, f)
    
    # Save feature columns for later use
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    # Save implemented features summary
    with open('implemented_features.pkl', 'wb') as f:
        pickle.dump(implemented_features, f)
    
    print("\nEnhanced model training completed successfully!")
    print("Files created:")
    print("- fraud_model.pkl (with enhanced features)")
    print("- scaler.pkl") 
    print("- encoder.pkl")
    print("- feature_columns.pkl")
    print("- implemented_features.pkl")
    
    # Print feature implementation summary
    print("\n" + "="*60)
    print("FEATURE IMPLEMENTATION SUMMARY")
    print("="*60)
    
    for category, features in implemented_features.items():
        print(f"\n{category}:")
        for feature in features:
            importance_val = feature_importance[feature_importance['feature'] == feature]['importance'].values
            if len(importance_val) > 0:
                print(f"  ✓ {feature} (importance: {importance_val[0]:.4f})")
            else:
                print(f"  ✓ {feature}")
    
    print("\nAll features successfully implemented and integrated into the model!")
    
    return rf_model, scaler, type_encoder, implemented_features

if __name__ == "__main__":
    model, scaler, encoder, features_summary = train_model()