#!/usr/bin/env python3
"""
Test script to verify that enhanced features are working correctly in the fraud detection app
"""

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path to import from app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_enhanced_features_for_prediction

def test_enhanced_features():
    """Test the enhanced feature creation"""
    print("Testing Enhanced Features Implementation")
    print("=" * 50)
    
    # Create sample test data
    test_data = pd.DataFrame({
        'step': [1, 2, 3, 4, 5],
        'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'PAYMENT'],
        'amount': [1000.0, 500.0, 2000.0, 100.0, 750.0],
        'nameOrig': ['C123456789', 'C123456789', 'C987654321', 'C123456789', 'C555666777'],
        'oldbalanceOrg': [5000.0, 4000.0, 10000.0, 3900.0, 2000.0],
        'newbalanceOrig': [4000.0, 3500.0, 8000.0, 3800.0, 1250.0],
        'nameDest': ['M111111111', 'C222222222', 'C333333333', 'M111111111', 'M444444444'],
        'oldbalanceDest': [0.0, 1000.0, 5000.0, 0.0, 0.0],
        'newbalanceDest': [0.0, 1500.0, 7000.0, 0.0, 0.0]
    })
    
    print("Original test data:")
    print(test_data)
    print(f"Shape: {test_data.shape}")
    
    # Apply enhanced feature engineering
    enhanced_data = create_enhanced_features_for_prediction(test_data)
    
    print("\nEnhanced features created:")
    enhanced_features = [
        'txn_count', 'avg_amount', 'std_amount', 'active_days', 'ratio_current_to_avg',
        'is_merchant_dest', 'unique_destinations', 'destination_diversity_ratio',
        'new_destination_flag', 'dest_lat', 'dest_lon', 'distance_from_common'
    ]
    
    for feature in enhanced_features:
        if feature in enhanced_data.columns:
            print(f"✓ {feature}: {enhanced_data[feature].tolist()}")
        else:
            print(f"✗ {feature}: NOT FOUND")
    
    # Verify feature mapping
    feature_mapping = {
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
    
    print(f"\nFeature name mapping verification:")
    for tech_name, display_name in feature_mapping.items():
        if tech_name in enhanced_data.columns:
            print(f"✓ {tech_name} → {display_name}")
        else:
            print(f"✗ {tech_name} → {display_name} (MISSING)")
    
    print(f"\nFinal enhanced dataset shape: {enhanced_data.shape}")
    print("Enhanced features test completed successfully!")
    
    return enhanced_data

if __name__ == "__main__":
    test_enhanced_features()