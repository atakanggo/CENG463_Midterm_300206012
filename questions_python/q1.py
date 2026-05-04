"""
Main Script for Question 1: Comprehensive ML Pipeline
Orchestrates all phases: EDA, Feature Engineering, Model Evaluation, and Diagnostics
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import RepeatedKFold, train_test_split

# Import custom modules
from california_housing_eda import detect_outliers_iqr
from feature_engineering import engineer_features, get_feature_importance
from model_evaluation import evaluate_models, get_default_models, perform_statistical_test
from residual_diagnostics import residual_diagnostics, train_and_evaluate_huber

# Set output directory - Q1 outputs go to outputs/q1/
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'q1')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    """
    Main pipeline orchestrating all ML phases for California Housing dataset.
    """
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MACHINE LEARNING PIPELINE - QUESTION 1")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    # ========================================================================
    # PHASE 1: LOAD DATA
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: LOADING DATA")
    print("="*80)
    
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: X={X.shape}, y={y.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target: Price")
    
    # ========================================================================
    # PHASE 2: EXPLORATORY DATA ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    print("\nNote: EDA visualizations are being generated...")
    print("Check 'outputs/q1/' folder for:")
    print("  - histograms_all_features.png")
    print("  - correlation_heatmap.png")
    print("  - pairplot_top3_features.png")
    print("  - boxplots_outliers.png")
    print("  - outliers_summary.csv")
    
    # Run EDA script
    print("\nRunning EDA analysis (this may take a minute)...")
    exec(open(os.path.join(os.path.dirname(__file__), 'california_housing_eda.py')).read())
    
    # ========================================================================
    # PHASE 3: FEATURE ENGINEERING
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 3: FEATURE ENGINEERING")
    print("="*80)
    
    # Convert Series to proper format if needed
    if isinstance(y, pd.Series):
        y_series = y
    else:
        y_series = pd.Series(y, name='Price')
    
    X_engineered, y_engineered = engineer_features(X, y_series)
    
    # Convert y_engineered to pandas Series if it's a numpy array
    if isinstance(y_engineered, np.ndarray):
        y_engineered = pd.Series(y_engineered, name='Price')
    
    print(f"\n✓ Feature engineering completed!")
    print(f"Original features: {X.shape[1]}")
    print(f"Engineered features (after poly + RFE): {X_engineered.shape[1]}")
    
    # ========================================================================
    # PHASE 4: MODEL EVALUATION
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 4: MODEL EVALUATION (CROSS-VALIDATION)")
    print("="*80)
    
    # Create CV strategy
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    # Get models
    models = get_default_models(random_state=42)
    
    # Evaluate models
    results_df, all_metrics, cv_obj = evaluate_models(X_engineered, y_engineered, 
                                                       models_dict=models, cv=cv)
    
    # Perform statistical test
    ttest_result = perform_statistical_test(
        all_metrics,
        model1='XGBRegressor (Tuned)',
        model2='LinearRegression',
        alpha=0.05
    )
    
    # ========================================================================
    # PHASE 5: RESIDUAL DIAGNOSTICS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 5: RESIDUAL DIAGNOSTICS")
    print("="*80)
    
    # Split data for final model training and diagnostics
    # Convert y_engineered to numpy array for train_test_split
    y_engineered_array = y_engineered.values if isinstance(y_engineered, pd.Series) else y_engineered
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y_engineered_array, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining Huber Regressor and Linear Regression for diagnostic analysis...")
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train and compare models
    results = train_and_evaluate_huber(X_train, X_test, y_train, y_test, epsilon=1.35)
    
    # Generate residual diagnostics
    fig_lr, axes_lr, residuals_lr = residual_diagnostics(
        y_test, results['y_pred_lr'], 
        model_name="LinearRegression",
        figsize=(14, 5)
    )
    output_path = os.path.join(OUTPUT_DIR, 'residual_diagnostics_linear_regression.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Residual diagnostics saved: {output_path}")
    
    fig_huber, axes_huber, residuals_huber = residual_diagnostics(
        y_test, results['y_pred_huber'], 
        model_name="HuberRegressor",
        figsize=(14, 5)
    )
    output_path = os.path.join(OUTPUT_DIR, 'residual_diagnostics_huber_regressor.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Residual diagnostics saved: {output_path}")
    
    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    
    print(f"\n✓ PHASE 1 - Data Loading: COMPLETED")
    print(f"  Dataset: California Housing")
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    
    print(f"\n✓ PHASE 2 - Exploratory Data Analysis: COMPLETED")
    print(f"  Generated 5 visualization files and 1 CSV summary")
    print(f"  Location: outputs/q1/")
    
    print(f"\n✓ PHASE 3 - Feature Engineering: COMPLETED")
    print(f"  Original Features: {X.shape[1]}")
    print(f"  Polynomial Features Created: 44")
    print(f"  Features Selected (RFE): {X_engineered.shape[1]}")
    print(f"  Target Transformation: {'YES (log1p)' if np.mean(y_engineered) != np.mean(y.values) else 'NO'}")
    
    print(f"\n✓ PHASE 4 - Model Evaluation: COMPLETED")
    print(f"  Cross-Validation: RepeatedKFold (5 splits, 3 repeats)")
    print(f"  Total Folds: 15")
    print(f"  Models Evaluated: 5")
    print(f"  Statistical Test: Paired t-test (XGBRegressor vs LinearRegression)")
    print(f"  Result: {'SIGNIFICANT' if ttest_result['is_significant'] else 'NOT SIGNIFICANT'} (p={ttest_result['p_value']:.6f})")
    
    print(f"\n✓ PHASE 5 - Residual Diagnostics: COMPLETED")
    print(f"  Models Analyzed:")
    print(f"    • LinearRegression (RMSE: {results['rmse_lr']:.6f}, R²: {results['r2_lr']:.6f})")
    print(f"    • HuberRegressor (RMSE: {results['rmse_huber']:.6f}, R²: {results['r2_huber']:.6f})")
    print(f"  Diagnostics Generated: 2 residual diagnostic plots")
    
    print("\n" + "="*80)
    print("ALL OUTPUTS SAVED TO: ../outputs/q1/")
    print("="*80)
    
    print("\nGenerated Files:")
    try:
        output_files = os.listdir(OUTPUT_DIR)
        for i, file in enumerate(sorted(output_files), 1):
            file_path = os.path.join(OUTPUT_DIR, file)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                print(f"  {i}. {file} ({file_size:,} bytes)")
    except Exception as e:
        print(f"  Could not list files: {e}")
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()