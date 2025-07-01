"""
Integrated Data Processing Pipeline
==================================

This module combines feature engineering and target variable creation
into a unified pipeline for credit risk modeling.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
from pathlib import Path

from data_processing import process_transaction_data
from target_engineering import RFMTargetEngineer, optimize_cluster_number

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_complete_pipeline(data_path: str, 
                         output_path: Optional[str] = None,
                         snapshot_date: Optional[str] = None,
                         optimize_clusters: bool = True,
                         n_clusters: int = 3) -> Tuple[pd.DataFrame, dict]:
    """
    Run the complete data processing pipeline.
    
    Parameters:
    -----------
    data_path : str
        Path to raw transaction data
    output_path : str, optional
        Path to save processed data
    snapshot_date : str, optional
        Reference date for RFM calculations
    optimize_clusters : bool
        Whether to optimize number of clusters
    n_clusters : int
        Number of clusters if not optimizing
        
    Returns:
    --------
    Tuple[pd.DataFrame, dict]: Processed data and pipeline summary
    """
    
    logger.info("Starting complete data processing pipeline...")
    
    # Load data
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} transactions for {df['CustomerId'].nunique()} customers")
    
    # Step 1: Create proxy target variable using RFM analysis
    logger.info("Step 1: Creating proxy target variable...")
    
    if optimize_clusters:
        optimal_k, cluster_metrics = optimize_cluster_number(df)
        n_clusters = optimal_k
    else:
        cluster_metrics = None
    
    # Create target variable
    target_engineer = RFMTargetEngineer(n_clusters=n_clusters, snapshot_date=snapshot_date)
    rfm_with_target = target_engineer.fit_transform(df)
    
    # Step 2: Merge target back to transaction data
    logger.info("Step 2: Merging target variable with transaction data...")
    df_with_target = df.merge(
        rfm_with_target[['CustomerId', 'is_high_risk']], 
        on='CustomerId', 
        how='left'
    )
    
    # Step 3: Feature engineering
    logger.info("Step 3: Performing feature engineering...")
    processed_data = process_transaction_data(
        df_with_target, 
        snapshot_date=snapshot_date,
        target_column='is_high_risk'
    )
    
    # Step 4: Final data preparation
    logger.info("Step 4: Final data preparation...")
    
    # Ensure we have the target variable in the final dataset
    if 'is_high_risk' not in processed_data.columns:
        processed_data = processed_data.merge(
            rfm_with_target[['CustomerId', 'is_high_risk']], 
            on='CustomerId', 
            how='left'
        )
    
    # Create pipeline summary
    pipeline_summary = {
        'input_shape': df.shape,
        'output_shape': processed_data.shape,
        'unique_customers': df['CustomerId'].nunique(),
        'target_distribution': processed_data['is_high_risk'].value_counts().to_dict(),
        'n_clusters_used': n_clusters,
        'cluster_optimization_metrics': cluster_metrics,
        'rfm_summary': target_engineer.get_cluster_summary()
    }
    
    # Save processed data if path provided
    if output_path:
        processed_data.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to: {output_path}")
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"Final dataset shape: {processed_data.shape}")
    logger.info(f"Target distribution: {pipeline_summary['target_distribution']}")
    
    return processed_data, pipeline_summary


if __name__ == "__main__":
    # Example usage
    data_path = "data/raw/data.csv"
    output_path = "data/processed/processed_data.csv"
    
    processed_data, summary = run_complete_pipeline(
        data_path=data_path,
        output_path=output_path,
        optimize_clusters=True
    )
    
    print("Pipeline Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")