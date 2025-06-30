"""
Proxy Target Variable Engineering for Credit Risk Model
=====================================================

This module implements RFM (Recency, Frequency, Monetary) analysis to create
a proxy target variable for credit risk assessment. Since we don't have direct
default labels, we use customer engagement patterns to identify high-risk segments.

The approach:
1. Calculate RFM metrics for each customer
2. Use K-Means clustering to segment customers
3. Identify the least engaged cluster as high-risk
4. Create binary target variable based on cluster assignment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RFMTargetEngineer:
    """
    Engineer proxy target variable using RFM analysis and clustering.
    
    This class implements a systematic approach to identify high-risk customers
    based on their transaction behavior patterns.
    """
    
    def __init__(self, n_clusters: int = 3, random_state: int = 42, 
                 snapshot_date: Optional[str] = None):
        """
        Initialize RFM Target Engineer.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters for K-Means (default: 3)
        random_state : int
            Random state for reproducibility
        snapshot_date : str, optional
            Reference date for recency calculation
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.snapshot_date = snapshot_date
        self.snapshot_date_parsed = None
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.rfm_data = None
        self.cluster_profiles = None
        self.high_risk_cluster = None
        
    def calculate_rfm_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics for each customer.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction data with required columns: CustomerId, TransactionStartTime, Amount
            
        Returns:
        --------
        pd.DataFrame: RFM metrics for each customer
        """
        logger.info("Calculating RFM metrics...")
        
        # Ensure proper data types
        df = df.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        # Set snapshot date
        if self.snapshot_date is None:
            self.snapshot_date_parsed = df['TransactionStartTime'].max()
        else:
            self.snapshot_date_parsed = pd.to_datetime(self.snapshot_date)
        
        logger.info(f"Using snapshot date: {self.snapshot_date_parsed}")
        
        # Calculate RFM metrics
        rfm_metrics = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (self.snapshot_date_parsed - x.max()).days,  # Recency
            'TransactionId': 'count',  # Frequency  
            'Amount': 'sum'  # Monetary
        }).round(2)
        
        # Rename columns
        rfm_metrics.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Handle edge cases
        rfm_metrics['Recency'] = rfm_metrics['Recency'].clip(lower=0)  # Ensure non-negative recency
        rfm_metrics['Monetary'] = rfm_metrics['Monetary'].abs()  # Use absolute values for monetary
        
        # Add additional metrics for better segmentation
        rfm_metrics['AvgTransactionAmount'] = rfm_metrics['Monetary'] / rfm_metrics['Frequency']
        
        # Calculate customer value score (simple heuristic)
        rfm_metrics['CustomerValue'] = (
            rfm_metrics['Frequency'] * rfm_metrics['Monetary'] / 
            (rfm_metrics['Recency'] + 1)  # Add 1 to avoid division by zero
        )
        
        # Reset index to make CustomerId a column
        rfm_metrics.reset_index(inplace=True)
        
        logger.info(f"RFM metrics calculated for {len(rfm_metrics)} customers")
        logger.info(f"RFM metrics summary:\n{rfm_metrics.describe()}")
        
        self.rfm_data = rfm_metrics
        return rfm_metrics
    
    def perform_clustering(self, rfm_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Perform K-Means clustering on RFM data.
        
        Parameters:
        -----------
        rfm_data : pd.DataFrame, optional
            RFM data to cluster. If None, uses self.rfm_data
            
        Returns:
        --------
        pd.DataFrame: RFM data with cluster assignments
        """
        if rfm_data is None:
            rfm_data = self.rfm_data
            
        if rfm_data is None:
            raise ValueError("No RFM data available. Run calculate_rfm_metrics first.")
        
        logger.info("Performing K-Means clustering on RFM data...")
        
        # Select features for clustering (exclude CustomerId and derived metrics)
        clustering_features = ['Recency', 'Frequency', 'Monetary']
        X_cluster = rfm_data[clustering_features].copy()
        
        # Handle missing values (shouldn't happen with proper RFM calculation)
        X_cluster = X_cluster.fillna(X_cluster.median())
        
        # Scale features for clustering
        X_scaled = self.scaler.fit_transform(X_cluster)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to data
        rfm_clustered = rfm_data.copy()
        rfm_clustered['Cluster'] = cluster_labels
        
        # Calculate silhouette score for cluster quality assessment
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        logger.info(f"Clustering completed. Silhouette score: {silhouette_avg:.3f}")
        
        # Create cluster profiles for analysis
        self.cluster_profiles = self._create_cluster_profiles(rfm_clustered)
        
        self.rfm_data = rfm_clustered
        return rfm_clustered
    
    def _create_cluster_profiles(self, rfm_clustered: pd.DataFrame) -> pd.DataFrame:
        """
        Create detailed profiles for each cluster.
        
        Parameters:
        -----------
        rfm_clustered : pd.DataFrame
            RFM data with cluster assignments
            
        Returns:
        --------
        pd.DataFrame: Cluster profiles with statistical summaries
        """
        logger.info("Creating cluster profiles...")
        
        # Calculate cluster statistics
        cluster_profiles = rfm_clustered.groupby('Cluster').agg({
            'Recency': ['mean', 'std', 'median'],
            'Frequency': ['mean', 'std', 'median'],
            'Monetary': ['mean', 'std', 'median'],
            'AvgTransactionAmount': ['mean', 'std', 'median'],
            'CustomerValue': ['mean', 'std', 'median'],
            'CustomerId': 'count'  # Cluster size
        }).round(2)
        
        # Flatten column names
        cluster_profiles.columns = [
            'Recency_Mean', 'Recency_Std', 'Recency_Median',
            'Frequency_Mean', 'Frequency_Std', 'Frequency_Median', 
            'Monetary_Mean', 'Monetary_Std', 'Monetary_Median',
            'AvgAmount_Mean', 'AvgAmount_Std', 'AvgAmount_Median',
            'CustomerValue_Mean', 'CustomerValue_Std', 'CustomerValue_Median',
            'ClusterSize'
        ]
        
        # Add cluster percentages
        total_customers = len(rfm_clustered)
        cluster_profiles['ClusterPercentage'] = (
            cluster_profiles['ClusterSize'] / total_customers * 100
        ).round(2)
        
        # Add risk interpretation
        cluster_profiles['RiskInterpretation'] = self._interpret_clusters(cluster_profiles)
        
        logger.info("Cluster profiles created:")
        print(cluster_profiles[['ClusterSize', 'ClusterPercentage', 'Recency_Mean', 
                               'Frequency_Mean', 'Monetary_Mean', 'RiskInterpretation']])
        
        return cluster_profiles
    
    def _interpret_clusters(self, cluster_profiles: pd.DataFrame) -> pd.Series:
        """
        Interpret clusters based on RFM characteristics.
        
        Parameters:
        -----------
        cluster_profiles : pd.DataFrame
            Cluster statistical profiles
            
        Returns:
        --------
        pd.Series: Risk interpretations for each cluster
        """
        interpretations = []
        
        for idx in cluster_profiles.index:
            recency = cluster_profiles.loc[idx, 'Recency_Mean']
            frequency = cluster_profiles.loc[idx, 'Frequency_Mean'] 
            monetary = cluster_profiles.loc[idx, 'Monetary_Mean']
            
            # Risk assessment logic
            if recency > cluster_profiles['Recency_Mean'].median() and \
               frequency < cluster_profiles['Frequency_Mean'].median() and \
               monetary < cluster_profiles['Monetary_Mean'].median():
                interpretation = "High Risk - Disengaged"
            elif recency < cluster_profiles['Recency_Mean'].median() and \
                 frequency > cluster_profiles['Frequency_Mean'].median() and \
                 monetary > cluster_profiles['Monetary_Mean'].median():
                interpretation = "Low Risk - Highly Engaged"
            else:
                interpretation = "Medium Risk - Moderately Engaged"
                
            interpretations.append(interpretation)
        
        return pd.Series(interpretations, index=cluster_profiles.index)
    
    def identify_high_risk_cluster(self, rfm_clustered: Optional[pd.DataFrame] = None) -> int:
        """
        Identify the cluster representing high-risk customers.
        
        High-risk customers are typically characterized by:
        - High recency (haven't transacted recently)
        - Low frequency (few transactions)
        - Low monetary value (small transaction amounts)
        
        Parameters:
        -----------
        rfm_clustered : pd.DataFrame, optional
            RFM data with clusters. If None, uses self.rfm_data
            
        Returns:
        --------
        int: Cluster number identified as high-risk
        """
        if rfm_clustered is None:
            rfm_clustered = self.rfm_data
            
        if self.cluster_profiles is None:
            raise ValueError("Cluster profiles not available. Run perform_clustering first.")
        
        logger.info("Identifying high-risk cluster...")
        
        # Calculate risk scores for each cluster
        # Higher recency = worse, Lower frequency = worse, Lower monetary = worse
        cluster_risk_scores = {}
        
        for cluster in self.cluster_profiles.index:
            # Normalize metrics (higher score = higher risk)
            recency_score = self.cluster_profiles.loc[cluster, 'Recency_Mean']
            frequency_score = 1 / (self.cluster_profiles.loc[cluster, 'Frequency_Mean'] + 1)
            monetary_score = 1 / (self.cluster_profiles.loc[cluster, 'Monetary_Mean'] + 1)
            
            # Composite risk score
            risk_score = recency_score + frequency_score + monetary_score
            cluster_risk_scores[cluster] = risk_score
        
        # Identify cluster with highest risk score
        self.high_risk_cluster = max(cluster_risk_scores, key=cluster_risk_scores.get)
        
        logger.info(f"High-risk cluster identified: Cluster {self.high_risk_cluster}")
        logger.info(f"Cluster risk scores: {cluster_risk_scores}")
        
        # Print high-risk cluster characteristics
        high_risk_profile = self.cluster_profiles.loc[self.high_risk_cluster]
        logger.info(f"High-risk cluster profile:\n"
                   f"  - Size: {high_risk_profile['ClusterSize']} customers "
                   f"({high_risk_profile['ClusterPercentage']:.1f}%)\n"
                   f"  - Avg Recency: {high_risk_profile['Recency_Mean']:.1f} days\n"
                   f"  - Avg Frequency: {high_risk_profile['Frequency_Mean']:.1f} transactions\n"
                   f"  - Avg Monetary: ${high_risk_profile['Monetary_Mean']:.2f}")
        
        return self.high_risk_cluster
    
    def create_target_variable(self, rfm_clustered: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create binary target variable based on cluster assignment.
        
        Parameters:
        -----------
        rfm_clustered : pd.DataFrame, optional
            RFM data with clusters. If None, uses self.rfm_data
            
        Returns:
        --------
        pd.DataFrame: Data with binary 'is_high_risk' target variable
        """
        if rfm_clustered is None:
            rfm_clustered = self.rfm_data.copy()
            
        if self.high_risk_cluster is None:
            self.identify_high_risk_cluster(rfm_clustered)
        
        logger.info("Creating binary target variable...")
        
        # Create binary target variable
        rfm_with_target = rfm_clustered.copy()
        rfm_with_target['is_high_risk'] = (
            rfm_with_target['Cluster'] == self.high_risk_cluster
        ).astype(int)
        
        # Log target distribution
        target_distribution = rfm_with_target['is_high_risk'].value_counts()
        target_percentages = rfm_with_target['is_high_risk'].value_counts(normalize=True) * 100
        
        logger.info(f"Target variable distribution:")
        logger.info(f"  - Low Risk (0): {target_distribution[0]} customers ({target_percentages[0]:.1f}%)")
        logger.info(f"  - High Risk (1): {target_distribution[1]} customers ({target_percentages[1]:.1f}%)")
        
        return rfm_with_target
    
    def plot_cluster_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualizations for cluster analysis.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, displays the plot.
        """
        if self.rfm_data is None:
            raise ValueError("No data available for plotting. Run the full pipeline first.")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RFM Cluster Analysis for Credit Risk Assessment', fontsize=16, fontweight='bold')
        
        # Color palette for clusters
        colors = sns.color_palette("Set2", self.n_clusters)
        
        # 1. RFM Distribution by Cluster
        for i, metric in enumerate(['Recency', 'Frequency', 'Monetary']):
            ax = axes[0, i]
            for cluster in sorted(self.rfm_data['Cluster'].unique()):
                cluster_data = self.rfm_data[self.rfm_data['Cluster'] == cluster]
                ax.hist(cluster_data[metric], alpha=0.7, label=f'Cluster {cluster}', 
                       color=colors[cluster], bins=20)
            
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{metric} Distribution by Cluster')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Cluster Scatter Plots
        scatter_combinations = [('Recency', 'Frequency'), ('Frequency', 'Monetary'), ('Recency', 'Monetary')]
        
        for i, (x_metric, y_metric) in enumerate(scatter_combinations):
            ax = axes[1, i]
            
            for cluster in sorted(self.rfm_data['Cluster'].unique()):
                cluster_data = self.rfm_data[self.rfm_data['Cluster'] == cluster]
                
                # Highlight high-risk cluster
                if cluster == self.high_risk_cluster:
                    ax.scatter(cluster_data[x_metric], cluster_data[y_metric], 
                             c=colors[cluster], label=f'Cluster {cluster} (High Risk)', 
                             alpha=0.7, s=50, edgecolors='red', linewidth=2)
                else:
                    ax.scatter(cluster_data[x_metric], cluster_data[y_metric], 
                             c=colors[cluster], label=f'Cluster {cluster}', alpha=0.7, s=30)
            
            ax.set_xlabel(x_metric)
            ax.set_ylabel(y_metric)
            ax.set_title(f'{x_metric} vs {y_metric}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster analysis plot saved to: {save_path}")
        else:
            plt.show()
    
    def get_cluster_summary(self) -> Dict:
        """
        Get comprehensive summary of the clustering analysis.
        
        Returns:
        --------
        Dict: Summary statistics and interpretations
        """
        if self.cluster_profiles is None:
            raise ValueError("Clustering not performed. Run the full pipeline first.")
        
        summary = {
            'total_customers': len(self.rfm_data),
            'n_clusters': self.n_clusters,
            'high_risk_cluster': self.high_risk_cluster,
            'cluster_profiles': self.cluster_profiles.to_dict(),
            'target_distribution': self.rfm_data['is_high_risk'].value_counts().to_dict() if 'is_high_risk' in self.rfm_data.columns else None,
            'silhouette_score': silhouette_score(
                self.scaler.transform(self.rfm_data[['Recency', 'Frequency', 'Monetary']]),
                self.rfm_data['Cluster']
            ) if 'Cluster' in self.rfm_data.columns else None
        }
        
        return summary
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete pipeline to create proxy target variable from transaction data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw transaction data
            
        Returns:
        --------
        pd.DataFrame: RFM data with cluster assignments and binary target variable
        """
        logger.info("Starting RFM target engineering pipeline...")
        
        # Step 1: Calculate RFM metrics
        rfm_data = self.calculate_rfm_metrics(df)
        
        # Step 2: Perform clustering
        rfm_clustered = self.perform_clustering(rfm_data)
        
        # Step 3: Identify high-risk cluster
        self.identify_high_risk_cluster(rfm_clustered)
        
        # Step 4: Create target variable
        rfm_with_target = self.create_target_variable(rfm_clustered)
        
        logger.info("RFM target engineering pipeline completed successfully!")
        
        return rfm_with_target


def optimize_cluster_number(df: pd.DataFrame, max_clusters: int = 10, 
                          random_state: int = 42) -> Tuple[int, Dict]:
    """
    Find optimal number of clusters using elbow method and silhouette analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction data
    max_clusters : int
        Maximum number of clusters to test
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    Tuple[int, Dict]: Optimal number of clusters and evaluation metrics
    """
    logger.info("Optimizing number of clusters...")
    
    # Calculate RFM metrics
    temp_engineer = RFMTargetEngineer(random_state=random_state)
    rfm_data = temp_engineer.calculate_rfm_metrics(df)
    
    # Prepare data for clustering
    numeric_cols = ['Recency', 'Frequency', 'Monetary']
    X = rfm_data[numeric_cols].fillna(rfm_data[numeric_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different numbers of clusters
    cluster_range = range(2, max_clusters + 1)
    inertias = []
    silhouette_scores = []
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
    
    # Find optimal number using silhouette score
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    
    evaluation_metrics = {
        'cluster_range': list(cluster_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'optimal_clusters': optimal_clusters,
        'max_silhouette_score': max(silhouette_scores)
    }
    
    logger.info(f"Optimal number of clusters: {optimal_clusters} "
               f"(Silhouette Score: {max(silhouette_scores):.3f})")
    
    return optimal_clusters, evaluation_metrics


if __name__ == "__main__":
    # Example usage
    logger.info("RFM Target Engineering module loaded successfully")
    
    # Test with sample data (this would be replaced with actual data loading)
    # sample_data = pd.read_csv('data/raw/sample_data.csv') 
    # 
    # # Find optimal number of clusters
    # optimal_k, metrics = optimize_cluster_number(sample_data)
    # 
    # # Create target variable
    # target_engineer = RFMTargetEngineer(n_clusters=optimal_k)
    # rfm_with_target = target_engineer.fit_transform(sample_data)
    # 
    # # Generate visualizations
    # target_engineer.plot_cluster_analysis(save_path='cluster_analysis.png')
    # 
    # # Get summary
    # summary = target_engineer.get_cluster_summary()
    # print("Clustering Summary:", summary)