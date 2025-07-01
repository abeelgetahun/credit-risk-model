"""
Feature Engineering Pipeline for Credit Risk Model
=================================================

This module implements a comprehensive feature engineering pipeline that transforms
raw transaction data into model-ready features using sklearn pipelines and specialized
libraries for credit risk modeling.

Key Features:
- Aggregate features (RFM metrics, transaction statistics)
- Temporal feature extraction
- Categorical encoding with WoE transformation
- Missing value handling
- Feature scaling and normalization
- Integration with xverse and woe packages for credit-specific transformations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Credit risk specific libraries
try:
    from xverse.transformer import WOE, MonotonicBinning
    XVERSE_AVAILABLE = True
except ImportError:
    XVERSE_AVAILABLE = False
    logging.warning("xverse package not available. Some transformations will be skipped.")

try:
    import woe as woe_pkg
    WOE_PKG_AVAILABLE = True
except ImportError:
    WOE_PKG_AVAILABLE = False
    logging.warning("woe package not available. Alternative WoE implementation will be used.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RFMCalculator(BaseEstimator, TransformerMixin):
    """
    Calculate Recency, Frequency, and Monetary (RFM) metrics for each customer.
    
    RFM Analysis is crucial for credit risk assessment as it provides insights into:
    - Recency: How recently a customer made a transaction
    - Frequency: How often a customer makes transactions
    - Monetary: How much money a customer spends
    """
    
    def __init__(self, snapshot_date: Optional[str] = None):
        """
        Initialize RFM Calculator.
        
        Parameters:
        -----------
        snapshot_date : str, optional
            Reference date for recency calculation. If None, uses max date in data.
        """
        self.snapshot_date = snapshot_date
        self.snapshot_date_parsed = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer - mainly to establish snapshot date."""
        if self.snapshot_date is None:
            self.snapshot_date_parsed = pd.to_datetime(X['TransactionStartTime']).max()
        else:
            self.snapshot_date_parsed = pd.to_datetime(self.snapshot_date)
        
        logger.info(f"RFM snapshot date set to: {self.snapshot_date_parsed}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM metrics for each customer.
        
        Returns:
        --------
        pd.DataFrame: DataFrame with RFM metrics aggregated by CustomerId
        """
        X_copy = X.copy()
        
        # Ensure datetime conversion
        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
        
        # Calculate RFM metrics
        rfm_data = X_copy.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (self.snapshot_date_parsed - x.max()).days,  # Recency
            'TransactionId': 'count',  # Frequency
            'Amount': ['sum', 'mean', 'std', 'count']  # Monetary metrics
        }).round(2)
        
        # Flatten column names
        rfm_data.columns = [
            'Recency', 'Frequency', 'TotalAmount', 'AvgAmount', 'StdAmount', 'TransactionCount'
        ]
        
        # Handle missing standard deviation (single transactions)
        rfm_data['StdAmount'].fillna(0, inplace=True)
        
        # Reset index to make CustomerId a column
        rfm_data.reset_index(inplace=True)
        
        logger.info(f"RFM calculation completed for {len(rfm_data)} customers")
        return rfm_data


class AggregateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract comprehensive aggregate features from transaction data.
    
    This transformer creates various statistical and behavioral features
    that are commonly used in credit risk modeling.
    """
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract aggregate features grouped by CustomerId."""
        X_copy = X.copy()
        
        # Ensure proper data types
        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
        X_copy['Amount'] = pd.to_numeric(X_copy['Amount'], errors='coerce')
        X_copy['Value'] = pd.to_numeric(X_copy['Value'], errors='coerce')
        
        logger.info("Extracting aggregate features...")
        
        # Transaction-level aggregations
        agg_features = X_copy.groupby('CustomerId').agg({
            # Amount-based features
            'Amount': ['sum', 'mean', 'std', 'min', 'max', 'count'],
            'Value': ['sum', 'mean', 'std', 'min', 'max'],
            
            # Transaction characteristics
            'TransactionId': 'count',
            'ChannelId': ['nunique', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]],
            'ProductCategory': ['nunique', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]],
            'CountryCode': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
            'CurrencyCode': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
            'ProviderId': 'nunique',
            'ProductId': 'nunique',
            'PricingStrategy': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
            'FraudResult': ['sum', 'mean'],  # Total fraud cases and fraud rate
        })
        
        # Flatten column names
        agg_features.columns = [
            'AmountSum', 'AmountMean', 'AmountStd', 'AmountMin', 'AmountMax', 'AmountCount',
            'ValueSum', 'ValueMean', 'ValueStd', 'ValueMin', 'ValueMax',
            'TransactionCount', 'UniqueChannels', 'PrimaryChannel',
            'UniqueProductCategories', 'PrimaryProductCategory',
            'PrimaryCountry', 'PrimaryCurrency', 'UniqueProviders', 'UniqueProducts',
            'PrimaryPricingStrategy', 'TotalFraudCases', 'FraudRate'
        ]
        
        # Handle missing values
        agg_features['AmountStd'].fillna(0, inplace=True)
        agg_features['ValueStd'].fillna(0, inplace=True)
        
        # Add derived features
        agg_features['AvgTransactionValue'] = agg_features['ValueSum'] / agg_features['TransactionCount']
        agg_features['AmountValueRatio'] = agg_features['AmountSum'] / (agg_features['ValueSum'] + 1e-8)
        agg_features['TransactionConsistency'] = 1 / (1 + agg_features['AmountStd'])
        
        # Reset index
        agg_features.reset_index(inplace=True)
        
        logger.info(f"Aggregate features extracted for {len(agg_features)} customers")
        return agg_features


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract temporal features from transaction timestamps.
    
    Temporal patterns are crucial in credit risk as they reveal:
    - Spending habits throughout the day/week/month
    - Seasonal patterns
    - Transaction timing consistency
    """
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from transaction data."""
        X_copy = X.copy()
        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
        
        logger.info("Extracting temporal features...")
        
        # Extract temporal components
        X_copy['TransactionHour'] = X_copy['TransactionStartTime'].dt.hour
        X_copy['TransactionDay'] = X_copy['TransactionStartTime'].dt.day
        X_copy['TransactionMonth'] = X_copy['TransactionStartTime'].dt.month
        X_copy['TransactionYear'] = X_copy['TransactionStartTime'].dt.year
        X_copy['TransactionDayOfWeek'] = X_copy['TransactionStartTime'].dt.dayofweek
        X_copy['TransactionDayOfYear'] = X_copy['TransactionStartTime'].dt.dayofyear
        X_copy['IsWeekend'] = X_copy['TransactionDayOfWeek'].isin([5, 6]).astype(int)
        
        # Aggregate temporal features by customer
        temporal_features = X_copy.groupby('CustomerId').agg({
            'TransactionHour': ['mean', 'std', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]],
            'TransactionDay': ['mean', 'std'],
            'TransactionMonth': ['nunique', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]],
            'TransactionDayOfWeek': ['mean', 'std'],
            'IsWeekend': 'mean',
            'TransactionStartTime': [
                lambda x: (x.max() - x.min()).days,  # Activity span in days
                'count'  # Transaction count
            ]
        })
        
        # Flatten column names
        temporal_features.columns = [
            'AvgTransactionHour', 'StdTransactionHour', 'PrimaryTransactionHour',
            'AvgTransactionDay', 'StdTransactionDay',
            'UniqueMonthsActive', 'PrimaryTransactionMonth',
            'AvgDayOfWeek', 'StdDayOfWeek', 'WeekendTransactionRatio',
            'ActivitySpanDays', 'TemporalTransactionCount'
        ]
        
        # Handle missing values
        temporal_features['StdTransactionHour'].fillna(0, inplace=True)
        temporal_features['StdTransactionDay'].fillna(0, inplace=True)
        temporal_features['StdDayOfWeek'].fillna(0, inplace=True)
        
        # Reset index
        temporal_features.reset_index(inplace=True)
        
        logger.info(f"Temporal features extracted for {len(temporal_features)} customers")
        return temporal_features


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Handle categorical variable encoding with multiple strategies.
    
    For credit risk modeling, we use:
    1. Weight of Evidence (WoE) encoding when target is available
    2. Label encoding for ordinal categories
    3. One-hot encoding for nominal categories with few levels
    """
    
    def __init__(self, target_column: Optional[str] = None, 
                 woe_columns: List[str] = None,
                 label_encode_columns: List[str] = None):
        """
        Initialize categorical encoder.
        
        Parameters:
        -----------
        target_column : str, optional
            Binary target column for WoE encoding
        woe_columns : list, optional
            Columns to apply WoE encoding
        label_encode_columns : list, optional
            Columns to apply label encoding
        """
        self.target_column = target_column
        self.woe_columns = woe_columns or []
        self.label_encode_columns = label_encode_columns or []
        self.label_encoders = {}
        self.woe_transformer = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit encoders to the data."""
        
        # Fit label encoders
        for col in self.label_encode_columns:
            if col in X.columns:
                le = LabelEncoder()
                # Handle missing values by treating them as a separate category
                X_col = X[col].fillna('Unknown').astype(str)
                le.fit(X_col)
                self.label_encoders[col] = le
        
        # Fit WoE transformer if target is available
        if self.target_column and self.target_column in X.columns and XVERSE_AVAILABLE:
            try:
                self.woe_transformer = WOE()
                woe_features = [col for col in self.woe_columns if col in X.columns]
                if woe_features:
                    X_woe = X[woe_features].fillna('Unknown')
                    self.woe_transformer.fit(X_woe, X[self.target_column])
                    logger.info(f"WoE transformer fitted for columns: {woe_features}")
            except Exception as e:
                logger.warning(f"WoE fitting failed: {e}. Falling back to label encoding.")
                self.woe_transformer = None
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical encoding transformations."""
        X_encoded = X.copy()
        
        # Apply label encoding
        for col, encoder in self.label_encoders.items():
            if col in X_encoded.columns:
                X_col = X_encoded[col].fillna('Unknown').astype(str)
                
                # Handle unseen categories
                unseen_mask = ~X_col.isin(encoder.classes_)
                if unseen_mask.any():
                    # Assign unseen categories to the most frequent class
                    most_frequent = encoder.classes_[0]  # First class (typically most frequent)
                    X_col.loc[unseen_mask] = most_frequent
                
                X_encoded[f'{col}_encoded'] = encoder.transform(X_col)
        
        # Apply WoE encoding
        if self.woe_transformer and self.target_column in X.columns:
            try:
                woe_features = [col for col in self.woe_columns if col in X.columns]
                if woe_features:
                    X_woe = X[woe_features].fillna('Unknown')
                    X_woe_transformed = self.woe_transformer.transform(X_woe)
                    
                    # Add WoE encoded features
                    for i, col in enumerate(woe_features):
                        X_encoded[f'{col}_woe'] = X_woe_transformed.iloc[:, i]
                        
            except Exception as e:
                logger.warning(f"WoE transformation failed: {e}")
        
        logger.info("Categorical encoding completed")
        return X_encoded


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Comprehensive missing value handling strategy.
    
    Uses different imputation strategies based on feature type and missingness pattern.
    """
    
    def __init__(self, numerical_strategy='median', categorical_strategy='most_frequent',
                 use_knn=False, knn_neighbors=5):
        """
        Initialize missing value handler.
        
        Parameters:
        -----------
        numerical_strategy : str
            Strategy for numerical features ('mean', 'median', 'constant')
        categorical_strategy : str
            Strategy for categorical features ('most_frequent', 'constant')
        use_knn : bool
            Whether to use KNN imputation for numerical features
        knn_neighbors : int
            Number of neighbors for KNN imputation
        """
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.use_knn = use_knn
        self.knn_neighbors = knn_neighbors
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.numerical_columns = []
        self.categorical_columns = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit imputers to the data."""
        
        # Identify column types
        self.numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Fit numerical imputer
        if self.numerical_columns:
            if self.use_knn:
                self.numerical_imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            else:
                self.numerical_imputer = SimpleImputer(strategy=self.numerical_strategy)
            
            self.numerical_imputer.fit(X[self.numerical_columns])
        
        # Fit categorical imputer
        if self.categorical_columns:
            self.categorical_imputer = SimpleImputer(strategy=self.categorical_strategy)
            self.categorical_imputer.fit(X[self.categorical_columns])
        
        logger.info(f"Missing value handlers fitted for {len(self.numerical_columns)} numerical and {len(self.categorical_columns)} categorical columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply missing value imputation."""
        X_imputed = X.copy()
        
        # Impute numerical columns
        if self.numerical_columns and self.numerical_imputer:
            X_imputed[self.numerical_columns] = self.numerical_imputer.transform(X[self.numerical_columns])
        
        # Impute categorical columns
        if self.categorical_columns and self.categorical_imputer:
            X_imputed[self.categorical_columns] = self.categorical_imputer.transform(X[self.categorical_columns])
        
        logger.info("Missing value imputation completed")
        return X_imputed


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Scale numerical features for model training.
    
    Standardization is preferred for most ML algorithms, especially for credit risk models
    where feature interpretability is important.
    """
    
    def __init__(self, scaling_method='standard', exclude_columns=None):
        """
        Initialize feature scaler.
        
        Parameters:
        -----------
        scaling_method : str
            Scaling method ('standard', 'minmax', 'robust')
        exclude_columns : list
            Columns to exclude from scaling
        """
        self.scaling_method = scaling_method
        self.exclude_columns = exclude_columns or []
        self.scaler = None
        self.columns_to_scale = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit scaler to numerical columns."""
        
        # Identify columns to scale
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.columns_to_scale = [col for col in numerical_columns if col not in self.exclude_columns]
        
        if self.columns_to_scale:
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler()
            elif self.scaling_method == 'robust':
                from sklearn.preprocessing import RobustScaler
                self.scaler = RobustScaler()
            
            self.scaler.fit(X[self.columns_to_scale])
            logger.info(f"Scaler fitted for {len(self.columns_to_scale)} columns using {self.scaling_method} method")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling transformation."""
        X_scaled = X.copy()
        
        if self.columns_to_scale and self.scaler:
            X_scaled[self.columns_to_scale] = self.scaler.transform(X[self.columns_to_scale])
        
        logger.info("Feature scaling completed")
        return X_scaled


def create_feature_engineering_pipeline(snapshot_date: Optional[str] = None,
                                       target_column: Optional[str] = None) -> Pipeline:
    """
    Create a comprehensive feature engineering pipeline.
    
    Parameters:
    -----------
    snapshot_date : str, optional
        Reference date for RFM calculation
    target_column : str, optional
        Binary target column for supervised transformations
    
    Returns:
    --------
    Pipeline: Sklearn pipeline with all transformation steps
    """
    
    # Define categorical columns for encoding
    woe_columns = ['ChannelId', 'ProductCategory', 'CountryCode', 'CurrencyCode', 'PricingStrategy']
    label_encode_columns = ['ProviderId', 'ProductId']
    
    # Create pipeline steps
    pipeline_steps = [
        ('missing_handler', MissingValueHandler(use_knn=False)),
        ('categorical_encoder', CategoricalEncoder(
            target_column=target_column,
            woe_columns=woe_columns,
            label_encode_columns=label_encode_columns
        )),
        ('scaler', FeatureScaler(exclude_columns=['CustomerId', target_column] if target_column else ['CustomerId']))
    ]
    
    pipeline = Pipeline(pipeline_steps)
    logger.info("Feature engineering pipeline created")
    
    return pipeline


def process_transaction_data(df: pd.DataFrame, 
                           snapshot_date: Optional[str] = None,
                           target_column: Optional[str] = None) -> pd.DataFrame:
    """
    Complete feature engineering process for transaction data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw transaction data
    snapshot_date : str, optional
        Reference date for RFM calculation
    target_column : str, optional
        Binary target column
    
    Returns:
    --------
    pd.DataFrame: Processed dataset with engineered features
    """
    
    logger.info("Starting feature engineering process...")
    
    # Step 1: Calculate RFM metrics
    rfm_calculator = RFMCalculator(snapshot_date=snapshot_date)
    rfm_data = rfm_calculator.fit_transform(df)
    
    # Step 2: Extract aggregate features
    agg_extractor = AggregateFeatureExtractor()
    agg_features = agg_extractor.fit_transform(df)
    
    # Step 3: Extract temporal features
    temporal_extractor = TemporalFeatureExtractor()
    temporal_features = temporal_extractor.fit_transform(df)
    
    # Step 4: Merge all features
    logger.info("Merging feature sets...")
    processed_data = rfm_data.merge(agg_features, on='CustomerId', how='inner')
    processed_data = processed_data.merge(temporal_features, on='CustomerId', how='inner')
    
    # Step 5: Apply main feature engineering pipeline
    pipeline = create_feature_engineering_pipeline(
        snapshot_date=snapshot_date,
        target_column=target_column
    )
    
    # Fit and transform
    processed_data = pipeline.fit_transform(processed_data)
    
    # Convert back to DataFrame if needed
    if not isinstance(processed_data, pd.DataFrame):
        # Get feature names (this is a simplified approach)
        feature_names = [f'feature_{i}' for i in range(processed_data.shape[1])]
        processed_data = pd.DataFrame(processed_data, columns=feature_names)
    
    logger.info(f"Feature engineering completed. Final dataset shape: {processed_data.shape}")
    
    return processed_data


if __name__ == "__main__":
    # Example usage
    logger.info("Feature engineering module loaded successfully")
    
    # Test with sample data (this would be replaced with actual data loading)
    # sample_data = pd.read_csv('data/raw/sample_data.csv')
    # processed_data = process_transaction_data(sample_data)
    # print(f"Processed data shape: {processed_data.shape}")