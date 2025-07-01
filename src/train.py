import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditRiskModelTrainer:
    def __init__(self, data_path, test_size=0.2, random_state=42):
        """
        Initialize the Credit Risk Model Trainer
        
        Args:
            data_path (str): Path to the processed data
            test_size (float): Proportion of data for testing
            random_state (int): Random state for reproducibility
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
        # Setup MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("credit-risk-modeling")
        
    def load_and_prepare_data(self):
        """Load data and prepare features and target"""
        try:
            logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            
            # Separate features and target
            target_col = 'is_high_risk'
            feature_cols = [col for col in df.columns if col not in ['CustomerId', target_col]]
            
            X = df[feature_cols]
            y = df[target_col]
            
            # Handle any remaining missing values
            X = X.fillna(X.median())
            
            logger.info(f"Data shape: {X.shape}")
            logger.info(f"Target distribution:\n{y.value_counts()}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def split_data(self, X, y):
        """Split data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, 
            stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        logger.info(f"Training set shape: {self.X_train.shape}")
        logger.info(f"Test set shape: {self.X_test.shape}")
        
    def define_models(self):
        """Define models and their hyperparameter grids"""
        models_config = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'use_scaled': True
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=self.random_state),
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                },
                'use_scaled': False
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                },
                'use_scaled': False
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'use_scaled': False
            }
        }
        return models_config
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model and return metrics"""
        try:
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            logger.info(f"{model_name} Metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return metrics, y_pred, y_pred_proba
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
            raise
    
    def train_model_with_cv(self, model_name, model_config):
        """Train model with cross-validation and hyperparameter tuning"""
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            try:
                logger.info(f"Training {model_name}...")
                
                # Choose data based on scaling requirement
                X_train_data = self.X_train_scaled if model_config['use_scaled'] else self.X_train
                X_test_data = self.X_test_scaled if model_config['use_scaled'] else self.X_test
                
                # Hyperparameter tuning with RandomizedSearchCV for efficiency
                search = RandomizedSearchCV(
                    model_config['model'],
                    model_config['params'],
                    cv=5,
                    scoring='roc_auc',
                    n_iter=20,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                search.fit(X_train_data, self.y_train)
                best_model = search.best_estimator_
                
                # Evaluate on test set
                metrics, y_pred, y_pred_proba = self.evaluate_model(
                    best_model, X_test_data, self.y_test, model_name
                )
                
                # Log parameters and metrics to MLflow
                mlflow.log_params(search.best_params_)
                mlflow.log_metrics(metrics)
                mlflow.log_metric("cv_best_score", search.best_score_)
                
                # Log model
                signature = infer_signature(X_train_data, y_pred_proba)
                mlflow.sklearn.log_model(
                    best_model, 
                    f"model_{model_name}",
                    signature=signature
                )
                
                # Store model results
                self.models[model_name] = {
                    'model': best_model,
                    'metrics': metrics,
                    'best_params': search.best_params_,
                    'cv_score': search.best_score_
                }
                
                logger.info(f"Completed training {model_name}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                raise
    
    def train_all_models(self):
        """Train all models"""
        models_config = self.define_models()
        
        for model_name, config in models_config.items():
            self.train_model_with_cv(model_name, config)
    
    def select_best_model(self):
        """Select the best model based on ROC-AUC score"""
        best_score = 0
        best_name = None
        
        logger.info("\n" + "="*50)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*50)
        
        for name, results in self.models.items():
            roc_auc = results['metrics']['roc_auc']
            logger.info(f"{name}:")
            logger.info(f"  ROC-AUC: {roc_auc:.4f}")
            logger.info(f"  CV Score: {results['cv_score']:.4f}")
            logger.info(f"  Best Params: {results['best_params']}")
            logger.info("-" * 30)
            
            if roc_auc > best_score:
                best_score = roc_auc
                best_name = name
        
        self.best_model = self.models[best_name]['model']
        self.best_model_name = best_name
        
        logger.info(f"\nBest Model: {best_name} with ROC-AUC: {best_score:.4f}")
        
        return self.best_model, best_name
    
    def register_best_model(self):
        """Register the best model in MLflow Model Registry"""
        try:
            # Register model
            model_name = "credit-risk-model"
            
            with mlflow.start_run(run_name=f"best_model_registration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Use appropriate data for the best model
                use_scaled = self.define_models()[self.best_model_name]['use_scaled']
                X_train_data = self.X_train_scaled if use_scaled else self.X_train
                
                signature = infer_signature(X_train_data, self.best_model.predict_proba(X_train_data))
                
                model_uri = mlflow.sklearn.log_model(
                    self.best_model,
                    "best_model",
                    signature=signature,
                    registered_model_name=model_name
                ).model_uri
                
                # Log scaler if needed
                if use_scaled:
                    mlflow.sklearn.log_model(
                        self.scaler,
                        "scaler",
                        registered_model_name=f"{model_name}-scaler"
                    )
                
                # Log best model metadata
                mlflow.log_params(self.models[self.best_model_name]['best_params'])
                mlflow.log_metrics(self.models[self.best_model_name]['metrics'])
                mlflow.log_param("model_type", self.best_model_name)
                mlflow.log_param("use_scaled_features", use_scaled)
                
                logger.info(f"Best model registered: {model_name}")
                logger.info(f"Model URI: {model_uri}")
                
                # Save locally as well
                os.makedirs("models", exist_ok=True)
                joblib.dump(self.best_model, f"models/best_model_{self.best_model_name}.pkl")
                if use_scaled:
                    joblib.dump(self.scaler, "models/scaler.pkl")
                
                return model_uri
                
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            logger.info("Starting Credit Risk Model Training Pipeline")
            
            # Load and prepare data
            X, y = self.load_and_prepare_data()
            
            # Split data
            self.split_data(X, y)
            
            # Train all models
            self.train_all_models()
            
            # Select best model
            self.select_best_model()
            
            # Register best model
            model_uri = self.register_best_model()
            
            logger.info("Training pipeline completed successfully!")
            
            return {
                'best_model': self.best_model,
                'best_model_name': self.best_model_name,
                'model_uri': model_uri,
                'metrics': self.models[self.best_model_name]['metrics']
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run training"""
    trainer = CreditRiskModelTrainer(data_path="data/processed/processed_data.csv")
    results = trainer.run_training_pipeline()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Best Model: {results['best_model_name']}")
    print(f"Model URI: {results['model_uri']}")
    print("Metrics:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()