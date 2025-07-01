import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import CreditRiskModelTrainer
from predict import CreditRiskPredictor

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'CustomerId': [1, 2, 3, 4, 5],
            'Recency': [1.5, -0.5, 0.0, 2.0, -1.0],
            'Frequency': [-0.5, 1.0, 0.5, -1.0, 0.0],
            'TotalAmount': [-0.2, 0.5, -1.0, 1.5, 0.0],
            'AvgAmount': [-0.3, 0.8, -0.5, 1.2, 0.1],
            'TransactionCount_x': [-0.4, 0.6, -0.8, 1.1, 0.2],
            'is_high_risk': [1, 0, 1, 0, 1]
        })
        
        # Save sample data for testing
        os.makedirs('test_data', exist_ok=True)
        self.test_data_path = 'test_data/test_processed_data.csv'
        self.sample_data.to_csv(self.test_data_path, index=False)
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)
        if os.path.exists('test_data'):
            os.rmdir('test_data')
    
    def test_data_loading_and_preparation(self):
        """Test data loading and preparation functionality"""
        trainer = CreditRiskModelTrainer(data_path=self.test_data_path, random_state=42)
        
        # Test data loading
        X, y = trainer.load_and_prepare_data()
        
        # Assertions
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), 5)
        self.assertEqual(len(y), 5)
        self.assertNotIn('CustomerId', X.columns)
        self.assertNotIn('is_high_risk', X.columns)
        self.assertTrue(all(col in X.columns for col in ['Recency', 'Frequency', 'TotalAmount']))
    
    def test_data_splitting(self):
        """Test data splitting functionality"""
        trainer = CreditRiskModelTrainer(data_path=self.test_data_path, test_size=0.4, random_state=42)
        
        # Load data
        X, y = trainer.load_and_prepare_data()
        
        # Split data
        trainer.split_data(X, y)
        
        # Assertions
        self.assertIsNotNone(trainer.X_train)
        self.assertIsNotNone(trainer.X_test)
        self.assertIsNotNone(trainer.y_train)
        self.assertIsNotNone(trainer.y_test)
        
        # Check split proportions (approximately)
        total_samples = len(X)
        expected_test_size = int(total_samples * 0.4)
        self.assertGreaterEqual(len(trainer.X_test), expected_test_size - 1)
        self.assertLessEqual(len(trainer.X_test), expected_test_size + 1)
    
    def test_model_evaluation_metrics(self):
        """Test model evaluation metrics calculation"""
        trainer = CreditRiskModelTrainer(data_path=self.test_data_path, random_state=42)
        
        # Create a mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 0, 1, 0])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.6, 0.4]])
        
        # Mock test data
        X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_test = np.array([1, 0, 1, 0])
        
        # Test evaluation
        metrics, y_pred, y_pred_proba = trainer.evaluate_model(mock_model, X_test, y_test, "test_model")
        
        # Assertions
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('roc_auc', metrics)
        
        # Check metric values are reasonable
        for metric_name, metric_value in metrics.items():
            self.assertGreaterEqual(metric_value, 0)
            self.assertLessEqual(metric_value, 1)
    
    def test_feature_scaling_consistency(self):
        """Test that feature scaling is applied consistently"""
        trainer = CreditRiskModelTrainer(data_path=self.test_data_path, random_state=42)
        
        # Load and split data
        X, y = trainer.load_and_prepare_data()
        trainer.split_data(X, y)
        
        # Check that scaled data has same shape
        self.assertEqual(trainer.X_train_scaled.shape, trainer.X_train.shape)
        self.assertEqual(trainer.X_test_scaled.shape, trainer.X_test.shape)
        
        # Check that scaling was applied (means should be close to 0 for scaled data)
        scaled_means = np.mean(trainer.X_train_scaled, axis=0)
        self.assertTrue(all(abs(mean) < 1e-10 for mean in scaled_means))
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    def test_mlflow_logging(self, mock_log_metrics, mock_log_params, mock_start_run):
        """Test MLflow logging functionality"""
        # Setup mock context manager
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()
        
        trainer = CreditRiskModelTrainer(data_path=self.test_data_path, random_state=42)
        
        # Test that MLflow functions would be called
        # This is a basic test to ensure the structure is correct
        self.assertTrue(hasattr(trainer, 'train_model_with_cv'))
        self.assertTrue(hasattr(trainer, 'register_best_model'))

class TestCreditRiskPredictor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for predictor"""
        self.sample_input = {
            'Recency': 1.5,
            'Frequency': -0.5,
            'TotalAmount': -0.2,
            'AvgAmount': -0.3,
            'TransactionCount_x': -0.4
        }
    
    @patch('mlflow.sklearn.load_model')
    def test_risk_probability_calculation(self, mock_load_model):
        """Test risk probability calculation"""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_load_model.return_value = mock_model
        
        # Initialize predictor with mock
        predictor = CreditRiskPredictor()
        predictor.model = mock_model
        predictor.use_scaled_features = False
        
        # Test prediction
        risk_prob, risk_category = predictor.predict_risk_probability(self.sample_input)
        
        # Assertions
        self.assertEqual(len(risk_prob), 1)
        self.assertEqual(risk_prob[0], 0.7)
        self.assertEqual(risk_category[0], "High Risk")
    
    def test_credit_score_calculation(self):
        """Test credit score calculation"""
        predictor = CreditRiskPredictor.__new__(CreditRiskPredictor)  # Create without __init__
        
        # Test different risk probabilities
        test_cases = [
            (0.1, 795),   # Low risk -> High score
            (0.5, 575),   # Medium risk -> Medium score
            (0.9, 355),   # High risk -> Low score
        ]
        
        for risk_prob, expected_score in test_cases:
            score = predictor.predict_credit_score(risk_prob)
            self.assertAlmostEqual(score, expected_score, delta=5)
            self.assertGreaterEqual(score, 300)
            self.assertLessEqual(score, 850)
    
    def test_loan_terms_calculation(self):
        """Test loan terms recommendation"""
        predictor = CreditRiskPredictor.__new__(CreditRiskPredictor)  # Create without __init__
        
        # Test low risk customer
        low_risk_terms = predictor._calculate_loan_terms(0.2, 30000)
        self.assertEqual(low_risk_terms["approval_status"], "Approved")
        self.assertEqual(low_risk_terms["recommended_amount"], 30000)
        self.assertEqual(low_risk_terms["max_amount"], 50000)
        
        # Test high risk customer
        high_risk_terms = predictor._calculate_loan_terms(0.8, 30000)
        self.assertEqual(high_risk_terms["approval_status"], "Conditional approval")
        self.assertEqual(high_risk_terms["recommended_amount"], 10000)  # Capped at max
        self.assertGreater(high_risk_terms["interest_rate"], 0.1)  # Higher interest rate
    
    def test_input_data_preprocessing(self):
        """Test input data preprocessing"""
        predictor = CreditRiskPredictor.__new__(CreditRiskPredictor)  # Create without __init__
        predictor.use_scaled_features = False
        predictor.scaler = None
        
        # Test dict input
        dict_input = {'feature1': 1.0, 'feature2': 2.0}
        processed = predictor.preprocess_input(dict_input)
        self.assertEqual(processed.shape[0], 1)
        
        # Test DataFrame input
        df_input = pd.DataFrame([{'feature1': 1.0, 'feature2': 2.0}])
        processed = predictor.preprocess_input(df_input)
        self.assertEqual(processed.shape[0], 1)

class TestHelperFunctions(unittest.TestCase):
    
    def test_monthly_payment_calculation(self):
        """Test monthly payment calculation helper function"""
        predictor = CreditRiskPredictor.__new__(CreditRiskPredictor)
        
        # Test with interest
        payment = predictor._calculate_monthly_payment(10000, 0.06, 12)
        self.assertGreater(payment, 800)  # Should be more than principal/months
        self.assertLess(payment, 1000)    # But reasonable
        
        # Test with zero interest
        payment_zero = predictor._calculate_monthly_payment(12000, 0.0, 12)
        self.assertEqual(payment_zero, 1000)  # Should equal principal/months
    
    def test_risk_category_assignment(self):
        """Test risk category assignment logic"""
        predictor = CreditRiskPredictor.__new__(CreditRiskPredictor)
        predictor.use_scaled_features = False
        predictor.scaler = None
        
        # Mock model for testing
        mock_model = MagicMock()
        
        # Test different probability ranges
        test_cases = [
            (0.1, "Low Risk"),
            (0.4, "Medium Risk"), 
            (0.8, "High Risk")
        ]
        
        for prob, expected_category in test_cases:
            mock_model.predict_proba.return_value = np.array([[1-prob, prob]])
            predictor.model = mock_model
            
            _, categories = predictor.predict_risk_probability({'test': 1})
            self.assertEqual(categories[0], expected_category)

if __name__ == '__main__':
    # Create test data directory if it doesn't exist
    os.makedirs('test_data', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)