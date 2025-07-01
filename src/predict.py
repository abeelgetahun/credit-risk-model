import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import logging
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditRiskPredictor:
    def __init__(self, model_name="credit-risk-model", model_version="latest"):
        """
        Initialize the Credit Risk Predictor
        
        Args:
            model_name (str): Name of the registered model
            model_version (str): Version of the model to load
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.scaler = None
        self.use_scaled_features = False
        self.feature_columns = None
        
        # Setup MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model from MLflow registry"""
        try:
            # Load model from registry
            model_uri = f"models:/{self.model_name}/{self.model_version}"
            self.model = mlflow.sklearn.load_model(model_uri)
            
            # Try to load run info to get metadata
            try:
                client = mlflow.tracking.MlflowClient()
                model_version_info = client.get_model_version(self.model_name, self.model_version)
                run_id = model_version_info.run_id
                run = client.get_run(run_id)
                
                # Get scaling information
                self.use_scaled_features = run.data.params.get('use_scaled_features', 'False') == 'True'
                
                # Load scaler if needed
                if self.use_scaled_features:
                    try:
                        scaler_uri = f"models:/{self.model_name}-scaler/{self.model_version}"
                        self.scaler = mlflow.sklearn.load_model(scaler_uri)
                    except:
                        # Fallback to local scaler
                        self.scaler = joblib.load("models/scaler.pkl")
                
            except Exception as e:
                logger.warning(f"Could not load run metadata: {str(e)}")
                # Try to load local scaler as fallback
                try:
                    self.scaler = joblib.load("models/scaler.pkl")
                    self.use_scaled_features = True
                except:
                    self.use_scaled_features = False
            
            logger.info(f"Model loaded successfully: {model_uri}")
            logger.info(f"Using scaled features: {self.use_scaled_features}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to local model
            try:
                import glob
                model_files = glob.glob("models/best_model_*.pkl")
                if model_files:
                    self.model = joblib.load(model_files[0])
                    logger.info(f"Loaded local model: {model_files[0]}")
                    
                    # Try to load scaler
                    try:
                        self.scaler = joblib.load("models/scaler.pkl")
                        self.use_scaled_features = True
                    except:
                        self.use_scaled_features = False
                else:
                    raise Exception("No model found")
            except Exception as fallback_error:
                logger.error(f"Fallback loading failed: {str(fallback_error)}")
                raise
    
    def preprocess_input(self, data):
        """Preprocess input data for prediction"""
        try:
            # Ensure data is DataFrame
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            elif isinstance(data, pd.Series):
                data = pd.DataFrame([data])
            
            # Handle missing values
            data = data.fillna(data.median())
            
            # Apply scaling if needed
            if self.use_scaled_features and self.scaler is not None:
                data_processed = self.scaler.transform(data)
            else:
                data_processed = data
            
            return data_processed
            
        except Exception as e:
            logger.error(f"Error preprocessing input: {str(e)}")
            raise
    
    def predict_risk_probability(self, data):
        """
        Predict risk probability for given data
        
        Args:
            data: Input data (DataFrame, Series, or dict)
            
        Returns:
            tuple: (risk_probability, risk_category)
        """
        try:
            # Preprocess data
            processed_data = self.preprocess_input(data)
            
            # Get probability predictions
            probabilities = self.model.predict_proba(processed_data)
            risk_probability = probabilities[:, 1]  # Probability of high risk (class 1)
            
            # Determine risk category
            risk_categories = []
            for prob in risk_probability:
                if prob < 0.3:
                    category = "Low Risk"
                elif prob < 0.7:
                    category = "Medium Risk"
                else:
                    category = "High Risk"
                risk_categories.append(category)
            
            return risk_probability, risk_categories
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def predict_credit_score(self, risk_probability):
        """
        Convert risk probability to credit score (300-850 scale)
        
        Args:
            risk_probability: Risk probability (0-1)
            
        Returns:
            int: Credit score
        """
        try:
            # Convert probability to score (inverse relationship)
            # Higher risk probability = lower credit score
            if isinstance(risk_probability, (list, np.ndarray)):
                scores = []
                for prob in risk_probability:
                    score = int(850 - (prob * 550))  # Scale to 300-850
                    score = max(300, min(850, score))  # Ensure within bounds
                    scores.append(score)
                return scores
            else:
                score = int(850 - (risk_probability * 550))
                return max(300, min(850, score))
                
        except Exception as e:
            logger.error(f"Error calculating credit score: {str(e)}")
            raise
    
    def recommend_loan_terms(self, risk_probability, requested_amount=None):
        """
        Recommend loan amount and duration based on risk
        
        Args:
            risk_probability: Risk probability (0-1)
            requested_amount: Requested loan amount (optional)
            
        Returns:
            dict: Loan recommendations
        """
        try:
            recommendations = []
            
            if isinstance(risk_probability, (list, np.ndarray)):
                for prob in risk_probability:
                    rec = self._calculate_loan_terms(prob, requested_amount)
                    recommendations.append(rec)
                return recommendations
            else:
                return self._calculate_loan_terms(risk_probability, requested_amount)
                
        except Exception as e:
            logger.error(f"Error calculating loan terms: {str(e)}")
            raise
    
    def _calculate_loan_terms(self, risk_prob, requested_amount):
        """Calculate individual loan terms"""
        if risk_prob < 0.3:  # Low risk
            max_amount = 50000
            max_duration = 60  # months
            interest_rate = 0.05  # 5%
            approval_status = "Approved"
        elif risk_prob < 0.7:  # Medium risk
            max_amount = 25000
            max_duration = 36
            interest_rate = 0.08  # 8%
            approval_status = "Approved with conditions"
        else:  # High risk
            max_amount = 10000
            max_duration = 12
            interest_rate = 0.15  # 15%
            approval_status = "Conditional approval"
        
        # Adjust amount if requested
        if requested_amount:
            recommended_amount = min(requested_amount, max_amount)
        else:
            recommended_amount = max_amount
        
        return {
            "approval_status": approval_status,
            "recommended_amount": recommended_amount,
            "max_amount": max_amount,
            "recommended_duration_months": max_duration,
            "interest_rate": interest_rate,
            "monthly_payment": self._calculate_monthly_payment(
                recommended_amount, interest_rate, max_duration
            )
        }
    
    def _calculate_monthly_payment(self, principal, annual_rate, months):
        """Calculate monthly payment"""
        monthly_rate = annual_rate / 12
        if monthly_rate == 0:
            return principal / months
        else:
            return (principal * monthly_rate * (1 + monthly_rate)**months) / \
                   ((1 + monthly_rate)**months - 1)
    
    def comprehensive_assessment(self, data, requested_amount=None):
        """
        Perform comprehensive risk assessment
        
        Args:
            data: Customer data
            requested_amount: Requested loan amount
            
        Returns:
            dict: Complete assessment
        """
        try:
            # Get risk probability
            risk_prob, risk_category = self.predict_risk_probability(data)
            
            # Calculate credit score
            credit_score = self.predict_credit_score(risk_prob)
            
            # Get loan recommendations
            loan_terms = self.recommend_loan_terms(risk_prob, requested_amount)
            
            # Handle single vs multiple predictions
            if isinstance(risk_prob, np.ndarray) and len(risk_prob) == 1:
                risk_prob = risk_prob[0]
                risk_category = risk_category[0]
                credit_score = credit_score[0] if isinstance(credit_score, list) else credit_score
            
            return {
                "risk_probability": float(risk_prob),
                "risk_category": risk_category,
                "credit_score": int(credit_score),
                "loan_recommendations": loan_terms
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive assessment: {str(e)}")
            raise

def main():
    """Demo function"""
    # Initialize predictor
    predictor = CreditRiskPredictor()
    
    # Sample data (you would replace this with actual customer data)
    sample_data = {
        'Recency': 1.5,
        'Frequency': -0.5,
        'TotalAmount': -0.2,
        'AvgAmount': -0.3,
        'StdAmount': -0.1,
        'TransactionCount_x': -0.4,
        'AmountSum': -0.2,
        'AmountMean': -0.3,
        'AmountStd': -0.1,
        'AmountMin': -0.2,
        'AmountMax': -0.2,
        'AmountCount': -0.4,
        'ValueSum': -0.2,
        'ValueMean': -0.1,
        'ValueStd': -0.1,
        'ValueMin': 0.0,
        'ValueMax': -0.1,
        'TransactionCount_y': -0.4,
        'UniqueChannels': -1.0,
        'UniqueProductCategories': -1.0,
        'UniqueProviders': -1.0,
        'UniqueProducts': -1.0,
        'TotalFraudCases': -0.1,
        'FraudRate': -0.1,
        'AvgTransactionValue': -0.1,
        'AmountValueRatio': -4.0,
        'TransactionConsistency': 1.5,
        'AvgTransactionHour': 0.5,
        'StdTransactionHour': -1.0,
        'AvgTransactionDay': 0.5,
        'StdTransactionDay': 0.5,
        'UniqueMonthsActive': 0.5,
        'AvgDayOfWeek': 0.5,
        'StdDayOfWeek': -0.5,
        'WeekendTransactionRatio': -0.5,
        'ActivitySpanDays': -0.5,
        'TemporalTransactionCount': -0.4
    }
    
    # Perform assessment
    assessment = predictor.comprehensive_assessment(sample_data, requested_amount=20000)
    
    print("Credit Risk Assessment Results:")
    print(f"Risk Probability: {assessment['risk_probability']:.3f}")
    print(f"Risk Category: {assessment['risk_category']}")
    print(f"Credit Score: {assessment['credit_score']}")
    print("\nLoan Recommendations:")
    for key, value in assessment['loan_recommendations'].items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()