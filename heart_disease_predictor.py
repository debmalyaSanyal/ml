import joblib
import numpy as np

class HeartDiseasePredictor:
    """
    A class for making heart disease predictions based on pre-trained models.
    """
    
    def __init__(self, model_path='heart_disease_model.pkl', scaler_path='heart_disease_scaler.pkl'):
        """
        Initialize the predictor with the trained model and scaler.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file
        scaler_path : str
            Path to the saved scaler file
        """
        # Load the model and scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Feature names in the correct order
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        # Feature descriptions and valid ranges for user input validation
        self.feature_info = {
            'age': {
                'description': 'Age in years',
                'type': 'int',
                'min': 20,
                'max': 100
            },
            'sex': {
                'description': 'Sex (0 = female, 1 = male)',
                'type': 'binary',
                'values': [0, 1]
            },
            'cp': {
                'description': 'Chest pain type (0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic)',
                'type': 'categorical',
                'values': [0, 1, 2, 3]
            },
            'trestbps': {
                'description': 'Resting blood pressure (mm Hg)',
                'type': 'int',
                'min': 80,
                'max': 200
            },
            'chol': {
                'description': 'Serum cholesterol (mg/dl)',
                'type': 'int',
                'min': 100,
                'max': 600
            },
            'fbs': {
                'description': 'Fasting blood sugar > 120 mg/dl (0 = false, 1 = true)',
                'type': 'binary',
                'values': [0, 1]
            },
            'restecg': {
                'description': 'Resting electrocardiographic results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)',
                'type': 'categorical',
                'values': [0, 1, 2]
            },
            'thalach': {
                'description': 'Maximum heart rate achieved',
                'type': 'int',
                'min': 60,
                'max': 220
            },
            'exang': {
                'description': 'Exercise induced angina (0 = no, 1 = yes)',
                'type': 'binary',
                'values': [0, 1]
            },
            'oldpeak': {
                'description': 'ST depression induced by exercise relative to rest',
                'type': 'float',
                'min': 0,
                'max': 10
            },
            'slope': {
                'description': 'Slope of the peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)',
                'type': 'categorical',
                'values': [0, 1, 2]
            },
            'ca': {
                'description': 'Number of major vessels colored by fluoroscopy (0-4)',
                'type': 'categorical',
                'values': [0, 1, 2, 3, 4]
            },
            'thal': {
                'description': 'Thalassemia (1: Normal, 2: Fixed defect, 3: Reversible defect)',
                'type': 'categorical',
                'values': [1, 2, 3]
            }
        }
    
    def get_feature_info(self):
        """Returns feature information for the UI"""
        return self.feature_info
    
    def validate_input(self, input_data):
        """
        Validate the input data against the expected ranges and types.
        
        Parameters:
        -----------
        input_data : dict
            Dictionary containing feature values
            
        Returns:
        --------
        tuple
            (is_valid, error_message)
        """
        missing_features = [f for f in self.feature_names if f not in input_data]
        if missing_features:
            return False, f"Missing features: {', '.join(missing_features)}"
        
        for feature, value in input_data.items():
            if feature not in self.feature_info:
                return False, f"Unknown feature: {feature}"
            
            info = self.feature_info[feature]
            
            # Check type
            if info['type'] == 'int':
                try:
                    input_data[feature] = int(value)
                    if not (info['min'] <= input_data[feature] <= info['max']):
                        return False, f"{feature} should be between {info['min']} and {info['max']}"
                except ValueError:
                    return False, f"{feature} should be an integer"
                
            elif info['type'] == 'float':
                try:
                    input_data[feature] = float(value)
                    if not (info['min'] <= input_data[feature] <= info['max']):
                        return False, f"{feature} should be between {info['min']} and {info['max']}"
                except ValueError:
                    return False, f"{feature} should be a number"
                
            elif info['type'] in ['binary', 'categorical']:
                try:
                    input_data[feature] = int(value)
                    if input_data[feature] not in info['values']:
                        return False, f"{feature} should be one of {info['values']}"
                except ValueError:
                    return False, f"{feature} should be one of {info['values']}"
        
        return True, ""
    
    def predict(self, input_data):
        """
        Make a prediction based on input data.
        
        Parameters:
        -----------
        input_data : dict
            Dictionary containing feature values
            
        Returns:
        --------
        dict
            Prediction results including probability and risk level
        """
        # Validate input
        is_valid, error_message = self.validate_input(input_data)
        if not is_valid:
            return {'error': error_message}
        
        # Convert input to feature array in the correct order
        features = np.array([[input_data[f] for f in self.feature_names]])
        
        # Scale the features
        scaled_features = self.scaler.transform(features)
        
        # Get prediction probability
        try:
            pred_prob = self.model.predict_proba(scaled_features)[0][1]
            prediction = 1 if pred_prob >= 0.5 else 0
            
            # Determine risk level
            if pred_prob < 0.2:
                risk_level = "Low"
                recommendation = "Maintain a healthy lifestyle."
            elif pred_prob < 0.5:
                risk_level = "Moderate"
                recommendation = "Consider discussing these results with your doctor."
            elif pred_prob < 0.7:
                risk_level = "High"
                recommendation = "We recommend consulting with a healthcare professional soon."
            else:
                risk_level = "Very High"
                recommendation = "Please consult a healthcare professional as soon as possible."
            
            return {
                'prediction': prediction,
                'probability': float(pred_prob),
                'risk_level': risk_level,
                'recommendation': recommendation,
                'final_tip': "This is a machine learning prediction. Always consult healthcare professionals for medical diagnosis."
            }
            
        except Exception as e:
            return {'error': f"Prediction error: {str(e)}"}






# # Example usage
# if __name__ == "__main__":
#     # Create a predictor instance
#     predictor = HeartDiseasePredictor()
    
#     # Example input data
#     sample_input = {
#         'age': 52,
#         'sex': 1,
#         'cp': 0,
#         'trestbps': 125,
#         'chol': 212,
#         'fbs': 0,
#         'restecg': 1,
#         'thalach': 168,
#         'exang': 0,
#         'oldpeak': 1.0,
#         'slope': 2,
#         'ca': 2,
#         'thal': 3
#     }
    
#     # Get prediction
#     result = predictor.predict(sample_input)
#     print("Prediction result:", result)




