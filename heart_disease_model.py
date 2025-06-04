import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
# Using the Cleveland heart disease dataset from UCI Machine Learning Repository
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    df = pd.read_csv(url, names=column_names, na_values='?')
    
    # Clean the data
    df = df.dropna()
    
    # Convert target to binary (0 = no disease, 1 = disease)
    df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)
    
    return df

def explore_data(df):
    # Data exploration
    print("Data shape:", df.shape)
    print("\nData info:")
    print(df.info())
    print("\nData description:")
    print(df.describe())
    print("\nTarget distribution:")
    print(df['target'].value_counts())
    
    # Create some visualizations
    plt.figure(figsize=(12, 8))
    
    # Age distribution by target
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='age', hue='target', kde=True, bins=20)
    plt.title('Age Distribution by Heart Disease Status')
    
    # Cholesterol by target
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='target', y='chol')
    plt.title('Cholesterol Levels by Heart Disease Status')
    
    # Heart rate by target
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='target', y='thalach')
    plt.title('Max Heart Rate by Heart Disease Status')
    
    # Correlation matrix
    plt.subplot(2, 2, 4)
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", square=True)
    plt.title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('heart_disease_exploratory.png')
    print("Saved exploratory data analysis to heart_disease_exploratory.png")

def preprocess_data(df):
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for later use with new data
    joblib.dump(scaler, 'heart_disease_scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns)

def train_models(X_train, y_train):
    models = {}
    
    # Logistic Regression
    print("Training Logistic Regression model...")
    lr_params = {'C': [0.01, 0.1, 1, 10, 100]}
    lr = LogisticRegression(max_iter=1000)
    lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='accuracy')
    lr_grid.fit(X_train, y_train)
    models['logistic_regression'] = lr_grid.best_estimator_
    print(f"Best Logistic Regression params: {lr_grid.best_params_}")
    print(f"Best Logistic Regression CV score: {lr_grid.best_score_:.4f}")
    
    # Random Forest
    print("\nTraining Random Forest model...")
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy')
    rf_grid.fit(X_train, y_train)
    models['random_forest'] = rf_grid.best_estimator_
    print(f"Best Random Forest params: {rf_grid.best_params_}")
    print(f"Best Random Forest CV score: {rf_grid.best_score_:.4f}")
    
    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Save results
        results[name] = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm
        }
        
        # Print results
        print(f"\n{name.upper()} EVALUATION:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'{name}_confusion_matrix.png')
        
        # Feature importance (for Random Forest)
        if name == 'random_forest':
            feature_importances = pd.DataFrame({
                'Feature': X_test_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importances)
            plt.title('Feature Importance (Random Forest)')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            print("\nFeature Importance:")
            print(feature_importances)
    
    return results

def save_best_model(models, results):
    # Compare accuracies
    accuracies = {name: results[name]['accuracy'] for name in models.keys()}
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]
    
    # Save the model
    joblib.dump(best_model, 'heart_disease_model.pkl')
    print(f"\nBest model ({best_model_name}) saved as 'heart_disease_model.pkl' with accuracy: {accuracies[best_model_name]:.4f}")
    
    return best_model, best_model_name

if __name__ == "__main__":
    # Load data
    df = load_data()
    
    # Explore data
    explore_data(df)
    
    # Preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, X_test_columns = preprocess_data(df)
    
    # Train models
    models = train_models(X_train_scaled, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test_scaled, y_test)
    
    # Save the best model
    best_model, best_model_name = save_best_model(models, results)
    
    print("\nModel training and evaluation complete!")