# restaurant_rating_prediction.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and explore the dataset"""
    print("Loading dataset...")
    
    df = pd.read_csv("dataset.csv", encoding="utf-8", sep="\t", quotechar='"', on_bad_lines="skip")

    
    print(f"Dataset Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Basic information
    print("\nDataset Info:")
    print(df.info())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    return df

def create_visualizations(df):
    plt.figure(figsize=(16, 12))
    plt.suptitle("Restaurant Rating Analysis - Comprehensive Visualizations", fontsize=18, fontweight="bold")

    # 1. Distribution of ratings
    plt.subplot(3, 3, 1)
    sns.histplot(df['Aggregate rating'], bins=20, kde=True, color="skyblue")
    plt.title("Distribution of Restaurant Ratings", fontsize=12)
    plt.xlabel("Aggregate Rating")
    plt.ylabel("Count")

    # 2. Rating by price range
    plt.subplot(3, 3, 2)
    sns.boxplot(x="Price range", y="Aggregate rating", data=df, palette="Set2")
    plt.title("Rating Distribution by Price Range", fontsize=12)
    plt.xlabel("Price Range")
    plt.ylabel("Aggregate Rating")

    # 3. Rating by online delivery
    plt.subplot(3, 3, 3)
    sns.boxplot(x="Has Online delivery", y="Aggregate rating", data=df, palette="Set3")
    plt.title("Rating by Online Delivery Availability", fontsize=12)
    plt.xlabel("Has Online Delivery")
    plt.ylabel("Aggregate Rating")

    # 4. Rating by table booking
    plt.subplot(3, 3, 4)
    sns.boxplot(x="Has Table booking", y="Aggregate rating", data=df, palette="Pastel1")
    plt.title("Rating by Table Booking Availability", fontsize=12)
    plt.xlabel("Has Table Booking")
    plt.ylabel("Aggregate Rating")

    # 5. Cost vs rating
    plt.subplot(3, 3, 5)
    sns.scatterplot(x="Average Cost for two", y="Aggregate rating", data=df, alpha=0.5, color="tomato")
    plt.title("Average Cost vs Rating", fontsize=12)
    plt.xlabel("Average Cost for Two")
    plt.ylabel("Aggregate Rating")

    # 6. Votes vs rating
    plt.subplot(3, 3, 6)
    sns.scatterplot(x="Votes", y="Aggregate rating", data=df, alpha=0.5, color="purple")
    plt.title("Number of Votes vs Rating", fontsize=12)
    plt.xlabel("Votes")
    plt.ylabel("Aggregate Rating")

    # 7. Top 10 cities by number of restaurants
    top_cities = df['City'].value_counts().nlargest(10)
    plt.subplot(3, 3, 7)
    sns.barplot(x=top_cities.values, y=top_cities.index, palette="Blues_r")
    plt.title("Top 10 Cities by Number of Restaurants", fontsize=12)
    plt.xlabel("Number of Restaurants")
    plt.ylabel("City")

    # 8. Top 10 cities by avg rating
    top_avg_rating = df.groupby('City')['Aggregate rating'].mean().nlargest(10)
    plt.subplot(3, 3, 8)
    sns.barplot(x=top_avg_rating.values, y=top_avg_rating.index, palette="Greens_r")
    plt.title("Top 10 Cities by Average Rating", fontsize=12)
    plt.xlabel("Average Rating")
    plt.ylabel("City")

    # 9. Correlation heatmap
    plt.subplot(3, 3, 9)
    corr = df[['Aggregate rating','Price range','Votes','Country Code','Average Cost for two','Latitude','Longitude','Restaurant ID']].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, cbar=True)
    plt.title("Correlation with Aggregate Rating", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # adjust to fit the suptitle
    plt.show()

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    print("\nPreprocessing data...")
    
    df_processed = df.copy()
    
    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    # Feature Engineering
    df_processed['City'] = df_processed['City'].fillna('Unknown')
    
    # Create binary features
    if 'Has Table booking' in df_processed.columns:
        df_processed['Has_Table_booking'] = df_processed['Has Table booking'].map({'Yes': 1, 'No': 0})
    if 'Has Online delivery' in df_processed.columns:
        df_processed['Has_Online_delivery'] = df_processed['Has Online delivery'].map({'Yes': 1, 'No': 0})
    
    # Encode categorical variables
    label_encoders = {}
    categorical_features = ['City', 'Currency', 'Rating color', 'Rating text']
    
    for feature in categorical_features:
        if feature in df_processed.columns:
            le = LabelEncoder()
            df_processed[feature] = le.fit_transform(df_processed[feature].astype(str))
            label_encoders[feature] = le
    
    # Select features for modeling
    features = [
        'Average Cost for two', 'Price range', 'Votes',
        'Longitude', 'Latitude'
    ]
    
    # Add binary features if they exist
    if 'Has_Table_booking' in df_processed.columns:
        features.append('Has_Table_booking')
    if 'Has_Online_delivery' in df_processed.columns:
        features.append('Has_Online_delivery')
    
    # Add encoded categorical features
    for feature in ['Currency', 'Rating color', 'City']:
        if feature in df_processed.columns:
            features.append(feature)
    
    target = 'Aggregate rating'
    
    return df_processed, features, target, label_encoders

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate multiple regression models"""
    print("\nTraining and evaluating models...")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
        
        print(f"{name} Results:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R² Score: {r2:.4f}")
    
    return results

def analyze_feature_importance(results, feature_names):
    """Analyze and visualize feature importance"""
    print("\nAnalyzing feature importance...")
    
    # Find the best model
    best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
    best_model = results[best_model_name]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importance = np.abs(best_model.coef_)
    else:
        print("Feature importance not available for this model type.")
        return None
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title(f'Top 15 Feature Importance - {best_model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    
    print(f"\nTop 10 Most Important Features ({best_model_name}):")
    for i, row in feature_importance.head(10).iterrows():
        print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
    
    return feature_importance, best_model_name

def plot_model_performance(results, y_test):
    """Plot model performance and predictions"""
    print("\nCreating model performance visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Model Comparison
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    rmse_scores = [results[name]['rmse'] for name in model_names]
    
    axes[0, 0].bar(model_names, r2_scores)
    axes[0, 0].set_title('R² Score Comparison')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].bar(model_names, rmse_scores)
    axes[0, 1].set_title('RMSE Comparison')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 2. Actual vs Predicted for best model
    best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
    best_predictions = results[best_model_name]['predictions']
    
    axes[1, 0].scatter(y_test, best_predictions, alpha=0.6)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual Ratings')
    axes[1, 0].set_ylabel('Predicted Ratings')
    axes[1, 0].set_title(f'Actual vs Predicted - {best_model_name}')
    
    # 3. Residuals analysis
    residuals = y_test - best_predictions
    axes[1, 1].scatter(best_predictions, residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title(f'Residuals vs Predicted - {best_model_name}')
    
    plt.tight_layout()
    plt.suptitle('Model Performance Analysis', fontsize=16, y=0.98)
    plt.show()

def main():
    """Main function to run the complete analysis"""
    print("=" * 60)
    print("RESTAURANT RATING PREDICTION ANALYSIS")
    print("=" * 60)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Create visualizations
    create_visualizations(df)
    
    # Preprocess data
    df_processed, features, target, label_encoders = preprocess_data(df)
    
    # Prepare data for modeling
    X = df_processed[features]
    y = df_processed[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, features)
    
    # Analyze feature importance
    feature_importance, best_model_name = analyze_feature_importance(results, features)
    
    # Plot model performance
    plot_model_performance(results, y_test)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    best_result = results[best_model_name]
    print(f"Best Model: {best_model_name}")
    print(f"R² Score: {best_result['r2']:.4f}")
    print(f"RMSE: {best_result['rmse']:.4f}")
    print(f"MAE: {best_result['mae']:.4f}")
    
    if feature_importance is not None:
        print("\nTop 5 Factors Affecting Restaurant Ratings:")
        for i, row in feature_importance.head(5).iterrows():
            print(f"{i+1}. {row['feature']} (Importance: {row['importance']:.4f})")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()