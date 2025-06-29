import pandas as pd
import numpy as np
import joblib
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set Streamlit page config
st.set_page_config(page_title="Disease Risk Prediction Analysis", layout="wide")

# Set random seed
np.random.seed(42)

# Load dataset
try:
    df = pd.read_csv("Health_Dataset_1000_Updated-FF.csv")
except FileNotFoundError:
    st.error("Error: Dataset 'Health_Dataset_1000_Updated-FF.csv' not found.")
    st.stop()

# Validate dataset integrity
if df.empty:
    st.error("Error: Dataset is empty.")
    st.stop()
if df.isnull().any().any():
    st.warning("Warning: Dataset contains missing values. Filling with mode for categorical and mean for numerical.")
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

# Define target columns
target_columns = ['Risk_Level_Diabetes', 'Risk_Level_Hypertension', 'Risk_Level_Heart_Disease']
if not all(col in df.columns for col in target_columns):
    st.error(f"Error: Dataset missing required target columns: {target_columns}")
    st.stop()

# Define features (all columns except target columns)
features = [col for col in df.columns if col not in target_columns]

# Define numerical and categorical features
numerical_features = ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Blood_Sugar', 'Sleep_Hours']
categorical_features = [col for col in features if col not in numerical_features]

# Label encode categorical columns
encoders = {}
for col in categorical_features:
    if col in df.columns:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col].astype(str))
        encoders[col] = enc
os.makedirs('models', exist_ok=True)
joblib.dump(encoders, "models/label_encoders.pkl")

# Label encode target columns
target_encoders = {}
for col in target_columns:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col].astype(str))
    target_encoders[col] = enc
joblib.dump(target_encoders, "models/target_encoders.pkl")

# Calculate information gain
def calculate_information_gain(X, y):
    try:
        info_gain = mutual_info_classif(X, y, random_state=42)
        return pd.Series(info_gain, index=X.columns).sort_values(ascending=False)
    except Exception as e:
        st.error(f"Error calculating information gain: {str(e)}")
        return pd.Series()

# Select features based on threshold
def select_features(feature_ig, threshold):
    return feature_ig[feature_ig >= threshold].index.tolist()

# Evaluate models for each target individually
def evaluate_models(X, target_columns):
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'k-NN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    training_results_dict = {}
    test_results_dict = {}
    threshold_metrics_dict = {}
    best_features_per_threshold_dict = {}
    best_models_dict = {}

    for target in target_columns:
        y = df[target]
        training_results = []
        test_results = []
        threshold_metrics = {}
        best_features_per_threshold = {}
        
        # Check class distribution for stratification
        class_counts = y.value_counts()
        if any(class_counts < 5):
            st.warning(f"Low sample count for some classes in {target}. Using non-stratified split.")
            stratify = None
        else:
            stratify = y
        
        # Calculate information gain
        feature_ig = calculate_information_gain(X, y)
        if feature_ig.empty:
            st.error(f"Failed to compute feature importance for {target}.")
            continue
        
        # Select features for thresholds
        thresholds = [0.01, 0.03, 0.05]
        feature_subsets = {f'threshold_{t}': select_features(feature_ig, t) for t in thresholds}
        best_features_per_threshold = {k: v for k, v in feature_subsets.items() if v}
        
        if not best_features_per_threshold:
            st.warning(f"No features selected for any threshold for {target}.")
            continue
        
        # Use highest threshold for training metrics
        highest_threshold = max([float(k.replace('threshold_', '')) for k in best_features_per_threshold.keys()])
        best_features = best_features_per_threshold[f'threshold_{highest_threshold}']
        X_subset = X[best_features]
        
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.3, random_state=42, stratify=stratify)
        
        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                training_results.append({
                    'Target': target,
                    'Model': model_name,
                    'Accuracy': round(accuracy_score(y_train, y_train_pred), 4),
                    'Precision': round(precision_score(y_train, y_train_pred, average='weighted', zero_division=0), 4),
                    'Recall': round(recall_score(y_train, y_train_pred, average='weighted', zero_division=0), 4),
                    'F1-Score': round(f1_score(y_train, y_train_pred, average='weighted', zero_division=0), 4)
                })
            except Exception as e:
                st.warning(f"Error evaluating {model_name} for {target}: {str(e)}")
        
        # Evaluate thresholds with cross-validation
        for threshold_key, features in feature_subsets.items():
            if not features:
                st.warning(f"No features selected for {threshold_key} for {target}.")
                continue
            threshold = float(threshold_key.replace('threshold_', ''))
            X_subset = X[features]
            X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.3, random_state=42, stratify=stratify)
            
            model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=20)
            try:
                cv_accuracy = cross_val_score(model, X_subset, y, cv=5, scoring='accuracy').mean()
                cv_precision = cross_val_score(model, X_subset, y, cv=5, scoring='precision_weighted').mean()
                cv_recall = cross_val_score(model, X_subset, y, cv=5, scoring='recall_weighted').mean()
                threshold_metrics[threshold] = {
                    'Accuracy': round(cv_accuracy, 4),
                    'Precision': round(cv_precision, 4),
                    'Recall': round(cv_recall, 4)
                }
            except Exception as e:
                st.warning(f"Cross-validation failed for {threshold_key} in {target}: {str(e)}")
                threshold_metrics[threshold] = {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0}
            
            # Use precomputed feature_ig for avg_info_gain
            try:
                avg_info_gain = np.mean([feature_ig[f] for f in features if f in feature_ig]) if features else 0.0
            except Exception as e:
                st.warning(f"Error calculating average information gain for {threshold_key} in {target}: {str(e)}")
                avg_info_gain = 0.0
            
            for model_name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_test_pred = model.predict(X_test)
                    test_results.append({
                        'Target': target,
                        'Feature Set': threshold_key,
                        'Model': model_name,
                        'Accuracy': round(accuracy_score(y_test, y_test_pred, 4),
                        'Precision': round(precision_score(y_test_pred, average='weighted', zero_division=0)),
                        'Recall': round(recall_score(y_test, y_test_pred, average='weighted', zero_division=0)),
                        'F1': round(f1_score(y_test, y_test_pred, average='weighted', zero_division=0)),
                        'Information Gain': round(avg_info_gain, 4),
                        'Threshold': threshold
                    })
                except Exception as e:
                    st.warning(f"Error evaluating {model_name} with {threshold_key} for {target}: {str(e)}")
        
        training_results_dict[target] = pd.DataFrame(training_results)
        test_results_dict[target] = pd.DataFrame(test_results)
        threshold_metrics_dict[target] = threshold_metrics
        best_features_per_threshold_dict[target] = best_features_per_threshold
        
        if not test_results_dict[target].empty:
            best_result = test_results_dict[target].loc[test_results_dict[target]['Accuracy'].idxmax()]
            best_models_dict[target] = {
                'best_model_name': best_result['Model'],
                'feature_set': best_result['Feature Set'],
                'accuracy': best_result['Accuracy']
            }
    
    return training_results_dict, test_results_dict, threshold_metrics_dict, best_features_per_threshold_dict, best_models_dict

# Train and save best model
def train_and_save_best_model(X, target, best_features, best_model_name):
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'k-NN': KNeighborsClassifier(),
        'NaiveBayes': GaussianNB(),
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }
    y = df[target]
    X_best = X[best_features]
    stratify = y if all(y.value_counts() >= 5) else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size=0.3, random_state=42, stratify=stratify)
    except ValueError:
        st.warning(f"Stratified split failed for {target}. Using non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size=0.3, random_state=42)
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, f"models/best_model_{target}.pkl")
    joblib.dump(best_features, f"models/features_{target}.pkl")

# Main Streamlit application
def main():
    st.title("Risk Prediction Model Training Analysis")
    
    X = df[features]
    
    # Display class distribution
    st.subheader("Class Distribution")
    for target in target_columns:
        st.write(f"**{target}**")
        class_counts = df[target].value_counts()
        st.write(class_counts)
        if df[target].nunique() < 2:
            st.error(f"Error: Target '{target}' has insufficient unique values: {df[target].unique()}")
            st.stop()
    
    # Evaluate models
    training_results_dict, test_results_dict, threshold_metrics_dict, best_features_per_threshold_dict, best_models_dict = evaluate_models(X, target_columns)
    
    # Training metrics table
    st.subheader("Training Metrics for All Models (Averaged Across Diseases)")
    all_training_results = pd.concat(training_results_dict.values(), ignore_index=True)
    if not all_training_results.empty:
        avg_training_metrics = all_training_results.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].mean().reset_index()
        pivot_table = avg_training_metrics.pivot_table(
            index='Model',
            values=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            aggfunc='first'
        ).reindex(['Naive Bayes', 'Decision Tree', 'k-NN', 'Random Forest', 'SVM'])
        st.dataframe(pivot_table.style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}'
        }))
    else:
        st.warning("No training metrics available.")
    
    # Threshold-based feature importance
    st.subheader("Feature Importance and Metrics by Threshold")
    for target in target_columns:
        st.write(f"**{target}**")
        for threshold in [0.01, 0.03, 0.05]:
            st.write(f"Threshold {threshold}")
            feature_ig = calculate_information_gain(X, df[target])
            selected_features = select_features(feature_ig, threshold)
            if selected_features:
                table_data = []
                for feature in features:
                    info_gain = feature_ig.get(feature, 0.0)
                    metrics = threshold_metrics_dict[target].get(threshold, {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0})
                    table_data.append([feature, round(info_gain, 4), threshold, metrics['Accuracy'], metrics['Precision'], metrics['Recall']])
                
                threshold_table = pd.DataFrame(table_data, columns=['Feature', 'Information Gain', 'Threshold', 'Accuracy', 'Precision', 'Recall'])
                st.dataframe(threshold_table.style.format({
                    'Information Gain': '{:.4f}',
                    'Threshold': '{:.2f}',
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}'
                }))
            else:
                st.warning(f"No features selected for threshold {threshold} for {target}.")
    
    # Line chart visualization
    st.subheader("Performance Visualization by Disease")
    for target in target_columns:
        st.write(f"**{target}**")
        if target in test_results_dict and not test_results_dict[target].empty:
            all_models = ['Decision Tree', 'k-NN', 'Naive Bayes', 'Random Forest', 'SVM']
            plot_data = test_results_dict[target].pivot_table(index='Feature Set', columns='Model', values='Accuracy', aggfunc='mean').reindex(columns=all_models, fill_value=np.nan)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            for model in all_models:
                model_data = plot_data[model].dropna()
                if not model_data.empty:
                    ax.plot(model_data.index, model_data, marker='o', label=model, linewidth=2)
            
            ax.set_title(f'Model Accuracy Across Feature Sets for {target}')
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Feature Set')
            ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            os.makedirs('plots', exist_ok=True)
            fig.savefig(f'plots/model_performance_{target}.png')
            plt.close(fig)
    
    # Bar chart for model comparison
    st.subheader("Model Comparison (Metrics) Averaged Across Diseases")
    all_test_results = pd.concat(test_results_dict.values(), ignore_index=True)
    if not all_test_results.empty:
        all_models = ['Decision Tree', 'k-NN', 'Naive Bayes', 'Random Forest', 'SVM']
        avg_metrics = all_test_results.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].mean().reindex(all_models)
        
        bar_width = 0.2
        index = np.arange(len(all_models))
        
        fig, ax = plt.subplots(figsize=(12, 6))  # Increased size for more space around bars
        
        ax.bar(index, avg_metrics['Accuracy'], bar_width, label='Accuracy', color='blue')
        ax.bar(index + bar_width, avg_metrics['Precision'], bar_width, label='Precision', color='orange')
        ax.bar(index + 2 * bar_width, avg_metrics['Recall'], bar_width, label='Recall', color='green')
        ax.bar(index + 3 * bar_width, avg_metrics['F1-Score'], bar_width, label='F1-Score', color='red')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Metrics')
        ax.set_title('Model Comparison (Average Metrics Across Diseases)')
        ax.set_xticks(index + 1.5 * bar_width)
        ax.set_xticklabels(all_models, rotation=45)
        ax.set_ylim(0, 1.2)  # Increased y-axis limit to 1.2
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])  # Updated ticks
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        os.makedirs('plots', exist_ok=True)
        fig.savefig('plots/model_comparison.png')
        plt.close(fig)
    
    # Best model display
    st.subheader("Best Model for Each Disease (Based on Test Data)")
    for target in target_columns:
        if target in best_models_dict and best_models_dict[target]:
            best_model_name = best_models_dict[target]['model_name']
            best_feature_set = best_models_dict[target]['feature_set']
            best_accuracy = best_models_dict[target]['accuracy']
            st.write(f"**{target}**: {best_model_name} with an accuracy of {best_accuracy:.4f} using feature set {best_feature_set}.")
            if best_feature_set in best_features_per_threshold_dict[target]:
                train_and_save_best_model(X, target, best_features_per_threshold_dict[target][best_feature_set], best_model_name)
        else:
            st.warning(f"No valid best model available for {target}.")
    
    # Save results to CSV
    st.subheader("Saved Results")
    os.makedirs('results', exist_ok=True)
    for target in target_columns:
        if target in test_results_dict and not test_results_dict[target].empty:
            test_results_dict[target].to_csv(f'results/model_comparison_results_{target}.csv', index=False)
            st.write(f"Saved model comparison results for {target} to 'results/model_comparison_results_{target}.csv'")

if __name__ == "__main__":
    main()