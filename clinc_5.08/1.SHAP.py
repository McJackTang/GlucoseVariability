import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
    GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, confusion_matrix, \
    classification_report
import shap
import os
from pathlib import Path
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# --- 载入数据
print("Loading dataset...")
df = pd.read_csv('dataset.csv')

print(f"Data shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types and non-null values:")
print(df.info())
print("\nNumerical variables summary:")
print(df.describe())

missing_values = df.isnull().sum()
print("\nMissing values:")
print(missing_values[missing_values > 0])

# --- 预处理：用中位数补缺失
print("\nPreprocessing data...")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# --- 血糖新特征 ---
df['glucose_cv'] = df['std_glucose'] / df['mean_glucose']   # 变异系数
df['glucose_fluctuation_index'] = df['glucose_range'] / np.sqrt(df['glucose_measurement_count'])  # 波动指标
df['severe_hyperglycemia_ratio'] = df.apply(lambda x: x['hyperglycemia_ratio'] * (x['max_glucose'] > 250),
                                            axis=1)# 重度高血糖比例
df['severe_hypoglycemia_ratio'] = df.apply(lambda x: x['hypoglycemia_ratio'] * (x['min_glucose'] < 50),
                                           axis=1)  # 重度低血糖比例
# --- 年龄分组 ---
if 'age_group' not in df.columns:
    df['age_group'] = pd.cut(
        df['age_at_admission'],
        bins=[0, 18, 31, 51, 71, 120],
        labels=['0-17', '18-30', '31-50', '51-70', '71+']
    )
    age_dummies = pd.get_dummies(df['age_group'], prefix='age')
    df = pd.concat([df, age_dummies], axis=1)

# --- 分类变量处理 ---
print("\nProcessing categorical variables...")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Categorical columns found: {categorical_cols}")

high_cardinality_threshold = 50
exclude_from_encoding = ['subject_id', 'hadm_id', 'age_group', 'hospital_expire_flag']
high_cardinality_cols = []

for col in categorical_cols:
    if col in exclude_from_encoding:
        continue

    n_unique = df[col].nunique()
    print(f"  {col}: {n_unique} unique values")

    if n_unique > high_cardinality_threshold:
        high_cardinality_cols.append(col)
        print(f"  Warning: {col} is a high-cardinality variable and will be excluded from one-hot encoding")

exclude_from_encoding.extend(high_cardinality_cols)
print(f"\nThe following columns will be excluded from one-hot encoding: {exclude_from_encoding}")

for col in categorical_cols:
    if col in exclude_from_encoding:
        continue

    print(f"  One-hot encoding column '{col}'")
    # 独热编码处理离散数据
    dummies = pd.get_dummies(df[col], prefix=col).astype(float)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=[col])

# --- 把高基数列删掉 ---
for col in high_cardinality_cols:
    if col in df.columns:
        print(f"  Excluding high-cardinality column '{col}'")
        df = df.drop(columns=[col])

non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
non_numeric_cols_to_keep = ['age_group']  # We keep age_group but don't use it as a feature
non_numeric_cols_to_drop = [col for col in non_numeric_cols if col not in non_numeric_cols_to_keep]

if non_numeric_cols_to_drop:
    print(f"\nWarning: Non-numeric columns that need processing: {non_numeric_cols_to_drop}")
    for col in non_numeric_cols_to_drop:
        print(f"  Removing non-numeric column '{col}'")
        df = df.drop(columns=[col])
else:
    print("\nAll categorical variables have been properly encoded as numeric")

# ----- Check for potential data leakage variables -----
potential_leakage_vars = ['hospital_expire_flag', 'death', 'expired', 'discharge_disposition', 'mortality_indicator']
leakage_vars_found = [var for var in potential_leakage_vars if var in df.columns]
if leakage_vars_found:
    print(f"\nWARNING: Potential data leakage variables found: {leakage_vars_found}")
    print("These variables may directly reveal mortality outcome and should be excluded from features.")
    # Remove these from the dataframe to prevent accidental inclusion
    for var in leakage_vars_found:
        if var != 'mortality' and var in df.columns:  # Keep the target variable
            df = df.drop(columns=[var])

# --- 保存预处理结果 ---
df.to_csv('processed_dataset.csv', index=False)
print("Preprocessed data saved to 'processed_dataset.csv'")

# --- 准备模型数据 ---
print("\nPreparing modeling data...")
# 预测住院时长
exclude_cols_los = ['los_days', 'mortality', 'subject_id', 'hadm_id', 'age_group']
# Add any discovered leakage variables
exclude_cols_los.extend([var for var in leakage_vars_found if var in df.columns])
X_los = df.drop(columns=[col for col in exclude_cols_los if col in df.columns])
y_los = df['los_days']

# 预测死亡率
exclude_cols_mort = ['los_days', 'mortality', 'subject_id', 'hadm_id', 'age_group']
exclude_cols_mort.extend([var for var in leakage_vars_found if var in df.columns])
X_mortality = df.drop(columns=[col for col in exclude_cols_mort if col in df.columns])
y_mortality = df['mortality']

print("\nFeatures used for mortality prediction:")
print(X_mortality.columns.tolist())
# 标准化
scaler = StandardScaler()
X_los_scaled = scaler.fit_transform(X_los)
X_mortality_scaled = scaler.fit_transform(X_mortality)

# 划分训练集和测试集
X_los_train, X_los_test, y_los_train, y_los_test = train_test_split(
    X_los_scaled, y_los, test_size=0.2, random_state=42
)

X_mort_train, X_mort_test, y_mort_train, y_mort_test = train_test_split(
    X_mortality_scaled, y_mortality, test_size=0.2, random_state=42
)

# --- 训练模型-
print("\nTraining length of stay prediction models...")
# Create multiple models for comparison
los_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

los_results = {}

for name, model in los_models.items():
    print(f"Training {name} model...")
    model.fit(X_los_train, y_los_train)

    y_pred = model.predict(X_los_test)
    mse = mean_squared_error(y_los_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_los_test, y_pred)

    print(f"{name} performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.2f}")

    los_results[name] = {
        'model': model,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred
    }

# 最好的模型是谁
best_los_model_name = max(los_results, key=lambda x: los_results[x]['r2'])
best_los_model = los_results[best_los_model_name]['model']
print(f"\nBest model for length of stay prediction: {best_los_model_name}")

# 训练死亡率模型
print("\nTraining mortality prediction models...")
mortality_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

mortality_results = {}

for name, model in mortality_models.items():
    print(f"Training {name} model...")
    model.fit(X_mort_train, y_mort_train)

    y_pred = model.predict(X_mort_test)
    y_prob = model.predict_proba(X_mort_test)[:, 1]

    acc = accuracy_score(y_mort_test, y_pred)
    auc = roc_auc_score(y_mort_test, y_prob)

    print(f"{name} performance:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC: {auc:.4f}")

    #
    cm = confusion_matrix(y_mort_test, y_pred)
    print(f"  Confusion Matrix: \n{cm}")

    mortality_results[name] = {
        'model': model,
        'accuracy': acc,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_prob,
        'confusion_matrix': cm
    }

# 最好的模型
best_mort_model_name = max(mortality_results, key=lambda x: mortality_results[x]['auc'])
best_mort_model = mortality_results[best_mort_model_name]['model']
print(f"\nBest model for mortality prediction: {best_mort_model_name}")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
model_names = list(los_results.keys())
rmse_values = [los_results[name]['rmse'] for name in model_names]
r2_values = [los_results[name]['r2'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width / 2, rmse_values, width, label='RMSE')
plt.bar(x + width / 2, r2_values, width, label='R²')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Length of Stay Prediction Model Comparison')
plt.xticks(x, model_names, rotation=45)
plt.legend()

plt.subplot(1, 2, 2)
model_names = list(mortality_results.keys())
acc_values = [mortality_results[name]['accuracy'] for name in model_names]
auc_values = [mortality_results[name]['auc'] for name in model_names]

x = np.arange(len(model_names))

plt.bar(x - width / 2, acc_values, width, label='Accuracy')
plt.bar(x + width / 2, auc_values, width, label='AUC')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Mortality Prediction Model Comparison')
plt.xticks(x, model_names, rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nPerforming SHAP analysis for length of stay prediction...")
# Note: For SHAP analysis, we need original features (not standardized)
X_los_test_original = X_los.iloc[y_los_test.index]

try:
    if best_los_model_name in ['Random Forest', 'Gradient Boosting']:
        explainer = shap.TreeExplainer(best_los_model)

        sample_size = min(1000, len(X_los_test_original))
        print(f"Using {sample_size} samples for SHAP analysis...")
        X_los_sample = X_los_test_original.iloc[:sample_size]
        shap_values = explainer.shap_values(X_los_sample)

        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_los_sample, feature_names=X_los.columns, show=False)
        plt.title(f'SHAP Feature Importance for Length of Stay ({best_los_model_name})')
        plt.tight_layout()
        plt.savefig(output_dir / 'los_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

        np.save(output_dir / 'los_shap_values.npy', shap_values)

        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_df = pd.DataFrame({
            'Feature': X_los.columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)

        for i, feature in enumerate(feature_importance_df['Feature'][:3]):
            feature_idx = list(X_los.columns).index(feature)
            plt.figure(figsize=(10, 7))
            shap.dependence_plot(feature_idx, shap_values, X_los_sample,
                                 feature_names=X_los.columns, show=False)
            plt.title(f'Impact of {feature} on Length of Stay')
            plt.tight_layout()
            plt.savefig(output_dir / f'los_shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
            plt.close()
    else:
        print("Preparing background data for KernelExplainer...")
        background_sample_size = min(1000, len(X_los_train))
        background_data = X_los_train[:background_sample_size]

        kmeans_clusters = 10  # Reduced from 50 to save memory
        print(f"Using {kmeans_clusters} kmeans clusters...")

        explainer = shap.KernelExplainer(
            best_los_model.predict,
            shap.kmeans(background_data, kmeans_clusters)
        )

        sample_size = min(500, len(X_los_test_original))
        print(f"Computing SHAP values for {sample_size} samples...")
        X_los_sample = X_los_test_original.iloc[:sample_size]
        shap_values = explainer.shap_values(X_los_sample)

        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_los_sample, feature_names=X_los.columns, show=False)
        plt.title(f'SHAP Feature Importance for Length of Stay ({best_los_model_name})')
        plt.tight_layout()
        plt.savefig(output_dir / 'los_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
except Exception as e:
    print(f"Error during length of stay SHAP analysis: {str(e)}")
    print("Trying with smaller sample size...")
    try:
        if best_los_model_name in ['Random Forest', 'Gradient Boosting']:
            explainer = shap.TreeExplainer(best_los_model)
            sample_size = 100  # Very small sample size
            X_los_sample = X_los_test_original.iloc[:sample_size]
            shap_values = explainer.shap_values(X_los_sample)

            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_los_sample, feature_names=X_los.columns, show=False)
            plt.title(f'SHAP Feature Importance for Length of Stay (small sample, {best_los_model_name})')
            plt.tight_layout()
            plt.savefig(output_dir / 'los_shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e2:
        print(f"SHAP analysis with very small sample also failed: {str(e2)}")
        print("Skipping length of stay SHAP analysis")

print("\nPerforming SHAP analysis for mortality prediction...")
try:
    X_mort_test_original = X_mortality.iloc[y_mort_test.index]

    if best_mort_model_name in ['Random Forest', 'Gradient Boosting']:
        explainer = shap.TreeExplainer(best_mort_model)

        sample_size = min(1000, len(X_mort_test_original))
        print(f"Using {sample_size} samples for SHAP analysis...")
        X_mort_sample = X_mort_test_original.iloc[:sample_size]
        shap_values = explainer.shap_values(X_mort_sample)

        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1]  # Positive class SHAP values
        else:
            shap_values_class1 = shap_values

        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values_class1, X_mort_sample, feature_names=X_mortality.columns, show=False)
        plt.title(f'SHAP Feature Importance for Mortality ({best_mort_model_name})')
        plt.tight_layout()
        plt.savefig(output_dir / 'mortality_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

        np.save(output_dir / 'mortality_shap_values.npy', shap_values_class1)

        feature_importance = np.abs(shap_values_class1).mean(0)
        feature_importance_df = pd.DataFrame({
            'Feature': X_mortality.columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)

        for i, feature in enumerate(feature_importance_df['Feature'][:3]):
            feature_idx = list(X_mortality.columns).index(feature)
            plt.figure(figsize=(10, 7))
            shap.dependence_plot(feature_idx, shap_values_class1, X_mort_sample,
                                 feature_names=X_mortality.columns, show=False)
            plt.title(f'Impact of {feature} on Mortality')
            plt.tight_layout()
            plt.savefig(output_dir / f'mortality_shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
            plt.close()
    else:
        print("Preparing background data for KernelExplainer...")
        background_sample_size = min(1000, len(X_mort_train))
        background_data = X_mort_train[:background_sample_size]

        kmeans_clusters = 10  # Reduced from 50 to save memory
        print(f"Using {kmeans_clusters} kmeans clusters...")

        explainer = shap.KernelExplainer(
            lambda x: best_mort_model.predict_proba(x)[:, 1],
            shap.kmeans(background_data, kmeans_clusters)
        )

        sample_size = min(500, len(X_mort_test_original))
        print(f"Computing SHAP values for {sample_size} samples...")
        X_mort_sample = X_mort_test_original.iloc[:sample_size]
        shap_values = explainer.shap_values(X_mort_sample)

        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_mort_sample, feature_names=X_mortality.columns, show=False)
        plt.title(f'SHAP Feature Importance for Mortality ({best_mort_model_name})')
        plt.tight_layout()
        plt.savefig(output_dir / 'mortality_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
except Exception as e:
    print(f"Error during mortality SHAP analysis: {str(e)}")
    print("Trying with smaller sample size...")
    try:
        if best_mort_model_name in ['Random Forest', 'Gradient Boosting']:
            X_mort_test_original = X_mortality.iloc[y_mort_test.index]
            explainer = shap.TreeExplainer(best_mort_model)
            sample_size = 100
            X_mort_sample = X_mort_test_original.iloc[:sample_size]
            shap_values = explainer.shap_values(X_mort_sample)

            if isinstance(shap_values, list):
                shap_values_class1 = shap_values[1]
            else:
                shap_values_class1 = shap_values

            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values_class1, X_mort_sample, feature_names=X_mortality.columns, show=False)
            plt.title(f'SHAP Feature Importance for Mortality (small sample, {best_mort_model_name})')
            plt.tight_layout()
            plt.savefig(output_dir / 'mortality_shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e2:
        print(f"SHAP analysis with very small sample also failed: {str(e2)}")
        print("Skipping mortality SHAP analysis")

print("\nModel building and SHAP analysis completed!")