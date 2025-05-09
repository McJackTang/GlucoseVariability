import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
import matplotlib


matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# 开始读取处理好的数据
print("Loading data...")
df = pd.read_csv('processed_dataset.csv')

print("Available columns in the dataset:")
print(df.columns.tolist())

# 定义高血糖阈值，生成二元标签
hyperglycemia_threshold = 0.3  # Hyperglycemia ratio threshold
df['hyperglycemia'] = (df['hyperglycemia_ratio'] >= hyperglycemia_threshold).astype(int)

print(f"Created hyperglycemia treatment variable (threshold: {hyperglycemia_threshold})")
print(f"Hyperglycemia group sample size: {df['hyperglycemia'].sum()}")
print(f"Normal glycemia group sample size: {len(df) - df['hyperglycemia'].sum()}")


if 'gender' in df.columns:
    gender_cols = ['gender']
else:
    # Look for gender_F, gender_M, etc.
    gender_cols = [col for col in df.columns if col.startswith('gender_')]
    print(f"Using one-hot encoded gender columns: {gender_cols}")

# 基础协变量：年龄、性别和测糖次数
basic_covariates = ['age_at_admission'] + gender_cols + ['glucose_measurement_count']

# 年龄分组字段一并放进协变量
age_group_cols = [col for col in df.columns
                  if col.startswith('age_')
                  and col != 'age_group'  # Exclude the categorical column
                  and col != 'age_at_admission']  # This is already in basic_covariates

covariates = basic_covariates.copy()
if age_group_cols:
    # Only add if they appear to be one-hot encoded columns (check if numeric)
    valid_age_cols = []
    for col in age_group_cols:
        # Check if the column contains numeric data
        if df[col].dtype in ['int64', 'float64', 'bool']:
            valid_age_cols.append(col)

    covariates.extend(valid_age_cols)
    print(f"Added one-hot encoded age group columns to covariates: {valid_age_cols}")

#分析的结果：住院天数和死亡率
outcome_cols = ['los_days', 'mortality']

# 计算倾向得分
print("Calculating propensity scores...")
X = df[covariates]
y = df['hyperglycemia']

# 标准化啦
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ps_model = LogisticRegression(max_iter=1000, random_state=42)
ps_model.fit(X_scaled, y)

# 保存带倾向得分的数据
df['propensity_score'] = ps_model.predict_proba(X_scaled)[:, 1]

# Save data with propensity scores
df_before_matching = df.copy()
df_before_matching.to_csv(output_dir / 'data_with_propensity_scores.csv', index=False)

# 匹配前倾向得分分布图
plt.figure(figsize=(10, 6))
for value, group in df.groupby('hyperglycemia'):
    label = "Hyperglycemia" if value == 1 else "Normal Glycemia"
    sns.kdeplot(group['propensity_score'], label=label, fill=True, alpha=0.5)

plt.title('Propensity Score Distribution Before Matching')
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.legend()
plt.savefig(output_dir / 'propensity_score_before_matching.png', dpi=300, bbox_inches='tight')
plt.close()

print("Performing propensity score matching...")


def match_samples(df, treatment_col, propensity_score_col, caliper=0.05, ratio=1):
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()

    print(f"Treatment group sample size: {len(treated)}")
    print(f"Control group sample size: {len(control)}")

    treated['original_index'] = treated.index
    control['original_index'] = control.index

    matched_indices = []

    for idx, treated_row in treated.iterrows():
        control['ps_diff'] = abs(control[propensity_score_col] - treated_row[propensity_score_col])
        potential_matches = control[control['ps_diff'] <= caliper].sort_values('ps_diff')

        if not potential_matches.empty:
            matches = potential_matches.head(ratio)

            all_matches = [idx] + matches['original_index'].tolist()
            matched_indices.extend(all_matches)

            control = control.drop(matches.index)

    matched_df = df.loc[matched_indices].copy()

    print(f"Sample size after matching: {len(matched_df)}")

    return matched_df


matched_df = match_samples(df, 'hyperglycemia', 'propensity_score', caliper=0.05)
matched_df.to_csv(output_dir / 'matched_data.csv', index=False)

print("Evaluating covariate balance...")

# 评估匹配前后协变量平衡
def assess_balance(df_before, df_after, treatment_col, covariates):
    results = []

    for dataset_name, dataset in [('Before Matching', df_before), ('After Matching', df_after)]:
        treated = dataset[dataset[treatment_col] == 1]
        control = dataset[dataset[treatment_col] == 0]

        for var in covariates:
            treated_mean = treated[var].mean()
            control_mean = control[var].mean()

            treated_sd = treated[var].std()
            control_sd = control[var].std()

            pooled_sd = np.sqrt((treated_sd ** 2 + control_sd ** 2) / 2)

            if pooled_sd == 0:
                std_mean_diff = 0
            else:
                std_mean_diff = (treated_mean - control_mean) / pooled_sd

            results.append({
                'Dataset': dataset_name,
                'Variable': var,
                'Treated_Mean': treated_mean,
                'Control_Mean': control_mean,
                'Std_Mean_Diff': std_mean_diff,
                'Abs_Std_Mean_Diff': abs(std_mean_diff)
            })

    balance_df = pd.DataFrame(results)
    return balance_df



balance_df = assess_balance(df_before_matching, matched_df, 'hyperglycemia', covariates)
balance_df.to_csv(output_dir / 'covariate_balance.csv', index=False)

# 匹配后倾向得分分布图
plt.figure(figsize=(10, 6))
for value, group in matched_df.groupby('hyperglycemia'):
    label = "Hyperglycemia" if value == 1 else "Normal Glycemia"
    sns.kdeplot(group['propensity_score'], label=label, fill=True, alpha=0.5)

plt.title('Propensity Score Distribution After Matching')
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.legend()
plt.savefig(output_dir / 'propensity_score_after_matching.png', dpi=300, bbox_inches='tight')
plt.close()

# Balance plot - Fixed to handle NaN values and ensure proper data types
plt.figure(figsize=(12, 10))

before_data = balance_df[balance_df['Dataset'] == 'Before Matching'].copy()
after_data = balance_df[balance_df['Dataset'] == 'After Matching'].copy()

# Ensure both dataframes have the same variables in the same order
before_data = before_data.sort_values('Variable')
after_data = after_data.sort_values('Variable')

# Create plot data, filtering out any rows with NaN values
plot_data = pd.DataFrame({
    'Variable': before_data['Variable'].values,
    'Before': before_data['Std_Mean_Diff'].values,
    'After': after_data['Std_Mean_Diff'].values
})

# Remove any rows with NaN values
plot_data = plot_data.dropna()

# Sort by absolute value before matching
plot_data = plot_data.iloc[np.argsort(abs(plot_data['Before']))[::-1]]

plt.figure(figsize=(10, 8))

# Ensure variables are strings and data are floats
variables = plot_data['Variable'].astype(str).tolist()
before_values = plot_data['Before'].astype(float).tolist()
after_values = plot_data['After'].astype(float).tolist()

# Create the plot using lists instead of pandas Series
for i in range(len(variables)):
    plt.hlines(y=i, xmin=before_values[i], xmax=after_values[i],
               color='gray', alpha=0.5, linewidth=1)

plt.scatter(before_values, range(len(variables)), color='red', label='Before Matching', s=50)
plt.scatter(after_values, range(len(variables)), color='blue', label='After Matching', s=50)

# Set y-axis labels
plt.yticks(range(len(variables)), variables)

threshold = 0.1
plt.axvline(x=threshold, color='gray', linestyle='--')
plt.axvline(x=-threshold, color='gray', linestyle='--')
plt.axvline(x=0, color='black', linestyle='-')

plt.xlabel('Standardized Mean Difference')
plt.ylabel('Variable')
plt.title('Covariate Balance Before and After Matching')
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / 'balance_before_after.png', dpi=300, bbox_inches='tight')
plt.close()

# 估计治疗效果：住院天数和死亡率
print("Estimating treatment effects...")


def estimate_treatment_effects(df, treatment_col, outcome_cols, covariates):
    results = []

    for outcome in outcome_cols:
        print(f"Analyzing outcome variable: {outcome}")
        treated_mean = df[df[treatment_col] == 1][outcome].mean()
        control_mean = df[df[treatment_col] == 0][outcome].mean()
        mean_diff = treated_mean - control_mean

        formula = f"{outcome} ~ {treatment_col} + " + " + ".join(covariates)

        try:
            if outcome == 'mortality':
                model = smf.logit(formula, data=df).fit(disp=0)
                coef = model.params[treatment_col]
                std_err = model.bse[treatment_col]
                p_value = model.pvalues[treatment_col]
                odds_ratio = np.exp(coef)

                results.append({
                    'Outcome': outcome,
                    'Treatment': treatment_col,
                    'Unadjusted_Diff': mean_diff,
                    'Treated_Mean': treated_mean,
                    'Control_Mean': control_mean,
                    'Coef': coef,
                    'Std_Error': std_err,
                    'P_Value': p_value,
                    'Odds_Ratio': odds_ratio,
                    'OR_95CI_Lower': np.exp(coef - 1.96 * std_err),
                    'OR_95CI_Upper': np.exp(coef + 1.96 * std_err)
                })
            else:
                model = smf.ols(formula, data=df).fit()
                coef = model.params[treatment_col]
                std_err = model.bse[treatment_col]
                p_value = model.pvalues[treatment_col]

                results.append({
                    'Outcome': outcome,
                    'Treatment': treatment_col,
                    'Unadjusted_Diff': mean_diff,
                    'Treated_Mean': treated_mean,
                    'Control_Mean': control_mean,
                    'Coef': coef,
                    'Std_Error': std_err,
                    'P_Value': p_value,
                    '95CI_Lower': coef - 1.96 * std_err,
                    '95CI_Upper': coef + 1.96 * std_err
                })
        except Exception as e:
            print(f"Regression analysis failed: {e}")

    return pd.DataFrame(results)


treatment_effects = estimate_treatment_effects(matched_df, 'hyperglycemia', outcome_cols, basic_covariates[:2])
treatment_effects.to_csv(output_dir / 'treatment_effects.csv', index=False)

plt.figure(figsize=(12, 6))

continuous_outcomes = treatment_effects[treatment_effects['Outcome'] == 'los_days']
binary_outcomes = treatment_effects[treatment_effects['Outcome'] == 'mortality']

if not continuous_outcomes.empty:
    plt.subplot(1, 2, 1)
    outcome = continuous_outcomes.iloc[0]

    plt.errorbar(
        x=[outcome['Coef']],
        y=[0],
        xerr=[[outcome['Coef'] - outcome['95CI_Lower']], [outcome['95CI_Upper'] - outcome['Coef']]],
        fmt='o',
        capsize=5,
        color='blue'
    )

    plt.axvline(x=0, color='gray', linestyle='--')

    plt.yticks([])
    plt.xlabel('Adjusted Effect (Days)')
    plt.title('Effect of Hyperglycemia on Length of Stay')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

if not binary_outcomes.empty:
    plt.subplot(1, 2, 2)
    outcome = binary_outcomes.iloc[0]

    plt.errorbar(
        x=[outcome['Odds_Ratio']],
        y=[0],
        xerr=[[outcome['Odds_Ratio'] - outcome['OR_95CI_Lower']], [outcome['OR_95CI_Upper'] - outcome['Odds_Ratio']]],
        fmt='o',
        capsize=5,
        color='red'
    )

    plt.axvline(x=1, color='gray', linestyle='--')

    plt.yticks([])
    plt.xlabel('Odds Ratio')
    plt.title('Effect of Hyperglycemia on Mortality')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xscale('log')  # Use log scale

plt.tight_layout()
plt.savefig(output_dir / 'treatment_effects_plot.png', dpi=300, bbox_inches='tight')
plt.close()

if 'age_group' in df.columns:
    print("Performing stratified analysis by age group...")

    stratified_results = []

    for age_group in df['age_group'].unique():
        print(f"Analyzing age group: {age_group}")

        age_data = matched_df[matched_df['age_group'] == age_group]

        if len(age_data) < 50:
            print(f"  Insufficient sample size, skipping analysis (n={len(age_data)})")
            continue

        age_effects = estimate_treatment_effects(age_data, 'hyperglycemia', outcome_cols, basic_covariates[:2])
        age_effects['Age_Group'] = age_group
        age_effects['Sample_Size'] = len(age_data)

        stratified_results.append(age_effects)

    if stratified_results:
        stratified_df = pd.concat(stratified_results, ignore_index=True)
        stratified_df.to_csv(output_dir / 'stratified_treatment_effects.csv', index=False)

        for outcome in outcome_cols:
            outcome_data = stratified_df[stratified_df['Outcome'] == outcome]

            if len(outcome_data) == 0:
                continue

            fig, ax = plt.subplots(figsize=(12, 6))

            age_order = ['0-17', '18-30', '31-50', '51-70', '71+']
            outcome_data['Age_Group'] = pd.Categorical(
                outcome_data['Age_Group'],
                categories=age_order,
                ordered=True
            )
            outcome_data = outcome_data.sort_values('Age_Group')

            if outcome == 'mortality':
                y_pos = range(len(outcome_data))
                ax.errorbar(
                    x=outcome_data['Odds_Ratio'],
                    y=y_pos,
                    xerr=[
                        outcome_data['Odds_Ratio'] - outcome_data['OR_95CI_Lower'],
                        outcome_data['OR_95CI_Upper'] - outcome_data['Odds_Ratio']
                    ],
                    fmt='o',
                    capsize=5
                )

                for i, row in enumerate(outcome_data.itertuples()):
                    ax.text(
                        max(outcome_data['OR_95CI_Upper']) * 1.1,
                        i,
                        f"n={row.Sample_Size}",
                        va='center'
                    )

                ax.axvline(x=1, color='gray', linestyle='--')

                ax.set_yticks(y_pos)
                ax.set_yticklabels(outcome_data['Age_Group'])
                ax.set_xlabel('Odds Ratio')
                ax.set_xscale('log')  # Use log scale
                ax.set_title('Effect of Hyperglycemia on Mortality by Age Group')

            else:
                y_pos = range(len(outcome_data))
                ax.errorbar(
                    x=outcome_data['Coef'],
                    y=y_pos,
                    xerr=[
                        outcome_data['Coef'] - outcome_data['95CI_Lower'],
                        outcome_data['95CI_Upper'] - outcome_data['Coef']
                    ],
                    fmt='o',
                    capsize=5
                )

                for i, row in enumerate(outcome_data.itertuples()):
                    ax.text(
                        max(outcome_data['95CI_Upper']) * 1.1,
                        i,
                        f"n={row.Sample_Size}",
                        va='center'
                    )

                ax.axvline(x=0, color='gray', linestyle='--')

                ax.set_yticks(y_pos)
                ax.set_yticklabels(outcome_data['Age_Group'])
                ax.set_xlabel('Adjusted Effect')
                ax.set_title(
                    f'Effect of Hyperglycemia on {"Length of Stay" if outcome == "los_days" else "Mortality"} by Age Group')

            ax.grid(axis='x', linestyle='--', alpha=0.7)
            fig.tight_layout()
            fig.savefig(output_dir / f'stratified_{outcome}_by_age.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

print("\nPropensity score matching analysis completed!")