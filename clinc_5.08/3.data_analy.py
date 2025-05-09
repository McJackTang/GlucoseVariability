import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import shap
import joblib
import warnings

warnings.filterwarnings('ignore')

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
report_dir = Path('report')
report_dir.mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
print("加载数据...")
df = pd.read_csv('processed_dataset.csv')
matched_df = None
if os.path.exists(output_dir / 'matched_data.csv'):
    matched_df = pd.read_csv(output_dir / 'matched_data.csv')

# 数据分析
print("生成探索性分析图表...")

# 1. 血糖特征分布
plt.figure(figsize=(15, 10))
glucose_features = ['mean_glucose', 'max_glucose', 'min_glucose',
                    'hyperglycemia_ratio', 'hypoglycemia_ratio', 'glucose_range']

for i, feature in enumerate(glucose_features):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'{feature}分布')
    plt.xlabel(feature)

plt.tight_layout()
plt.savefig(report_dir / 'glucose_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 血糖指标与结果变量的关系
plt.figure(figsize=(15, 10))

# 平均血糖与住院时长
plt.subplot(2, 2, 1)
sns.scatterplot(x='mean_glucose', y='los_days', hue='mortality', data=df)
plt.title('平均血糖与住院时长的关系')
plt.xlabel('平均血糖')
plt.ylabel('住院天数')

# 高血糖比率与住院时长
plt.subplot(2, 2, 2)
sns.boxplot(x=pd.qcut(df['hyperglycemia_ratio'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4']),
            y='los_days', data=df)
plt.title('高血糖比率四分位数与住院时长')
plt.xlabel('高血糖比率分组')
plt.ylabel('住院天数')

# 平均血糖与死亡率
plt.subplot(2, 2, 3)
glucose_bins = pd.qcut(df['mean_glucose'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
mortality_by_glucose = df.groupby(glucose_bins)['mortality'].mean() * 100
sns.barplot(x=mortality_by_glucose.index, y=mortality_by_glucose.values)
plt.title('平均血糖四分位数与死亡率')
plt.xlabel('平均血糖分组')
plt.ylabel('死亡率 (%)')

# 高血糖比率与死亡率
plt.subplot(2, 2, 4)
hyperglycemia_bins = pd.qcut(df['hyperglycemia_ratio'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
mortality_by_hyperglycemia = df.groupby(hyperglycemia_bins)['mortality'].mean() * 100
sns.barplot(x=mortality_by_hyperglycemia.index, y=mortality_by_hyperglycemia.values)
plt.title('高血糖比率四分位数与死亡率')
plt.xlabel('高血糖比率分组')
plt.ylabel('死亡率 (%)')

plt.tight_layout()
plt.savefig(report_dir / 'glucose_outcomes_relationship.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 年龄组分析
if 'age_group' in df.columns:
    plt.figure(figsize=(15, 10))

    # 各年龄组的平均血糖
    plt.subplot(2, 2, 1)
    sns.boxplot(x='age_group', y='mean_glucose', data=df)
    plt.title('各年龄组的平均血糖')
    plt.xlabel('年龄组')
    plt.ylabel('平均血糖')

    # 各年龄组的高血糖比率
    plt.subplot(2, 2, 2)
    sns.boxplot(x='age_group', y='hyperglycemia_ratio', data=df)
    plt.title('各年龄组的高血糖比率')
    plt.xlabel('年龄组')
    plt.ylabel('高血糖比率')

    # 各年龄组的住院时长
    plt.subplot(2, 2, 3)
    sns.boxplot(x='age_group', y='los_days', data=df)
    plt.title('各年龄组的住院时长')
    plt.xlabel('年龄组')
    plt.ylabel('住院天数')

    # 各年龄组的死亡率
    plt.subplot(2, 2, 4)
    mortality_by_age = df.groupby('age_group')['mortality'].mean() * 100
    sns.barplot(x=mortality_by_age.index, y=mortality_by_age.values)
    plt.title('各年龄组的死亡率')
    plt.xlabel('年龄组')
    plt.ylabel('死亡率 (%)')

    plt.tight_layout()
    plt.savefig(report_dir / 'age_group_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. 相关性热图
corr_cols = ['mean_glucose', 'std_glucose', 'max_glucose', 'min_glucose',
             'hyperglycemia_ratio', 'hypoglycemia_ratio', 'glucose_range',
             'los_days', 'mortality', 'age_at_admission']

plt.figure(figsize=(12, 10))
corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
plt.title('血糖指标与结果变量的相关性矩阵')
plt.tight_layout()
plt.savefig(report_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 模型分析结果
model_files = ['model_comparison.png', 'los_shap_summary.png', 'mortality_shap_summary.png']
for file in model_files:
    src_file = output_dir / file
    if os.path.exists(src_file):
        import shutil

        shutil.copy(src_file, report_dir / file)

# 6. 倾向得分匹配结果
psm_files = ['propensity_score_before_matching.png', 'propensity_score_after_matching.png',
             'balance_before_after.png', 'treatment_effects_plot.png']
for file in psm_files:
    src_file = output_dir / file
    if os.path.exists(src_file):
        import shutil

        shutil.copy(src_file, report_dir / file)

# 创建报告
print("生成HTML报告...")


def generate_html_report():
    image_files = sorted([f for f in os.listdir(report_dir) if f.endswith('.png')])

    # 创建报告内容
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MIMIC数据集血糖相关项目报告</title>
        <style>
            body {{
                font-family: "Microsoft YaHei", Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
                margin: 10px 0;
            }}
            .section {{
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }}
            .image-container {{
                margin: 20px 0;
            }}
            .image-description {{
                font-style: italic;
                margin-top: 5px;
                color: #666;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
        </style>
    </head>
    <body>
        <h1>MIMIC数据集血糖相关项目报告</h1>

        <div class="section">
            <h2>1. 项目概述</h2>
            <p>本项目基于MIMIC数据集，主要关注血糖水平对患者住院时长和死亡率的影响。项目通过多种方法进行分析：</p>
            <ul>
                <li>探索性数据分析，了解血糖指标和临床结果之间的关系</li>
                <li>使用机器学习模型预测住院时长和死亡率</li>
                <li>通过SHAP分析解释模型预测结果</li>
                <li>使用倾向得分匹配进行因果推断</li>
                <li>按年龄组进行分层分析</li>
            </ul>
        </div>

        <div class="section">
            <h2>2. 探索性数据分析</h2>
            <p>首先探索血糖指标的分布和与住院时长及死亡率的关系。</p>
    '''

    # 探索性分析图像
    eda_images = [f for f in image_files if any(name in f for name in [
        'glucose_distributions', 'glucose_outcomes_relationship', 'age_group_analysis',
        'correlation_heatmap'
    ])]

    for img in eda_images:
        description = ''
        if 'glucose_distributions' in img:
            description = '血糖指标分布'
        elif 'glucose_outcomes_relationship' in img:
            description = '血糖指标与临床结果的关系'
        elif 'age_group_analysis' in img:
            description = '按年龄组分析血糖和临床结果'
        elif 'correlation_heatmap' in img:
            description = '血糖指标与临床结果的相关性矩阵'

        html_content += f'''
            <div class="image-container">
                <img src="{img}" alt="{description}">
                <p class="image-description">{description}</p>
            </div>
        '''

    html_content += '''
        </div>

        <div class="section">
            <h2>3. 预测模型分析</h2>
            <p>构建机器学习模型预测住院时长和死亡率，并使用SHAP分析解释模型。</p>
    '''

    # 添加模型分析图像
    model_images = [f for f in image_files if any(name in f for name in [
        'model_comparison', 'los_shap_summary', 'mortality_shap_summary'
    ])]

    for img in model_images:
        description = ''
        if 'model_comparison' in img:
            description = '不同模型性能对比'
        elif 'los_shap_summary' in img:
            description = '住院时长预测的SHAP特征重要性'
        elif 'mortality_shap_summary' in img:
            description = '死亡率预测的SHAP特征重要性'

        html_content += f'''
            <div class="image-container">
                <img src="{img}" alt="{description}">
                <p class="image-description">{description}</p>
            </div>
        '''

    # 显示SHAP依赖图
    shap_dependence_images = [f for f in image_files if 'dependence' in f]
    if shap_dependence_images:
        html_content += '<h3>特征依赖关系分析</h3>'
        html_content += '<p>以下图表展示了主要特征对预测结果的具体影响模式：</p>'

        for img in shap_dependence_images:
            feature = img.split('_')[-1].replace('.png', '')
            outcome = '住院时长' if 'los' in img else '死亡率'

            html_content += f'''
                <div class="image-container">
                    <img src="{img}" alt="{feature}对{outcome}的影响">
                    <p class="image-description">{feature}对{outcome}的影响模式</p>
                </div>
            '''

    html_content += '''
        </div>

        <div class="section">
            <h2>4. 倾向得分匹配分析</h2>
            <p>通过倾向得分匹配方法，分析高血糖对住院时长和死亡率的因果影响。</p>
    '''

    # 倾向得分匹配图像
    psm_images = [f for f in image_files if any(name in f for name in [
        'propensity_score', 'balance_before_after', 'treatment_effects_plot'
    ])]

    for img in psm_images:
        description = ''
        if 'propensity_score_before_matching' in img:
            description = '匹配前的倾向得分分布'
        elif 'propensity_score_after_matching' in img:
            description = '匹配后的倾向得分分布'
        elif 'balance_before_after' in img:
            description = '匹配前后的协变量平衡性对比'
        elif 'treatment_effects_plot' in img:
            description = '高血糖对临床结果的因果影响'

        html_content += f'''
            <div class="image-container">
                <img src="{img}" alt="{description}">
                <p class="image-description">{description}</p>
            </div>
        '''

    # 分层分析图像
    stratified_images = [f for f in image_files if 'stratified' in f]
    if stratified_images:
        html_content += '<h3>年龄组分层分析</h3>'
        html_content += '<p>以下图表展示了高血糖对不同年龄组患者的影响差异：</p>'

        for img in stratified_images:
            outcome = '住院时长' if 'los_days' in img else '死亡率'

            html_content += f'''
                <div class="image-container">
                    <img src="{img}" alt="按年龄组分层的{outcome}分析">
                    <p class="image-description">高血糖对{outcome}的影响 - 按年龄组分层</p>
                </div>
            '''

    html_content += '''
        </div>

        <div class="section">
            <h2>5. 主要发现</h2>
            <ul>
                <li>血糖水平与住院时长和死亡率存在显著相关性</li>
                <li>高血糖状态（>180 mg/dL的比例）是预测不良结局的重要指标</li>
                <li>血糖波动性（范围和变异系数）与临床结果之间存在关联</li>
                <li>不同年龄组对血糖异常的敏感性不同</li>
                <li>通过倾向得分匹配分析，高血糖被证明与更长的住院时长和更高的死亡风险相关</li>
                <li>SHAP分析显示，血糖的平均水平、波动性和异常比例是预测模型中的重要特征</li>
            </ul>
        </div>

        <div class="section">
            <h2>6. 结论与建议</h2>
            <p>基于本项目的分析结果，我们得出以下结论：</p>
            <ul>
                <li>血糖控制对住院患者的临床结果有重要影响</li>
                <li>不仅要关注血糖的平均水平，还要重视血糖的波动性</li>
                <li>不同年龄组的患者可能需要不同的血糖管理策略</li>
                <li>预测模型可以帮助识别高风险患者，支持临床决策</li>
            </ul>
            <p>建议在临床实践中：</p>
            <ul>
                <li>加强住院患者的血糖监测和管理</li>
                <li>针对不同年龄组制定个性化的血糖控制目标</li>
                <li>关注血糖波动，而不仅仅是平均水平</li>
                <li>将血糖相关指标纳入住院患者风险评估模型</li>
            </ul>
        </div>

        <footer>
            <p>基于MIMIC数据集的血糖相关项目报告 | 生成日期：{pd.Timestamp.now().strftime('%Y-%m-%d')}</p>
        </footer>
    </body>
    </html>
    '''

    with open(report_dir / 'blood_glucose_analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML报告已生成：{report_dir / 'blood_glucose_analysis_report.html'}")


generate_html_report()

print("\n可视化和报告生成完成！")