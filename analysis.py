import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Создаем папку для графиков, если её нет
os.makedirs('images', exist_ok=True)

# Настройка стиля: пастельные цвета, поддержка русского языка
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# === ЗАГРУЗКА ДАННЫХ ===
# Если у вас есть CSV файл с Kaggle, раскомментируйте строку ниже:
# df = pd.read_csv('data/synthetic_clinical_data.csv')

# Если файла нет, генерируем синтетические данные для теста (как в документе):
np.random.seed(42)
n = 1000
data = {
    'age': np.random.randint(18, 91, n),
    'gender': np.random.choice(['Male', 'Female'], n),
    'bmi': np.random.normal(26, 4, n),
    'systolic_bp': np.random.normal(130, 15, n),
    'diastolic_bp': np.random.normal(85, 10, n),
    'cholesterol': np.random.normal(200, 30, n),
    'glucose': np.random.exponential(100, n) + 70,
    'smoking_status': np.random.choice(['Never', 'Former', 'Current'], n, p=[0.5, 0.3, 0.2]),
    'physical_activity': np.random.choice(['Low', 'Medium', 'High'], n),
    'disease_risk': np.random.choice([0, 1], n, p=[0.6, 0.4])
}
df = pd.DataFrame(data)

# Добавляем пропуски (как в документе: bmi ~2.3%, cholesterol ~4.1%, glucose ~3.5%)
df.loc[df.sample(frac=0.023).index, 'bmi'] = np.nan
df.loc[df.sample(frac=0.041).index, 'cholesterol'] = np.nan
df.loc[df.sample(frac=0.035).index, 'glucose'] = np.nan

# Добавляем 4 дубликата
df = pd.concat([df, df.iloc[:4]], ignore_index=True)

numeric_cols = ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'glucose']
categorical_cols = ['gender', 'smoking_status', 'physical_activity', 'disease_risk']

# РИСУНОК 1. Гистограммы числовых признаков
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.histplot(df[col].dropna(), bins=30, kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(f'Распределение: {col}')
    axes[i].set_xlabel('Значение')
    axes[i].set_ylabel('Частота')
plt.suptitle('Рисунок 1. Графики распределения вещественных признаков', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('images/fig1_histograms.png', dpi=300, bbox_inches='tight')
plt.show()

# РИСУНОК 2. Столбчатые диаграммы категориальных признаков
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, col in enumerate(categorical_cols):
    sns.countplot(data=df, x=col, ax=axes[i], palette='pastel')
    axes[i].set_title(f'Распределение: {col}')
    axes[i].set_xlabel('Категория')
    axes[i].set_ylabel('Количество')
    if col not in ['gender', 'disease_risk']:
        axes[i].tick_params(axis='x', rotation=30)
plt.suptitle('Рисунок 2. Распределения категориальных признаков', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('images/fig2_categorical.png', dpi=300, bbox_inches='tight')
plt.show()

# РИСУНОК 3. Scatterplot (Seaborn)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='bmi', y='cholesterol', hue='disease_risk', 
                palette='muted', alpha=0.7)
plt.title('Рисунок 3. Зависимость холестерина от ИМТ')
plt.xlabel('Индекс массы тела (BMI)')
plt.ylabel('Уровень холестерина (мг/дл)')
plt.legend(title='Риск заболевания')
plt.savefig('images/fig3_scatter_bmi_chol.png', dpi=300, bbox_inches='tight')
plt.show()

# РИСУНОК 4. Boxplot по статусу курения (Seaborn)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='smoking_status', y='systolic_bp', palette='pastel')
plt.title('Рисунок 4. Систолическое давление по статусу курения')
plt.xlabel('Статус курения')
plt.ylabel('Систолическое давление (мм рт. ст.)')
plt.savefig('images/fig4_boxplot_smoking_bp.png', dpi=300, bbox_inches='tight')
plt.show()

# РИСУНОК 5. Violinplot по целевой переменной (Seaborn)
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='disease_risk', y='glucose', palette='muted')
plt.title('Рисунок 5. Распределение глюкозы по классам риска')
plt.xlabel('Риск заболевания (0 — низкий, 1 — высокий)')
plt.ylabel('Уровень глюкозы (мг/дл)')
plt.savefig('images/fig5_violin_glucose.png', dpi=300, bbox_inches='tight')
plt.show()

# РИСУНОК 6. Интерактивная диаграмма рассеяния (Plotly)
fig6 = px.scatter(df, x='age', y='systolic_bp', color='disease_risk',
                  title='Рисунок 6. Возраст и систолическое давление',
                  labels={'age': 'Возраст', 'systolic_bp': 'Систолическое давление', 
                          'disease_risk': 'Риск заболевания'},
                  color_discrete_sequence=px.colors.qualitative.Pastel)
fig6.write_html('images/fig6_interactive_scatter.html')
fig6.show()

# РИСУНОК 7. Интерактивная гистограмма ИМТ по полу (Plotly)
fig7 = px.histogram(df, x='bmi', color='gender', barmode='overlay',
                    title='Рисунок 7. Распределение ИМТ по полу',
                    labels={'bmi': 'Индекс массы тела (BMI)', 'gender': 'Пол'},
                    color_discrete_map={'Male': '#AEC7E8', 'Female': '#FFBB78'})
fig7.write_html('images/fig7_interactive_hist_bmi.html')
fig7.show()

# РИСУНОК 8. Тепловая карта корреляций (Seaborn)
plt.figure(figsize=(10, 8))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Рисунок 8. Тепловая карта корреляций числовых признаков')
plt.savefig('images/fig8_heatmap_corr.png', dpi=300, bbox_inches='tight')
plt.show()

# РИСУНОК 9. Тепловая карта пропусков (Seaborn)
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Рисунок 9. Тепловая карта пропущенных значений')
plt.xlabel('Признаки')
plt.ylabel('Наблюдения')
plt.savefig('images/fig9_heatmap_missing.png', dpi=300, bbox_inches='tight')
plt.show()

# РИСУНОК 10. Боксплоты всех числовых признаков (Seaborn)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.boxplot(y=df[col], ax=axes[i], color='lightgreen')
    axes[i].set_title(f'Боксплот: {col}')
    axes[i].set_ylabel('Значение')
plt.suptitle('Рисунок 10. Боксплоты числовых признаков', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('images/fig10_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# РИСУНОК 11. Сравнение исходных и зашумлённых распределений
noise_systolic = np.random.normal(0, 3, size=len(df))
df['systolic_bp_noisy'] = df['systolic_bp'] + noise_systolic
noise_chol = np.random.normal(0, 5, size=len(df))
df['cholesterol_noisy'] = df['cholesterol'] + noise_chol

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df['systolic_bp'].dropna(), bins=30, ax=axes[0], color='skyblue', label='Исходные', alpha=0.6)
sns.histplot(df['systolic_bp_noisy'].dropna(), bins=30, ax=axes[0], color='salmon', label='С шумом', alpha=0.6)
axes[0].set_title('Систолическое давление')
axes[0].set_xlabel('Давление (мм рт. ст.)')
axes[0].set_ylabel('Частота')
axes[0].legend()

sns.histplot(df['cholesterol'].dropna(), bins=30, ax=axes[1], color='skyblue', label='Исходные', alpha=0.6)
sns.histplot(df['cholesterol_noisy'].dropna(), bins=30, ax=axes[1], color='salmon', label='С шумом', alpha=0.6)
axes[1].set_title('Уровень холестерина')
axes[1].set_xlabel('Холестерин (мг/дл)')
axes[1].set_ylabel('Частота')
axes[1].legend()

plt.suptitle('Рисунок 11. Исходные и зашумлённые распределения', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('images/fig11_noisy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Все 11 графиков успешно сохранены в папку 'images/'!")
