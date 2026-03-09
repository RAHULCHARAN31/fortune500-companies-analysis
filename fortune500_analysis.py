"""
Data Analysis on Fortune 500 Companies
Author: Rahul Charan Erigirala
Description: EDA on Fortune 500 companies - revenue, industries, employees
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")

print("=" * 55)
print("  FORTUNE 500 COMPANIES - EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# ─────────────────────────────────────────────
# 1. GENERATE SAMPLE DATA (simulates scraped Fortune 500 data)
# ─────────────────────────────────────────────
np.random.seed(42)
n = 500

industries = ['Technology', 'Healthcare', 'Finance', 'Retail',
              'Energy', 'Manufacturing', 'Automotive', 'Telecom',
              'Food & Beverage', 'Insurance']

industry_col = np.random.choice(industries, n, p=[0.15, 0.12, 0.13, 0.12,
                                                    0.10, 0.10, 0.08, 0.08,
                                                    0.07, 0.05])

revenue_base = {'Technology': 80, 'Healthcare': 60, 'Finance': 90,
                'Retail': 50, 'Energy': 70, 'Manufacturing': 45,
                'Automotive': 65, 'Telecom': 55, 'Food & Beverage': 40,
                'Insurance': 75}

revenues   = np.array([revenue_base[i] + np.random.normal(0, 20) for i in industry_col])
revenues   = np.clip(revenues, 5, 500)
employees  = revenues * np.random.uniform(800, 2000, n) + np.random.normal(0, 5000, n)
employees  = np.clip(employees, 1000, 2200000).astype(int)
profits    = revenues * np.random.uniform(0.03, 0.20, n)
profits    = np.clip(profits, -10, 100)
profit_margin = (profits / revenues * 100).round(2)
rank       = np.arange(1, n + 1)

companies = [f"Company_{i}" for i in range(1, n + 1)]

df = pd.DataFrame({
    'Rank':           rank,
    'Company':        companies,
    'Industry':       industry_col,
    'Revenue_B':      revenues.round(2),
    'Profit_B':       profits.round(2),
    'Profit_Margin':  profit_margin,
    'Employees':      employees,
})

# ─────────────────────────────────────────────
# 2. DATA OVERVIEW
# ─────────────────────────────────────────────
print("\n[1] Dataset Overview:")
print(df.head(10).to_string(index=False))
print(f"\nShape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Statistics:\n{df.describe().round(2)}")

# ─────────────────────────────────────────────
# 3. DATA CLEANING
# ─────────────────────────────────────────────
print("\n[2] Data Cleaning...")
df.dropna(inplace=True)
df = df[df['Revenue_B'] > 0]
print(f"Clean dataset shape: {df.shape}")
print("No missing values found — data is clean!")

# ─────────────────────────────────────────────
# 4. EDA - INDUSTRY ANALYSIS
# ─────────────────────────────────────────────
print("\n[3] Industry Analysis...")
industry_stats = df.groupby('Industry').agg(
    Company_Count=('Company', 'count'),
    Total_Revenue=('Revenue_B', 'sum'),
    Avg_Revenue=('Revenue_B', 'mean'),
    Avg_Profit_Margin=('Profit_Margin', 'mean'),
    Total_Employees=('Employees', 'sum')
).round(2).sort_values('Total_Revenue', ascending=False)

print("\nIndustry Summary:")
print(industry_stats.to_string())

# ─────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n[4] Creating Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Fortune 500 Companies — EDA Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Top industries by total revenue
top_revenue = industry_stats['Total_Revenue'].sort_values(ascending=False)
axes[0, 0].bar(top_revenue.index, top_revenue.values, color=sns.color_palette("Blues_d", len(top_revenue)))
axes[0, 0].set_title('Total Revenue by Industry (Billion $)')
axes[0, 0].set_xlabel('Industry')
axes[0, 0].set_ylabel('Total Revenue ($B)')
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Company count per industry
count_data = industry_stats['Company_Count'].sort_values(ascending=False)
axes[0, 1].bar(count_data.index, count_data.values, color=sns.color_palette("Greens_d", len(count_data)))
axes[0, 1].set_title('Number of Companies per Industry')
axes[0, 1].set_xlabel('Industry')
axes[0, 1].set_ylabel('Count')
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Revenue distribution
axes[0, 2].hist(df['Revenue_B'], bins=30, color='steelblue', edgecolor='white')
axes[0, 2].set_title('Revenue Distribution')
axes[0, 2].set_xlabel('Revenue ($B)')
axes[0, 2].set_ylabel('Frequency')

# Plot 4: Employees vs Revenue scatter
axes[1, 0].scatter(df['Revenue_B'], df['Employees'] / 1000,
                   alpha=0.4, color='coral', edgecolors='white', s=40)
axes[1, 0].set_title('Revenue vs Employees')
axes[1, 0].set_xlabel('Revenue ($B)')
axes[1, 0].set_ylabel('Employees (thousands)')

# Plot 5: Profit Margin by Industry (box plot)
industry_order = industry_stats.index.tolist()
df_plot = df[df['Industry'].isin(industry_order)]
sns.boxplot(data=df_plot, x='Industry', y='Profit_Margin',
            order=industry_order, ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Profit Margin by Industry (%)')
axes[1, 1].set_xlabel('Industry')
axes[1, 1].set_ylabel('Profit Margin (%)')
axes[1, 1].tick_params(axis='x', rotation=45)

# Plot 6: Correlation heatmap
corr = df[['Revenue_B', 'Profit_B', 'Profit_Margin', 'Employees']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[1, 2], center=0, square=True)
axes[1, 2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('fortune500_eda.png', dpi=120, bbox_inches='tight')
plt.show()
print("[Dashboard saved as fortune500_eda.png]")

# ─────────────────────────────────────────────
# 6. KEY INSIGHTS
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("KEY INSIGHTS")
print("=" * 55)
print(f"1. Highest revenue industry : {industry_stats['Total_Revenue'].idxmax()}")
print(f"2. Most companies in        : {industry_stats['Company_Count'].idxmax()}")
print(f"3. Best avg profit margin   : {industry_stats['Avg_Profit_Margin'].idxmax()} ({industry_stats['Avg_Profit_Margin'].max():.1f}%)")
print(f"4. Most employees in        : {industry_stats['Total_Employees'].idxmax()}")
print(f"5. Average company revenue  : ${df['Revenue_B'].mean():.1f}B")
print(f"6. Median profit margin     : {df['Profit_Margin'].median():.1f}%")
top10 = df.nlargest(10, 'Revenue_B')[['Rank', 'Company', 'Industry', 'Revenue_B', 'Profit_Margin']]
print(f"\nTop 10 Companies by Revenue:\n{top10.to_string(index=False)}")
print("\n✅ Analysis complete!")
