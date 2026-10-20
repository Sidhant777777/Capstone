# Part 2: Patterns & Charts (EDA)
# Project 12 - Public Transit Bus Safety

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt

df = pd.read_csv('bus_accidents_clean.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# -------------------------------------------------------
# Chart: Daily accident count over time
# -------------------------------------------------------

daily = df.groupby('Date').size().reset_index(name='Accident_Count')

plt.figure(figsize=(12, 4))
plt.plot(daily['Date'], daily['Accident_Count'], color='steelblue', linewidth=0.8)
plt.title('Daily Bus Accident Count (UK, 2005-2016)')
plt.xlabel('Date')
plt.ylabel('Number of Accidents')
plt.tight_layout()
plt.savefig('daily_accident_trend.png', dpi=150)
plt.close()
print("Chart saved: daily_accident_trend.png")

# -------------------------------------------------------
# 2x2 Contingency Table: Summer x Skidding
# -------------------------------------------------------

df['Is_Summer'] = df['Season'].apply(lambda x: 'Summer' if x == 'Summer' else 'Non-Summer')

# Skidding occurred = row has an actual skid/overturn value (not NaN, not 'Data missing')
real_skid = ['Skidded', 'Skidded and overturned', 'Overturned', 'Jackknifed', 'Jackknifed and overturned']
df['Skidded'] = df['Skidding_and_Overturning'].apply(lambda x: 'Yes' if x in real_skid else 'No')

table = pd.crosstab(df['Is_Summer'], df['Skidded'])
print("\n2x2 Contingency Table: Season vs Skidding Occurrence")
print(table.to_string())
print(f"\nRow totals: {table.sum(axis=1).to_dict()}")
print(f"Column totals: {table.sum(axis=0).to_dict()}")

# -------------------------------------------------------
# Chi-Square Test (manual — no scipy needed)
# -------------------------------------------------------

observed = table.values.astype(float)
row_totals   = observed.sum(axis=1, keepdims=True)
col_totals   = observed.sum(axis=0, keepdims=True)
grand_total  = observed.sum()

expected = (row_totals * col_totals) / grand_total
chi2     = float(np.sum((observed - expected) ** 2 / expected))
dof      = (observed.shape[0] - 1) * (observed.shape[1] - 1)

# p-value: for dof=1, chi2 ~ chi-squared(1)
# Use: p = 2*(1 - Phi(sqrt(chi2)))  where Phi is standard normal CDF
p = 2 * (1 - 0.5 * (1 + erf(sqrt(chi2) / sqrt(2))))

print(f"\nChi-Square Statistic : {chi2:.4f}")
print(f"Degrees of Freedom   : {dof}")
print(f"p-value              : {p:.4f}")

print("\nExpected Frequencies:")
print(pd.DataFrame(expected.round(4), index=table.index, columns=table.columns).to_string())

if p < 0.05:
    print("\nResult: Statistically significant (p < 0.05).")
    print("Summer season IS statistically linked to higher skidding frequency in bus accidents.")
else:
    print("\nResult: Not statistically significant (p >= 0.05).")
    print("Summer season is NOT significantly linked to skidding in bus accidents.")
