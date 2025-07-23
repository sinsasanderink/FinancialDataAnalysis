# 📊 Financial Data Analysis & Alternative Credit Data Insights

## Part 1: IVV ETF Replication Using Technical Indicators and MLP

This project replicates IVV ETF trading behavior using technical indicators and machine learning. Built with `pandas-ta-openbb` (NumPy 2 compatible), it includes full data preprocessing, technical indicator extraction, classification target generation (`Gamma`), model training using MLP, cross-validation, visualizations, and feature correlation analysis.

### ✅ Highlights
- Historical IVV data from **2009-12-12 to 2020-01-01**
- 275+ technical indicators from `pandas-ta-openbb`
- Custom classification target (`Gamma`)
- 10-fold cross-validation using `MLPClassifier`
- Robust NaN handling and feature normalization
- Visualization of price and target dynamics
- Correlation analysis to identify top predictive indicators

### 📈 Results Summary
- 🔢 Observations: 2529 days
- 🧠 Model: Multi-Layer Perceptron (MLP)
- 🎯 Accuracy: **Avg: 79.3%**, Best Fold: **83.8%**
- 🔍 Most Predictive Feature: `SMCbp_14_50_20_5` (corr ≈ 0.55)

### 📂 Dependencies
- `pandas-ta-openbb`
- `yfinance`, `numpy`, `scikit-learn`, `matplotlib`

### 📌 Replication Steps
1. Download and clean IVV data
2. Add technical indicators with fallback logic
3. Define target: `Gamma(t) = 1` if `Open(t) > Open(t−1)`, else `-1`
4. Drop high-NaN columns, clean rows, forward/backward fill
5. Normalize features using `MinMaxScaler`
6. Train MLP with 10-fold CV, record accuracies
7. Visualize price and target behavior
8. Analyze feature correlation with target

---

## Part 2: Unlocking the Power of Credit Data 🔐

This section explores the potential of credit data as a rich source of alternative data for credit risk modeling and financial insights.

### 💡 Why Credit Data?
- Captures financial behavior & repayment reliability
- Supports risk assessment and financial inclusion
- Enhances predictive modeling with structured monthly updates

### 🗂️ Data Sources
- **Credit Bureaus**: Experian, Equifax, TransUnion
- **P2P Lending Platforms**: Granular loan-level data
- **Open Banking APIs**: Real-time transactional access
- **Banks & Institutions**: Aggregated/anonymized credit info

### 🧾 Data Types
- Repayment history (defaults, days past due)
- Balances and utilization
- Loan application metadata
- Transaction-level financial behavior

### ⚙️ Feature Engineering + Aggregation Example

```python
df['is_delinquent'] = df['days_past_due'] > 30
df['balance_log'] = np.log1p(df['balance'])
df['month'] = df['report_date'].dt.to_period('M')
user_month = df.groupby(['user_id','month']).agg({
  'balance': 'mean',
  'is_delinquent': 'max',
  'credit_limit': 'first'
}).reset_index()
```

### 📊 Exploratory Data Analysis (EDA)
- Balance distribution (log-normal)
- Time series trends for delinquency
- Scatter plot: balance vs. credit limit
- User-level credit behavior across months

### 🧪 Advanced Analysis Opportunities
- Predictive modeling with delinquency as target
- Fairness audits on bias and explainability
- Incorporation of telecom or transactional data
- Anonymized data simulations for research use

---

## 🔍 Recent Research Insights
- **📉 Bias Reduction**: Lee & Yang (2024) — [PDF](https://data.mlr.press/assets/pdf/v02-2.pdf)
- **📱 Telecom Signals**: Óskarsdóttir et al. (2020) — [arXiv](https://arxiv.org/abs/2002.09931)
- **🧠 Explainable AI**: Demajo et al. (2020) — [arXiv](https://arxiv.org/abs/2012.03749)
- **📚 Alternative Data Landscape**: Sun et al. (2024) — [Springer](https://jfin-swufe.springeropen.com/articles/10.1186/s40854-024-00652-0)

---

## ✅ Ethical and Regulatory Considerations
- **Privacy**: Anonymization of personal data
- **Fairness**: Avoiding model bias by group
- **Transparency**: Explainable models required by law
- **Consent**: Clear opt-in for credit data usage
- **Compliance**: GDPR, CPRA, and other frameworks

---

## 🚀 Future Directions
- Incorporate alternative signals (e.g. telco, utility)
- Develop interpretable and fair ML credit models
- Explore reinforcement learning for credit risk
- Expand to real-time credit risk monitoring

---

## 📎 Acknowledgements
This project combines financial data science techniques from market replication modeling with next-generation credit data analysis for inclusive, data-driven finance.
