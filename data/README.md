# SmartBank AI — Dataset Documentation

## Overview

This directory contains synthetic datasets generated for model training
and demonstration. **No real personal or financial data is used.**

---

## Files

### `credit_applications.csv`
**5,000 rows** of synthetic loan/credit applications.

| Column | Type | Description |
|---|---|---|
| age | int | Applicant age (21–65) |
| annual_income | float | Annual income in INR |
| employment_type | str | Salaried, Self-Employed, Business, or Freelance |
| employment_stability_years | float | Years at current employment (0–30) |
| credit_score | int | Credit score (300–900) |
| existing_loans_count | int | Number of existing active loans (0–5) |
| existing_monthly_debt | float | Total monthly debt payments in INR |
| requested_loan_amount | float | Amount of loan requested in INR |
| loan_tenure_months | int | Desired loan duration in months |
| dependents | int | Number of dependents (0–5) |
| savings_balance | float | Current savings balance in INR |
| risk_label | str | Target: LOW, MEDIUM, or HIGH risk |
| approved | int | Target: 1 = approved, 0 = rejected |

**Target generation**: Risk labels are computed from a weighted score
combining credit score (30%), debt burden (20%), DTI ratio (15%),
LTI ratio (15%), employment stability (10%), and savings ratio (10%),
with Gaussian noise added to prevent trivial prediction.

**Class distribution** (approximate): LOW ~40%, MEDIUM ~35%, HIGH ~25%

---

### `transactions.csv`
**10,000 rows** of synthetic banking transactions.

| Column | Type | Description |
|---|---|---|
| transaction_id | str | Unique transaction ID (TXN000000) |
| customer_id | str | Customer ID (CUST0001–CUST0500) |
| amount | float | Transaction amount in INR |
| transaction_type | str | Transfer, Payment, Withdrawal, or Purchase |
| hour_of_day | int | Hour (0–23) |
| day_of_week | int | Day (0=Mon, 6=Sun) |
| location_change | int | 1 if transaction from unusual location |
| new_device | int | 1 if from a new/unrecognized device |
| transaction_frequency_24h | int | Number of transactions in last 24 hours |
| customer_avg_amount | float | Customer's historical average amount |
| distance_from_home_km | float | Distance from customer's home location |
| account_age_days | int | Age of customer's account in days |
| is_fraud | int | Target: 1 = fraudulent, 0 = legitimate |

**Fraud patterns**: Fraudulent transactions (3%) exhibit higher
amounts, unusual hours (night-skewed), location changes (65%),
new devices (55%), high transaction frequency, large distances from home,
and newer accounts.

---

## Regeneration

To regenerate the datasets:

```bash
python scripts/generate_data.py
```

All generation uses `RANDOM_SEED = 42` for full reproducibility.

---

## Important Notes

- All data is **synthetic** — generated algorithmically
- No real personal information is contained in any dataset
- Feature distributions are designed to approximate realistic banking patterns
- The datasets are for **educational and demonstration purposes only**
