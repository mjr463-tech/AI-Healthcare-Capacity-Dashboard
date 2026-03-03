# Healthcare Capacity Planning Dashboard

An AI-powered dashboard for analyzing patient seasonality and optimizing resource allocation in healthcare settings.

## Features
- 📊 Real-time patient volume monitoring
- 🔍 Seasonality decomposition analysis
- 🔮 90-day demand forecasting with Prophet
- 💡 Resource allocation recommendations
- 📂 Interactive data upload (CSV/Excel)

## Quick Start

1. **Upload Your Data**
   - Click "Upload your healthcare dataset" in the sidebar
   - Select your CSV or Excel file with columns: `date`, `department`, `patient_count`

2. **View Dashboard**
   - Charts and insights update automatically

## Data Format

Your dataset should have these columns:
- `date` (YYYY-MM-DD format)
- `department` (e.g., Emergency, ICU, Surgery)
- `patient_count` (number of patients)
- Optional: `admissions`, `discharges`, `length_of_stay`

## Example Data

```csv
date,department,patient_count,admissions,discharges,length_of_stay
2024-01-01,Emergency,120,72,68,4
2024-01-01,ICU,30,18,15,6