# 🎓 JoSAA College Predictor (2026 Edition)

A machine learning-based tool designed to help JEE Aspirants predict their potential college and branch based on historical JoSAA opening and closing ranks from **2023 to 2025**.

## 🚀 Features
- **Accurate Predictions**: Uses a Random Forest Regressor to predict closing ranks with high accuracy.
- **Interactive Dashboard**: Explore trends across NITs, IIITs, and GFTIs using Altair-powered visualizations.
- **Personalized Filters**: Filter by Institute, Program, Quota, Seat Type, and Gender.
- **2026 Ready**: Designed for the upcoming JoSAA 2026 counseling season.

## 📊 Model Performance
The model was trained on ~35,000 entries of JoSAA data.
- **R-squared (R²):** 0.865
- **Mean Absolute Error (MAE):** ~2,333 ranks

## 🛠️ Tech Stack
- **Languages**: Python
- **Libraries**: Scikit-learn, Pandas, NumPy, Altair
- **Deployment**: Streamlit

## 💻 How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rama-Krishna43/JoSAA-College-Predictor.git
   cd JoSAA-College-Predictor
   ```

2. **Install dependencies**:
   ```bash
   pip install streamlit pandas scikit-learn altair
   ```

3. **Run the app**:
   ```bash
   streamlit run Deployment.py
   ```

## 📸 Visualizations
The project includes a comprehensive data dashboard showing:
- Top 20 Institutes by program entries.
- Seat Type and Gender distributions.
- Quota-wise entry breakdowns.
