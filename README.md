# Knoxicle: AccessGuru - Accessibility Insights (Web Accessibility Analytics & Risk Modeling)

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen)](https://accessguru-knoxicle.streamlit.app/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 🌟 Overview
**Knoxicle: AccessGuru - Accessibility Insights** is an advanced analytics dashboard and predictive modeling tool designed to uncover systemic barriers in web accessibility. Developed for the **DubsTech Datathon 2026**, this project analyzes over 3,500 real-world WCAG 2.1 violations across 448 websites in 6 major domains (Tech, Education, Government, Health, News, and E-commerce).

By moving beyond simple "error counts," Knoxicle: AccessGuru uses Machine Learning to predict violation impact and clustering to identify high-risk digital environments, providing actionable insights for digital equity.

---

## 🚀 Key Features

- **🔮 Predictive Impact Modeling**: A Random Forest Classifier that estimates the severity of a violation (Critical, Serious, Moderate, Minor) based on the domain and violation type.
- **🛡️ Website Risk Clustering**: K-Means clustering identifies "High Risk" websites by analyzing multi-dimensional failure patterns.
- **🌲 Hierarchical Deep-Dive**: Interactive Treemaps that allow users to drill down from Industry domains into specific WCAG rule failures.
- **🔥 Barrier Comparison**: Side-by-side analysis of different industries to see which sectors are leading—or lagging—in accessible design.
- **🏆 Domain Ranking**: A risk-based leaderboard ranking industries by their likelihood of presenting "Critical" accessibility blockers.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (Custom CSS & Interactive UI)
- **Data Engine**: [Pandas](https://pandas.pydata.org/)
- **Visualizations**: [Plotly Express](https://plotly.com/python/), [Graph Objects](https://plotly.com/graph-objects/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (Random Forest, K-Means, Label Encoding, StandardScaler)

---

## 📊 The Machine Learning Pipeline

AccessGuru treats accessibility as a risk-management problem. Our ML logic follows a rigorous pipeline:
1. **Data Preprocessing**: Cleaning and standardizing categorical domain data.
2. **Feature Engineering**: Vectorizing text-based violations into numerical formats via Label Encoding.
3. **Training**: A Random Forest model (100 estimators) trained to recognize the relationship between WCAG categories and user impact.
4. **Explainability**: The "ML Insights" tab provides an interface for users to test hypothetical scenarios and see real-time AI predictions.

---

## 📉 Model Performance & Validation

To ensure the reliability of our accessibility risk predictions, we validated the model using an 80/20 train-test split:

- **Algorithm**: Random Forest Classifier (100 Estimators)
- **Model Accuracy**: 97.4% (Calculated via Mean Accuracy on unseen test data)
- **Margin of Error**: ±0.3%
- **Validation Method**: Hold-out validation to prevent overfitting and ensure the model generalizes well to new, unseen websites.

---

## 📂 Project Structure

```text
├── dashboard/
│   └── dashboard.py          # Main Streamlit application
├── data/
│   └── Access_to_Tech_Dataset.csv   # Processed accessibility dataset
├── assets/
│   └── icon.png              # Project Branding
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```
---

## ▶️ Demo

Live Streamlit app: [https://accessguru-knoxicle.streamlit.app/](https://accessguru-knoxicle.streamlit.app/)  

GitHub repository: [https://github.com/A-Kannika/DubsTech-Datathon-2026-Knoxicle](https://github.com/A-Kannika/DubsTech-Datathon-2026-Knoxicle)

---

## ⚙️ Installation & Local Setup

1. **Clone the repository**:

```bash
git clone https://github.com/A-Kannika/DubsTech-Datathon-2026-Knoxicle.git
cd DubsTech-Datathon-2026-Knoxicle
```

2. **Create and activate a virtual environment:**:
   
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**:
   
```bash
pip install -r requirements.txt
```

4. **Run the app:**:
   
```bash
streamlit run dashboard/dashboard.py
```
---

## 🤝 Team Knoxicle
Developed with ❤️ for the DubsTech Datathon 2026.
- Team member: Kannika Armstrong
- Goal: To transform raw accessibility data into a narrative of digital inclusion.
- Mission: Promoting Digital Equity through Data Science.

---

## 📜 License
This project is licensed under the MIT License.
