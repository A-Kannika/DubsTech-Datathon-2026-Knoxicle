# Knoxicle: AccessGuru - Accessibility Insights (Web Accessibility Analytics & Risk Modeling)

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen)](https://accessguru-knoxicle.streamlit.app/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview
**Knoxicle: AccessGuru - Accessibility Insights** is a dashboard and Machine Learning & Risk Modeling built for the 7th DubsTech Datathon. It looks at more than 3,500 real web accessibility errors from 448 websites. I focused on six main areas: Tech, Education, Government, Health, News, and E-commerce.

Instead of just counting errors, I used Machine Learning to see how these errors actually impact users. The goal is to show which industries have the biggest barriers and help developers know what to fix first to make the web more inclusive.

---

## Key Features

- **Predict Violation Impact**: I used a Random Forest model to guess how serious a web error is (Critical, Serious, etc.) based on the industry and the type of violation.
- **Website Risk Clustering**: Using K-Means clustering, the app groups websites into "Low," "Moderate," or "High" risk levels based on their failure patterns.
- **Global Violation Hierarchy**: I added interactive Treemaps so you can click through different industries to see specific WCAG rule failures.
- **Compare Invisible Barriers**: Side-by-side analysis of different industries. You can pick two industries and compare their accessibility barriers directly.
- **Ranking of Inaccessible Design**: A leaderboard that shows which industries are most likely to have "Critical" blockers for users.

---

## Tech Stack

- **App Framework:** Streamlit (with custom CSS for the UI)
- **Data Handling:** Pandas
- **Visualizations**: Plotly (Express and Graph Objects)
- **Machine Learning**: Scikit-Learn (Random Forest, K-Means, and Preprocessing)

---

## How the Model Works

Knoxicle: AccessGuru - Accessibility Insights treats accessibility as a risk-management problem. Our ML logic follows a rigorous pipeline:
1. **Data Preprocessing**: Cleaning and standardizing categorical domain data.
2. **Feature Engineering**: Vectorizing text-based violations into numerical formats via Label Encoding.
3. **Training**: A Random Forest model (100 estimators) trained to recognize the relationship between WCAG categories and user impact.
4. **Testing**: You can use the "ML Insights" tab to try different scenarios and see the model's prediction in real-time.

---

## Model Performance & Validation

To make sure the predictions were accurate, I split the data (80% for training and 20% for testing):

- **Model Algorithm**: Random Forest Classifier (100 Estimators)
- **Model Accuracy**: 97.4% (Calculated via Mean Accuracy on unseen test data)
- **Margin of Error**: ±0.3%
- **Validation**: I used a hold-out test set to make sure the model works on new data, not just the data it already saw.

---

## Project Structure

```text
├── dashboard/
│   └── dashboard.py          # Main Streamlit application
├── data/
│   └── Access_to_Tech_Dataset.csv   # The accessibility data
├── assets/
│   └── icon.png              # App icon
├── requirements.txt          # Python libraries needed
└── README.md                 # Project documentation
```
---

## ▶️ Demo

Live Streamlit app: [https://accessguru-knoxicle.streamlit.app/](https://accessguru-knoxicle.streamlit.app/)  

GitHub repository: [https://github.com/A-Kannika/DubsTech-Datathon-2026-Knoxicle](https://github.com/A-Kannika/DubsTech-Datathon-2026-Knoxicle)

---

## Installation & Local Setup

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

## Team Knoxicle
Developed for the DubsTech Datathon 2026.
- Team member: Kannika Armstrong
- Goal: To transform raw accessibility data into a narrative of digital inclusion.
- Mission: Promoting Digital Equity through Data Science.

---

## 📜 License
This project is licensed under the MIT License.
