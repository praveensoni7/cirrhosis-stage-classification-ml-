# liver cirrhosis stage 

_Data-driven approach to staging liver cirrhosis with advanced classification techniques._

---
## 📌 Table of contents
- <a href = "#overview">Overview</a>
- <a href = "#problem">Problem</a>
- <a href = "#dataset"> Dataset </a>
- <a href = "#tool--technologies" >Tools & Techologies </a>
- <a href = "#project-structure"> project Structure </a>
- <a href = "#data-cleaning--reparation" >Data Cleaning & Preparation </a>
- <a herf = "#exploratory-data-analysis-eda">Exploratory Data Analysis </a>
- <a href = "#research-questions--key-findings">Research questions & Key Findings</a>
- <a href = "#Deployementpage">Deployement</a>
- <a href = "#How-to-run-this-project">How to run This Project </a>
- <a href = "#final-Recommendations">Final Recommendations</a>
- <a href = "#author--Contact">Author & Contact </a>

---
<h2><a class="anchor" id="overview"></a>Overview</h2>
This project focuses on predicting the stage of liver cirrhosis using machine learning models. By leveraging clinical and biochemical data, the goal is to build a robust pipeline that supports medical decision-making and highlights the potential of data-driven healthcare solutions.
---
<h2><a class="anchor" id="problem"></a>Problem</h2>
Liver cirrhosis is a progressive disease where accurate staging is critical for treatment planning. Traditional methods can be invasive and time-consuming. This project aims to provide a non-invasive, data-driven approach to predict cirrhosis stages efficiently

---
<h2><a class="anchor" id="dataset"></a>Dataset</h2>
- CSV file located in `/data/` folder 
- Summary table created from ingested data and used form analysis
- Data Source: Publicly available medical datasets in kaggle.
- Target: Cirrhosis stage classification

![Liver Cirrhosis image](Liver_Disease_Cirrhosis.png)

---
<h2><a class="anchor" id="tool--technologies"></a>Tools & Techologies</h2>
- Languages: Python
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- Deployment: Streamlit / Flask

---
<h2><a class="anchor" id="project-structure"></a>Project Structure</h2>

```
liver-cirrhosis-stage-prediction/
│── dataset/             # Raw and processed datasets
│── notebooks/           # Jupyter notebooks for EDA & modeling
│── scripts/             # Source code
├── outcome/             # Reports
│── README.md            # Project documentation
│── requirements.txt     # Dependencies
├── .gitignore             # Ignore unnecessary files

```

---
<h2><a class="anchor" id="data-cleaning--reparation"></a>Data Cleaning & Preparation</h2>
- Handling missing values (mean/median imputation).
- Encoding categorical variables (gender, diagnosis).
- Normalization/standardization of numerical features.
- Splitting dataset into train/test sets.
---
<h2><a class="anchor" id="exploratory-data-analysis-eda"></a>Exploratory Data Analysis</h2>
-	Distribution plots of key features (bilirubin, albumin, cholesterol).
- 	Correlation heatmaps to identify relationships.
- 	Boxplots to compare feature values across cirrhosis stages.
- 	Outlier detection and treatment.
---
<h2><a class="anchor" id="research-questions--key-findings"></a>Research questions & Key Findings</h2>
- Which clinical features are most predictive of cirrhosis stage?
- How do ensemble models compare to deep learning approaches?
- Key finding: Bilirubin, albumin, and platelet count emerged as strong predictors.
- Random Forest classifier achieved the highest accuracy (~95%) in classification
---
<h2><a class="anchor" id="Deployementpage"></a>Deployement</h2>
- Built a Streamlit web app for interactive stage prediction.
- Users can input patient data and receive stage predictions instantly.
---
<h2><a class="anchor" id="How-to-run-this-project"></a>How to run This Project </h2>

1) Clone the repository
```
git clone https://github.com/praveensoni7/cirrhosis-stage-classification-ml-.git

```
2) Install dependencies
```
pip install -r requirements.txt
```
3) Run the notebook for training
```
jupyter notebook notebooks/liver_cirrhosis_stage_prediction.ipynb
```
4) Launch the app:
```
streamlit run liver_cirrohsis_model.py
```
---
<h2><a class="anchor" id="final-Recommendations"></a>Final Recommendations
- Ensemble models (XGBoost/LightGBM) are recommended for best performance.
- Larger datasets and external validation would improve generalizability.
- Future work: Incorporate explainable AI (SHAP/LIME) for clinical interpretability.

---
<h2><a class="anchor" id="author--Contact"></a>Author & Contact</h2>

- Author: **Praveen Soni**
- 📧 Email: sonipraveen220@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/praveensoni7
- 🌐 

---

