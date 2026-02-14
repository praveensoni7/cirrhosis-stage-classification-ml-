import os
import io
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc, RocCurveDisplay
)
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from joblib import load

# -------------------------------
# App config
# -------------------------------
st.set_page_config(
    page_title="Liver Cirrhosis Stage Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Sidebar: Controls
# -------------------------------
st.sidebar.title("⚙️ Controls")

# Data upload
uploaded = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
target_col = st.sidebar.text_input("Target column (default: 'Stage')", value="Stage")

# Model selection and CV
model_choice = st.sidebar.selectbox("Model", ["SVM (RBF/Linear)", "Logistic Regression", "Random Forest"])
cv_folds = st.sidebar.slider("Cross-validation folds", min_value=3, max_value=10, value=5)
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
use_class_weight = st.sidebar.checkbox("Use class_weight='balanced' (recommended if imbalanced)", value=True)

# SVM specific grid
st.sidebar.subheader("SVM Grid")
C_values = st.sidebar.multiselect("C values", [0.5, 1, 2, 5, 10], default=[0.5, 1, 2])
kernels = st.sidebar.multiselect("Kernels", ["rbf", "linear"], default=["rbf", "linear"])
gammas = st.sidebar.multiselect("Gamma", ["scale", "auto"], default=["scale", "auto"])

# RandomForest grid
st.sidebar.subheader("Random Forest Grid")
n_estimators = st.sidebar.multiselect("n_estimators", [100, 200, 400], default=[100, 200])
max_depths = st.sidebar.multiselect("max_depth", [None, 10, 20, 30], default=[None, 10, 20])

# Logistic Regression grid
st.sidebar.subheader("Logistic Regression Grid")
lr_C_values = st.sidebar.multiselect("C values (LR)", [0.5, 1, 2, 5, 10], default=[1, 2, 5])
lr_penalty = st.sidebar.multiselect("Penalty", ["l2"], default=["l2"])
lr_solver = st.sidebar.selectbox("Solver", ["lbfgs", "saga"], index=0)

train_button = st.sidebar.button("🚀 Train & Evaluate")
save_button = st.sidebar.button("💾 Save Best Model")
load_button = st.sidebar.button("📂 Load Saved Model")
predict_mode = st.sidebar.radio("Prediction mode", ["Form input", "Batch upload"])

# -------------------------------
# Helper functions
# -------------------------------
@st.cache_data(show_spinner=False)
def load_data(file: io.BytesIO) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df

def detect_column_types(df: pd.DataFrame, target: str):
    # Infer numerical vs categorical
    feature_cols = [c for c in df.columns if c != target]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor

def get_param_grid(model_choice, C_values, kernels, gammas, n_estimators, max_depths, lr_C_values, lr_penalty, lr_solver, use_class_weight):
    if model_choice == "SVM (RBF/Linear)":
        grid = {
            "clf__C": C_values,
            "clf__kernel": kernels,
        }
        # gamma only applies to rbf kernel; GridSearch will try combos anyway
        grid["clf__gamma"] = gammas
        base = SVC(probability=True, class_weight=("balanced" if use_class_weight else None), random_state=42)
    elif model_choice == "Random Forest":
        grid = {
            "clf__n_estimators": n_estimators,
            "clf__max_depth": max_depths,
        }
        base = RandomForestClassifier(
            class_weight=("balanced" if use_class_weight else None),
            random_state=42,
            n_jobs=-1
        )
    else:  # Logistic Regression
        grid = {
            "clf__C": lr_C_values,
            "clf__penalty": lr_penalty,
            "clf__solver": [lr_solver],
            "clf__max_iter": [200]
        }
        base = LogisticRegression(
            class_weight=("balanced" if use_class_weight else None),
            random_state=42,
            n_jobs=-1
        )
    return base, grid

def plot_confusion(cm, class_names):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

def plot_roc(y_test, y_proba, classes):
    # One-vs-rest ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve((y_test == cls).astype(int), y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=str(cls)).plot(ax=ax)
    ax.set_title("ROC Curves (One-vs-Rest)")
    st.pyplot(fig)

def compute_permutation_importance(model, X_test, y_test, feature_names):
    try:
        r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
        imp_df = pd.DataFrame({"feature": feature_names, "importance": r.importances_mean})
        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=imp_df, x="importance", y="feature", ax=ax)
        ax.set_title("Permutation Feature Importance (Top 20)")
        st.pyplot(fig)
    except Exception as e:
        st.info(f"Feature importance not available: {e}")

# -------------------------------
# Data section
# -------------------------------
st.title("🩺 Liver Cirrhosis Stage Classifier")
st.markdown("A clean, interactive dashboard to train, evaluate, and deploy a stage prediction model.")

if uploaded:
    df = load_data(uploaded)
    st.subheader("Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Basic stats
    if target_col in df.columns:
        st.markdown("#### Target distribution")
        tgt_counts = df[target_col].value_counts(dropna=False)
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(x=tgt_counts.index.astype(str), y=tgt_counts.values, ax=ax, palette="viridis")
            ax.set_title("Stage counts")
            ax.set_xlabel("Stage")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        with col2:
            st.metric(label="Unique stages", value=str(df[target_col].nunique()))
            st.metric(label="Rows", value=str(len(df)))
    else:
        st.error(f"Target column '{target_col}' not found in dataset.")
        st.stop()

    # Detect dtypes
    numeric_cols, categorical_cols = detect_column_types(df, target_col)
    st.markdown("#### Detected columns")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Numeric:", numeric_cols)
    with c2:
        st.write("Categorical:", categorical_cols)

    # Train-test split
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str)  # Ensure categorical labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Build pipeline
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    base_estimator, param_grid = get_param_grid(
        model_choice, C_values, kernels, gammas,
        n_estimators, max_depths, lr_C_values, lr_penalty, lr_solver, use_class_weight
    )
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", base_estimator)])

    # Training
    if train_button:
        st.subheader("Training & tuning")
        with st.spinner("Running cross-validation..."):
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            grid = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
                verbose=0,
                refit=True
            )
            start = time.time()
            grid.fit(X_train, y_train)
            duration = time.time() - start

        best_model = grid.best_estimator_
        st.success(f"Training complete in {duration:.2f} seconds")
        st.write("Best params:", grid.best_params_)
        st.write(f"Best CV accuracy: {grid.best_score_:.4f}")

        # Evaluate on holdout
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        st.subheader("Evaluation on test set")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Accuracy", f"{acc:.4f}")
        with m2:
            st.metric("Weighted F1", f"{f1:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
        st.markdown("#### Confusion matrix")
        plot_confusion(cm, class_names=sorted(y.unique()))

        # Classification report
        st.markdown("#### Classification report")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(report).transpose()
        st.dataframe(rep_df, use_container_width=True)

        # ROC curves (if probabilities are available)
        try:
            y_proba = best_model.predict_proba(X_test)
            st.markdown("#### ROC curves")
            plot_roc(y_test.values, y_proba, classes=best_model.named_steps["clf"].classes_)
        except Exception:
            st.info("Probabilistic outputs not available for this estimator/setting.")

        # Permutation feature importance (works best with tree models)
        st.markdown("#### Feature importance (permutation)")
        # Extract transformed feature names
        try:
            prep = best_model.named_steps["prep"]
            cat_features = list(prep.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols))
            num_features = numeric_cols
            feature_names = num_features + cat_features
            # Get model’s last step
            clf = best_model.named_steps["clf"]
            # Use a small subset for speed
            X_test_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
            y_test_sample = y_test.loc[X_test_sample.index]
            # Get transformed data for permutation_importance (needs numeric array)
            Xt = prep.transform(X_test_sample)
            compute_permutation_importance(clf, Xt, y_test_sample, feature_names)
        except Exception as e:
            st.info(f"Could not compute feature importance: {e}")

        # Cache model in session state
        st.session_state["best_model"] = best_model
        st.session_state["numeric_cols"] = numeric_cols
        st.session_state["categorical_cols"] = categorical_cols
        st.session_state["classes"] = sorted(y.unique())

    # Save / Load model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    if save_button:
        if "best_model" in st.session_state:
            path = os.path.join(models_dir, f"best_model_{model_choice.replace(' ', '_')}.joblib")
            dump(st.session_state["best_model"], path)
            st.success(f"Model saved: {path}")
        else:
            st.warning("No trained model found. Train first.")

    if load_button:
       
        path = os.path.join(models_dir, f"best_model_{model_choice.replace(' ', '_')}.joblib")
        if os.path.exists(path):
            st.session_state["best_model"] = load(path)
            st.success(f"Loaded model: {path}")
        else:
            st.warning("No saved model found for this selection.")

    # -------------------------------
    # Prediction section
    # -------------------------------
    st.header("🔮 Predictions")

    if "best_model" not in st.session_state:
        st.info("Train or load a model to enable predictions.")
        st.stop()

    best_model = st.session_state["best_model"]
    classes = st.session_state.get("classes", [])

    if predict_mode == "Form input":
        st.markdown("Use the dynamic form to input patient attributes.")
        
        with st.form("prediction_form"):
            inputs = {}
            for col in X.columns:
                if col in numeric_cols:
                    inputs[col] = st.number_input(f"{col}", value=float(X[col].median()) if pd.api.types.is_numeric_dtype(X[col]) else 0.0)
                else:
                    
                    vals = sorted([v for v in df[col].dropna().unique().tolist() if str(v) != ""])
                    default = vals[0] if vals else ""
                    inputs[col] = st.selectbox(f"{col}", options=vals if len(vals) > 0 else [default])
            submit = st.form_submit_button("Predict")
        if submit:
            row = pd.DataFrame([inputs])
            try:
                proba = best_model.predict_proba(row)[0]
                pred = best_model.predict(row)[0]
                st.success(f"Predicted stage: {pred}")
                
                prob_df = pd.DataFrame({"stage": best_model.named_steps["clf"].classes_, "probability": proba})
                st.bar_chart(prob_df.set_index("stage"))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        st.markdown("Upload a CSV with the same feature columns for batch predictions.")
        batch_file = st.file_uploader("Upload batch CSV", type=["csv"], key="batch")
        if batch_file:
            batch_df = pd.read_csv(batch_file)
            st.dataframe(batch_df.head(), use_container_width=True)
            try:
                preds = best_model.predict(batch_df)
                out = batch_df.copy()
                out["predicted_stage"] = preds
                st.markdown("#### Results")
                st.dataframe(out.head(20), use_container_width=True)
                
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

else:
    st.info("Upload your CSV to begin.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit • Tuned with GridSearchCV • Designed for clear clinical insights")



#================== without uploading dataset ===================#
st.write("Enter patient details to predict Liver Cirrhosis Stage without uploading dataset.")

model_path = os.path.join("scripts","Liver_disease_pipeline.pkl")
model = joblib.load(model_path)

FEATURE_COLUMNS = [
    'N_Days', 'Spiders', 'Prothrombin', 'SGOT', 'Drug',
    'Age', 'Ascites', 'Platelets', 'Cholesterol',
    'Bilirubin', 'Copper', 'Alk_Phos',
    'Albumin', 'Status', 'Sex', 'Tryglicerides',
    'Hepatomegaly', 'Edema'
]

X_transformed = model.named_steps['preprocessor'].transform(input_data)

# ======================================
# Sidebar Inputs
input_data = pd.DataFrame([{
    "N_Days": N_Days,
    "Spiders": Spiders,
    "Prothrombin": Prothrombin,
    "SGOT": SGOT,
    "Drug": Drug,
    "Age": Age,
    "Ascites": Ascites,
    "Platelets": Platelets,
    "Cholesterol": Cholesterol,
    "Bilirubin": Bilirubin,
    "Copper": Copper,
    "Alk_Phos": Alk_Phos,
    "Albumin": Albumin,
    "Status": Status,
    "Sex": Sex,
    "Tryglicerides": Tryglicerides,
    "Hepatomegaly": Hepatomegaly,
    "Edema": Edema
}])



# ======================================
# Predict (pipeline handles preprocessing)
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Cirrhosis Stage: {prediction[0]}")
