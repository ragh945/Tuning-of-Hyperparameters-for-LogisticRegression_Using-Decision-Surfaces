import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlxtend.plotting import plot_decision_regions
from PIL import Image

Hyp=Image.open(r"C:\Users\RAGHAVENDRA KUMAR\Downloads\hyperparameters-in-machine-learning2.png")
st.image(Hyp,use_column_width=False)

# Generate synthetic dataset
X, y = make_classification(
    n_samples=1000,   # Number of samples
    n_features=2,     # Number of features
    n_informative=2,  # Number of informative features
    n_redundant=0,    # No redundant features
    n_clusters_per_class=1,  # Number of clusters per class
    n_classes=3,     # Number of classes
    random_state=42  # For reproducibility
)
df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
df['Target'] = y
df.to_csv('synthetic_dataset.csv', index=False)

# Load dataset
df = pd.read_csv('synthetic_dataset.csv')
X = df[['Feature 1', 'Feature 2']].values
y = df['Target'].values

# Sidebar for hyperparameter inputs
st.title("Logistic Regression Hyperparameters and Metrics")

# Logistic Regression hyperparameters
C_value = st.sidebar.slider("Inverse of regularization strength (C)", 0.01, 10.0, 1.0)
solver = st.sidebar.selectbox("Solver", ("liblinear", "newton-cg", "lbfgs", "sag", "saga"))
penalty = st.sidebar.selectbox("Penalty", ("l1", "l2", "elasticnet", "none"))
tol = st.sidebar.slider("Tolerance for stopping criteria (tol)", 0.001, 0.01, 0.001)
max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 200)
random_state = st.sidebar.slider("Random State", 23, 42)
l1_ratio = st.sidebar.slider("L1 Ratio (only for elasticnet)", 0.0, 1.0, 0.5)
multiclass = st.sidebar.selectbox("Multiclass", ("auto", "ovr", "multinomial"))
class_weight = st.sidebar.selectbox("Class Weight", (None, "balanced"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Logistic Regression model with hyperparameters from sidebar
model = LogisticRegression(
    C=C_value, 
    solver=solver, 
    penalty=penalty if solver != 'newton-cg' else 'l2',  # ElasticNet only works with 'saga'
    tol=tol, 
    max_iter=max_iter,
    random_state=random_state, 
    l1_ratio=l1_ratio if penalty == 'elasticnet' else None,  # Only for elasticnet
    multi_class=multiclass,
    class_weight=class_weight
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Display hyperparameters and metrics side by side
col1, col2 = st.columns(2)

with col1:
    st.write("### Hyperparameter Values")
    st.write(f"C (Regularization strength): {C_value}")
    st.write(f"Solver: {solver}")
    st.write(f"Penalty: {penalty}")
    st.write(f"Tolerance (tol): {tol}")
    st.write(f"Max Iterations: {max_iter}")
    st.write(f"Random State: {random_state}")
    st.write(f"L1 Ratio: {l1_ratio}")
    st.write(f"Multiclass Strategy: {multiclass}")
    st.write(f"Class Weight: {class_weight}")

with col2:
    st.write("### Classification Metrics")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")

# Plot decision surface
st.write("### Decision Surface for Logistic Regression")
plt.figure(figsize=(8, 6))
plot_decision_regions(X_train, y_train, clf=model, legend=2)
plt.title('Decision Surface for Logistic Regression')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
st.pyplot(plt)
