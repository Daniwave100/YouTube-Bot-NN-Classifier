import streamlit as st
import joblib
import torch
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

vectorizer = joblib.load('artifacts/vectorizer_real.pkl')
model = torch.load('artifacts/full_model_real.pt', weights_only=False)
model.eval()

tab1, tab2 = st.tabs(["Single Prediction", "Batch Test"])

with tab1:
    st.title("YouTube Bot Comment Classifier")
    user_input = st.text_area("Enter a YouTube comment:")
    if st.button("Predict"):
        input_corpus = [user_input]
        X_input = vectorizer.transform(input_corpus).toarray()
        X_input_tensor = torch.from_numpy(X_input).float()
        with torch.no_grad():
            output = model(X_input_tensor)
            pred = torch.argmax(output, dim=1).item()
            label = "Bot" if pred == 1 else "Human"
            st.write(f"Prediction: **{label}**")

with tab2:
    st.header("Batch Test: Upload CSV for Evaluation")
    test_file = st.file_uploader("Upload test CSV", type=["csv"])

    if test_file:
        df = pd.read_csv(test_file)
        st.write("Original data:")
        st.dataframe(df)

        text = st.selectbox("Select your text column", df.columns)
        boolean = st.selectbox("Select your boolean label column", df.columns)

        # Add validation to ensure proper encoding
        st.write(f"Unique values in {boolean} column: {sorted(df[boolean].unique())}")
        st.write("Expected: 0 = Human, 1 = Bot")

        X = vectorizer.transform(df[text].astype(str)).toarray()
        X_tensor = torch.from_numpy(X).float()
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1)[:, 1].numpy()  # Probability of being class 1 (Bot)
            preds = torch.argmax(outputs, dim=1).numpy()  # Predicted class label: 0 (Human) or 1 (Bot)
    
        y_true = df[boolean].values 
        cm = confusion_matrix(y_true, preds)
        
        # FIXED: Correct sensitivity and specificity calculations
        # Sensitivity = True Positive Rate = TP / (TP + FN) = How well we detect Bots
        sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        # Specificity = True Negative Rate = TN / (TN + FP) = How well we detect Humans
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        
        # FIXED: Correct descriptions
        st.write(f"Sensitivity (Bot Detection Rate - True Positive Rate): {sensitivity:.2f}")
        st.write(f"Specificity (Human Detection Rate - True Negative Rate): {specificity:.2f}")

        labels = ["Human", "Bot"]
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=labels, yticklabels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ROC Curve (this part was already correct)
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        st.subheader("Receiver Operating Characteristic (ROC Curve)")
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.legend()
        st.pyplot(fig2)