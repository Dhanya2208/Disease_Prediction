import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


@st.cache_data
def load_data():
    data = pd.read_csv("E:\DHANYA\python\Training.csv").dropna(axis=1)
    encoder = LabelEncoder()
    data["prognosis"] = encoder.fit_transform(data["prognosis"])
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return data, X, y, encoder

data, X, y, encoder = load_data()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


symptom_index = {}
for idx, col in enumerate(X.columns):
    formatted = " ".join([w.capitalize() for w in col.split("_")])
    symptom_index[formatted] = idx

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}


def predictDisease(symptoms_input):
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms_input:
        if symptom in data_dict["symptom_index"]:
            input_data[data_dict["symptom_index"][systmptom]] = 1
    input_data = np.array(input_data).reshape(1, -1)
    prediction = data_dict["predictions_classes"][rf_model.predict(input_data)[0]]
    return prediction


st.title("🩺 Disease Prediction System (Random Forest Based)")
st.markdown("Predict diseases based on symptoms using Machine Learning (Random Forest).")

tab1, tab2, tab3 = st.tabs(["🧠 Predict", "📊 Analysis", "📘 Info"])

with tab1:
    st.header("🔍 Enter Your Symptoms")
    symptom_options = sorted(list(symptom_index.keys()))
    selected_symptoms = st.multiselect("Select symptoms you are experiencing:", symptom_options)

    if st.button("Predict Disease"):
        if selected_symptoms:
            prediction = predictDisease(selected_symptoms)
            st.success(f"✅ *Predicted Disease:* {prediction}")
        else:
            st.warning("⚠ Please select at least one symptom.")

with tab2:
    st.header("📈 Training Data Visualizations")
    

     #CHART 1
    st.subheader("🔢 Disease Distribution")
    fig1, ax1 = plt.subplots(figsize=(16, 6))
    disease_names = [data_dict["predictions_classes"][i] for i in data["prognosis"]]
    sns.countplot(x=disease_names, order=pd.Series(disease_names).value_counts().index, ax=ax1)
    ax1.set_title("Disease Frequency", fontsize=14)
    ax1.set_xlabel("Disease", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    st.pyplot(fig1)

    #CHART 2
    st.subheader("🔥 Top 20 Frequent Symptoms")
    symptom_sums = X.sum().sort_values(ascending=False).head(20)
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    sns.barplot(x=symptom_sums.index, y=symptom_sums.values, palette="rocket", ax=ax2)
    ax2.set_title("Top 20 Symptoms", fontsize=14)
    ax2.set_ylabel("Number of Records", fontsize=12)
    ax2.set_xticklabels(symptom_sums.index, rotation=45)
    st.pyplot(fig2)

    #CHART 3
    st.subheader("📊 Average Number of Symptoms per Disease")
    df = X.copy()
    df["prognosis"] = y
    avg_symptoms = df.groupby("prognosis").sum().sum(axis=1) / df["prognosis"].value_counts()
    avg_symptoms = avg_symptoms.sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(20, 8))  # Bigger figure
    disease_labels = [data_dict["predictions_classes"][i] for i in avg_symptoms.index]
    sns.barplot(x=disease_labels, y=avg_symptoms.values, palette="viridis", ax=ax3)
    ax3.set_title("Average Number of Symptoms per Disease", fontsize=16)
    ax3.set_ylabel("Average Number of Symptoms", fontsize=14)
    ax3.set_xlabel("Disease", fontsize=14)
    ax3.tick_params(axis='x', rotation=90)
    plt.tight_layout()  # Fix overlaps
    st.pyplot(fig3)

    #CHART 4
    st.subheader("📌 Correlation of Top 20 Symptoms")
    top_symptoms = symptom_sums.head(20).index
    fig4, ax4 = plt.subplots(figsize=(14, 10))
    sns.heatmap(X[top_symptoms].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax4)
    ax4.set_title("Correlation Matrix - Top 20 Symptoms", fontsize=14)
    st.pyplot(fig4)

    #CHART 5
    st.subheader("👤 Number of Symptoms per Record")
    symptom_counts = X.sum(axis=1)
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    sns.histplot(symptom_counts, kde=True, bins=30, color='teal', ax=ax5)
    ax5.set_title("Distribution of Symptoms per Record", fontsize=14)
    ax5.set_xlabel("Number of Symptoms", fontsize=12)
    st.pyplot(fig5)

with tab3:
    st.header("ℹ About This App")
    st.markdown("""
    - This system is trained on a dataset with symptoms mapped to 40+ diseases.
    - Only the *Random Forest* model is used for prediction due to its high accuracy.
    - You can analyze disease trends and symptom distributions using the *Analysis* tab.
    - The system works purely on symptom inputs — no patient details are required.

    **Built with:**
    - Scikit-learn
    - Streamlit
    - Seaborn / Matplotlib
    """)
    #ACCURACY
    st.subheader("📏 Model Accuracy & Confusion Matrix")

    from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

    # Accuracy
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.markdown(f"### 🎯 Accuracy on Test Set: `{acc:.2%}`")

    # Confusion Matrix Plot
    fig6, ax6 = plt.subplots(figsize=(20, 18))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(ax=ax6, xticks_rotation=90, cmap="Blues", colorbar=False)
    ax6.set_title("Confusion Matrix - Random Forest", fontsize=16)
    st.pyplot(fig6)

