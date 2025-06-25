
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Load models
svm_model = joblib.load('svm_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("🧠 Analisis Sentimen Komentar TikTok")
st.write("Prediksi kecenderungan self-diagnose berdasarkan komentar terkait kesehatan mental.")

menu = st.radio("Pilih Mode:", ["Prediksi Komentar", "Upload Dataset & Visualisasi"])

if menu == "Prediksi Komentar":
    comment = st.text_area("📝 Masukkan komentar TikTok di bawah ini:")
    algo = st.radio("🔍 Pilih Algoritma:", ["Support Vector Machine (SVM)", "Extreme Gradient Boosting (XGBoost)"])

    if st.button("🧞‍♀️ Prediksi Sentimen"):
        if comment:
            vectorized = vectorizer.transform([comment])
            pred = svm_model.predict(vectorized)[0] if algo == "Support Vector Machine (SVM)" else xgb_model.predict(vectorized)[0]
            label_map = {0: "negatif", 1: "netral", 2: "positif"}
            st.success(f"Hasil prediksi: **{label_map[pred]}**")
        else:
            st.warning("Masukkan komentar terlebih dahulu.")

elif menu == "Upload Dataset & Visualisasi":
    file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])
    if file:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        st.write("📄 Data Awal", df.head())

        if 'label' not in df.columns:
            st.warning("Kolom 'label' tidak ditemukan. Melabeli ulang menggunakan model.")
            df['label'] = xgb_model.predict(vectorizer.transform(df['Comment'].astype(str)))

        st.subheader("📊 Visualisasi Data")

        # Pie chart
        st.markdown("**Distribusi Label**")
        label_counts = df['label'].value_counts().sort_index()
        label_names = ['Negatif', 'Netral', 'Positif']
        fig1, ax1 = plt.subplots()
        ax1.pie(label_counts, labels=label_names, autopct='%1.1f%%', startangle=140, colors=['#E91E63', '#BA68C8', '#2196F3'])
        ax1.axis('equal')
        st.pyplot(fig1)

        # Boxplot
        df['clean_text'] = df['Comment'].astype(str)
        df['Panjang Komentar'] = df['clean_text'].apply(lambda x: len(str(x).split()))
        st.markdown("**Distribusi Panjang Komentar**")
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x='label', y='Panjang Komentar', ax=ax2, palette=['#E91E63', '#BA68C8', '#2196F3'])
        ax2.set_xticklabels(label_names)
        st.pyplot(fig2)

        # Histogram jam
        if 'Create Time' in df.columns:
            df['Create Time'] = pd.to_datetime(df['Create Time'], errors='coerce')
            df['Jam'] = df['Create Time'].dt.hour
            st.markdown("**Komentar per Jam**")
            fig3, ax3 = plt.subplots()
            sns.histplot(data=df, x='Jam', hue='label', multiple='stack', bins=24, ax=ax3, palette=['#E91E63', '#BA68C8', '#2196F3'])
            ax3.set_xticks(range(24))
            ax3.set_xticklabels(range(24))
            st.pyplot(fig3)

        # Countplot Type
        if 'Type' in df.columns:
            st.markdown("**Jenis Komentar: Comment vs Reply**")
            fig4, ax4 = plt.subplots()
            sns.countplot(data=df, x='Type', hue='label', ax=ax4, palette=['#E91E63', '#BA68C8', '#2196F3'])
            st.pyplot(fig4)

        # Engagement barplot
        if 'Digg Count' in df.columns and 'Reply Comment Total' in df.columns:
            st.markdown("**Rata-rata Likes & Replies**")
            engagement = df.groupby('label')[['Digg Count', 'Reply Comment Total']].mean().reset_index()
            melt_engage = engagement.melt(id_vars='label', value_vars=['Digg Count', 'Reply Comment Total'],
                                          var_name='Engagement Type', value_name='Average Count')
            fig5, ax5 = plt.subplots()
            sns.barplot(data=melt_engage, x='label', y='Average Count', hue='Engagement Type', ax=ax5)
            ax5.set_xticklabels(label_names)
            st.pyplot(fig5)

        # Heatmap positif
        st.markdown("**Heatmap Komentar Self-Diagnose Positif per Jam**")
        heatmap_data = df[df['label'] == 2].groupby('Jam').size().reindex(range(24), fill_value=0)
        fig6, ax6 = plt.subplots(figsize=(8, 1))
        sns.heatmap(heatmap_data.values.reshape(1, -1), cmap='Reds', annot=True, fmt='d',
                    xticklabels=range(24), yticklabels=['Komentar Positif'], ax=ax6)
        st.pyplot(fig6)

        # Train-test & evaluation
        st.subheader("📈 Evaluasi Model")
        X_train, X_test, y_train, y_test = train_test_split(df['Comment'], df['label'], test_size=0.2, random_state=42)
        X_test_vec = vectorizer.transform(X_test)
        y_pred_svm = svm_model.predict(X_test_vec)
        y_pred_xgb = xgb_model.predict(X_test_vec)

        acc_svm = accuracy_score(y_test, y_pred_svm)
        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        st.markdown(f"**Akurasi SVM:** {acc_svm:.4f}")
        st.markdown(f"**Akurasi XGBoost:** {acc_xgb:.4f}")
        st.text("Classification Report (XGBoost):")
        st.text(classification_report(y_test, y_pred_xgb, target_names=label_names))
