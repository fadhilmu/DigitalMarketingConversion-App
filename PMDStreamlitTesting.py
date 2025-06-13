import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import joblib
from collections import Counter
import plotly.graph_objects as go

st.set_page_config(page_title="Conversion Prediction", layout="wide")
st.title("Conversion Prediction in Digital Marketing")
col1, col2 = st.columns(2)
st.markdown("---")

with col1:
    if st.button("📊 Deskripsi Data", use_container_width=True):
        st.session_state.menu = "Deskripsi Data"
    if st.button("📈 Performa Model", use_container_width=True):
        st.session_state.menu = "Performa Model"

with col2:
    if st.button("🧠 Deskripsi Model", use_container_width=True):
        st.session_state.menu = "Deskripsi Model"
    if st.button("📝 Input & Prediksi", use_container_width=True):
        st.session_state.menu = "Input & Prediksi"






# Algoritma Manual
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, random_state=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.random_state = np.random.RandomState(random_state)

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = self.random_state.choice(n_feats, self.n_features, replace=False)

        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])


    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.random_state = np.random.RandomState(random_state)
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree_random_state = self.random_state.randint(0, 1_000_000)
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features,
                                random_state=tree_random_state)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = self.random_state.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions

def compute_class_weights(y):
    counts = Counter(y)
    total = len(y)
    class_weights = {label: total/count for label, count in counts.items()}
    return class_weights

def svm_train(X, y, class_weights, lr=0.001, epochs=1000, C=1.0):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0

    for epoch in range(epochs):
        for idx in range(n_samples):
            xi = X[idx]
            yi = y[idx]

            weight = class_weights[yi]

            condition = yi * (np.dot(w, xi) + b)
            if condition >= 1:
                w -= lr * 2 * w
            else:
                w -= lr * (2 * w - C * weight * yi * xi)
                b -= lr * (-C * weight * yi)
    return w, b

def svm_predict(X, w, b):
    linear_output = np.dot(X, w) + b
    return np.where(linear_output >= 0, 1, -1)





if "menu" not in st.session_state:
    st.session_state.menu = None

@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    import joblib
    import pandas as pd

    df_original = pd.read_csv('digital_marketing_campaign_dataset.csv')
    df = df_original.copy()

    df.drop(columns=['CustomerID', 'AdvertisingTool', 'AdvertisingPlatform'], inplace=True)
    df.drop(columns=['SocialShares', 'CampaignChannel', 'Gender', 'Age'], inplace=True)

    preprocessor = joblib.load('preprocessor.pkl')
    encoders = preprocessor['encoders']
    scaler = preprocessor['scaler']
    categorical_feat = preprocessor['categorical_feat']
    numerical_feat = preprocessor['numerical_feat']
    target = 'Conversion'

    df_encoded = df.copy()
    for col in categorical_feat:
        df_encoded[col] = encoders[col].transform(df_encoded[col])

    df_scaled = df_encoded.copy()
    df_scaled[numerical_feat] = scaler.transform(df_scaled[numerical_feat])

    features = categorical_feat + numerical_feat
    X = df_scaled[features].values
    y = df_scaled[target].values

    return df, df_encoded, df_scaled, X, y, features, numerical_feat, categorical_feat, target

df, df_encoded, df_scaled, X, y, features, numerical_feat, categorical_feat, target = load_and_preprocess_data()





if st.session_state.menu == "Deskripsi Data":
    st.header("Deskripsi Data")

    raw_features = numerical_feat + categorical_feat

    st.write("Sampel Dataset:")
    st.dataframe(df[raw_features].head(10))

    st.write("Statistika Deskriptif Dataset:")
    st.dataframe(df[numerical_feat].describe())

    st.markdown("---")

    st.subheader("1. Demographic Information")
    st.markdown("""
    - **Income**: Pendapatan tahunan pelanggan dalam USD.
    """)

    st.subheader("2. Marketing-specific Variables")
    st.markdown("""
    - **CampaignType**: Jenis kampanye pemasaran (Awareness, Consideration, Conversion, Retention).
    - **AdSpend**: Jumlah dana yang dikeluarkan untuk kampanye dalam USD.
    - **ClickThroughRate**: Rasio klik terhadap konten pemasaran.
    - **ConversionRate**: Rasio konversi dari klik ke aksi yang diinginkan (misalnya pembelian).
    """)

    st.subheader("3. Customer Engagement Variables")
    st.markdown("""
    - **WebsiteVisits**: Jumlah kunjungan ke situs web.
    - **PagesPerVisit**: Rata-rata jumlah halaman yang dikunjungi per sesi.
    - **TimeOnSite**: Waktu rata-rata yang dihabiskan di situs per kunjungan (dalam menit).
    - **EmailOpens**: Jumlah email pemasaran yang dibuka oleh pelanggan.
    - **EmailClicks**: Jumlah klik pada tautan dalam email pemasaran.
    """)

    st.subheader("4. Historical Data")
    st.markdown("""
    - **PreviousPurchases**: Jumlah pembelian yang telah dilakukan sebelumnya oleh pelanggan.
    - **LoyaltyPoints**: Jumlah poin loyalitas yang telah dikumpulkan pelanggan.
    """)

    st.subheader("5. Target Variable")
    st.markdown("""
    - **Conversion**: Variabel target biner yang menunjukkan apakah pelanggan melakukan konversi atau tidak (1 = Ya, 0 = Tidak).
    """)

    st.markdown("---")

    st.markdown("##### Fitur Numerik")

    stats = df[numerical_feat].agg(['min', 'max']).transpose().reset_index()
    stats.columns = ['Fitur', 'Minimum', 'Maksimum']

    left, center, right = st.columns([2, 2, 2])

    with left:
        st.dataframe(stats)

    st.markdown("##### Fitur Kategorik")
    for col in categorical_feat:
        unique_vals = df[col].unique()
        st.markdown(f"- **{col}**: {', '.join([str(v) for v in unique_vals])}")

    st.markdown("---")

    left, right = st.columns([2, 2])
    with left:
        st.markdown("### Distribusi Target: Conversion")
        conversion_counts = df[target].value_counts().sort_index()
        conversion_labels = ['Tidak Konversi (0)', 'Konversi (1)']
        conversion_values = [conversion_counts.get(0, 0), conversion_counts.get(1, 0)]

        fig_pie = px.pie(
            names=conversion_labels,
            values=conversion_values,
            color=conversion_labels,
            color_discrete_map={
                'Tidak Konversi (0)': '#1f77b4',
                'Konversi (1)': '#ff7f0e'
            },
            title='Proporsi Conversion'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with right:
        st.markdown("### Distribusi Fitur")
        selected_feature = st.selectbox("Pilih fitur:", options=features, key="feature_select")

        color_palette = px.colors.qualitative.Set2

        numerical_feat = df_encoded.select_dtypes(include=['number']).columns.tolist()
        categorical_feat = [col for col in features if col not in numerical_feat]

        if selected_feature in categorical_feat:

            df_encoded[selected_feature] = df_encoded[selected_feature].astype(str)

            fig = px.histogram(
                df_encoded,
                x=selected_feature,
                color=selected_feature,
                labels={selected_feature: selected_feature, 'count': 'Frekuensi'},
                title=f'Distribusi Kategorikal - {selected_feature}',
                color_discrete_sequence=color_palette,
                category_orders={selected_feature: sorted(df_encoded[selected_feature].unique())}
            )
        else:
            fig = px.histogram(
                df_encoded,
                x=selected_feature,
                color='Conversion',
                barmode='overlay',
                nbins=30,
                labels={selected_feature: selected_feature, 'count': 'Frekuensi', 'Conversion': 'Conversion'},
                title=f'Distribusi Numerik - {selected_feature} Berdasarkan Conversion',
                color_discrete_map={'0': '#1f77b4', '1': '#ff7f0e'}
            )

            fig.update_traces(marker_line_width=0.5)

        fig.update_layout(
            xaxis_title=selected_feature,
            yaxis_title='Frekuensi',
            bargap=0.1,
            showlegend=True,
            margin=dict(t=50, b=40)
        )

        st.plotly_chart(fig, use_container_width=False, width=600, height=400)





elif st.session_state.menu == "Deskripsi Model":
    st.header("Deskripsi Model")
    st.write("Model yang digunakan:")
    st.write("""
    - **Support Vector Machine (SVM) manual** dengan parameter regulasi C (default 1.0), tanpa kernel (linear).
    - **Random Forest Classifier** dengan oversampling SMOTE untuk mengatasi ketidakseimbangan data.
    """)

    st.subheader("Support Vector Machine (SVM) Manual")
    st.write("""
    SVM adalah model klasifikasi yang mencari hyperplane terbaik untuk memisahkan dua kelas data secara linear.
    Model menggunakan bobot (`w`) dan bias (`b`) untuk menentukan posisi hyperplane.
    """)

    st.write("""
    Berikut penjelasan fungsi-fungsi utama yang digunakan untuk melatih dan memprediksi dengan SVM manual:

    **1. compute_class_weights(y):**
    - Menghitung bobot kelas berdasarkan frekuensi kelas agar model tidak bias pada kelas mayoritas.
    - Digunakan untuk mengatasi masalah _imbalanced dataset_ dengan memberikan bobot lebih besar pada kelas minoritas.

    **2. svm_train(X, y, class_weights, lr=0.001, epochs=1000, C=1.0):**
    - Melatih model SVM linear menggunakan _Stochastic Gradient Descent_ (SGD).
    - Fungsi loss yang digunakan adalah _hinge loss_ dengan regularisasi L2.
    - Bobot kelas (`class_weights`) membuat model lebih sensitif terhadap kelas minoritas.
    - Parameter `C` mengatur keseimbangan antara margin lebar dan kesalahan klasifikasi.
    - Di setiap epoch, bobot `w` dan bias `b` diperbarui berdasarkan apakah sampel sudah diklasifikasikan dengan margin cukup.

    **3. svm_predict(X, w, b):**
    - Memprediksi kelas berdasarkan posisi sampel terhadap hyperplane yang dibentuk oleh `w` dan `b`.
    - Jika nilai linear output ≥ 0 → kelas 1, jika < 0 → kelas -1.
    - Merupakan prediksi model SVM linear standar.
    """)

    st.code('''
    from collections import Counter
    import numpy as np

    def compute_class_weights(y):
        counts = Counter(y)
        total = len(y)
        class_weights = {label: total/count for label, count in counts.items()}
        return class_weights

    def svm_train(X, y, class_weights, lr=0.001, epochs=1000, C=1.0):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0

        for epoch in range(epochs):
            for idx in range(n_samples):
                xi = X[idx]
                yi = y[idx]
                weight = class_weights[yi]

                condition = yi * (np.dot(w, xi) + b)
                if condition >= 1:
                    w -= lr * 2 * w
                else:
                    w -= lr * (2 * w - C * weight * yi * xi)
                    b -= lr * (-C * weight * yi)
        return w, b

    def svm_predict(X, w, b):
        linear_output = np.dot(X, w) + b
        return np.where(linear_output >= 0, 1, -1)
        ''')
    
    st.markdown("""
    Random Forest adalah model _ensemble learning_ yang menggabungkan banyak pohon keputusan (_decision tree_) 
    untuk meningkatkan akurasi dan mengurangi overfitting. Model ini bekerja dengan prinsip _bagging_:
    melatih setiap pohon pada data yang dipilih secara acak (_bootstrap sampling_) dan menggabungkan hasilnya 
    melalui voting mayoritas.

    Implementasi manual ini terdiri dari dua bagian utama:
    - **DecisionTree**: membentuk satu pohon dengan memilih fitur terbaik menggunakan **information gain** (berbasis entropy).
    - **RandomForest**: membangun banyak pohon dengan data dan fitur acak, lalu hasil prediksi dikombinasikan.

    ### Kode Utama

    ```python
    class RandomForest:
        def fit(self, X, y):
            for _ in range(self.n_trees):
                tree = DecisionTree(...)
                X_sample, y_sample = self._bootstrap_samples(X, y)
                tree.fit(X_sample, y_sample)
                self.trees.append(tree)

        def predict(self, X):
            predictions = np.array([tree.predict(X) for tree in self.trees])
            tree_preds = np.swapaxes(predictions, 0, 1)
            return np.array([self._most_common_label(pred) for pred in tree_preds])""")







elif st.session_state.menu == "Performa Model":
    st.header("Performa Model")

    df, df_encoded, df_scaled, X, y, features, numerical_feat, categorical_feat, target = load_and_preprocess_data()

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    st.subheader("Support Vector Machine (SVM)")

    model_params = joblib.load('LinearSVM_Manual.joblib')
    w_loaded = model_params['w']
    b_loaded = model_params['b']

    # >= 0 → kelas 1, < 0 → kelas 0
    y_pred_svm = (np.dot(X_test, w_loaded) + b_loaded >= 0).astype(int)

    svm_report = classification_report(y_test, y_pred_svm, output_dict=True)
    svm_cm = confusion_matrix(y_test, y_pred_svm)

    st.write("**Classification Report (SVM):**")
    st.dataframe(pd.DataFrame(svm_report).transpose())
    st.write("**Confusion Matrix (SVM):**")
    st.write(svm_cm)

    import plotly.graph_objects as go
    labels = ["0", "1"]

    # SVM
    fig_svm_plotly = go.Figure(data=go.Heatmap(
        z=svm_cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        showscale=True,
        text=svm_cm,
        texttemplate="%{text}",
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Jumlah: %{z}<extra></extra>"
    ))

    fig_svm_plotly.update_layout(
        title="SVM Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        width=400,
        height=400,
        xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=labels),
        yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=labels, autorange='reversed')
    )

    st.plotly_chart(fig_svm_plotly, use_container_width=True)

    st.subheader("Random Forest Classification")

    rf_model = joblib.load('RandomForest_Manual.joblib')
    y_pred_rf = rf_model.predict(X_test)

    rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
    rf_cm = confusion_matrix(y_test, y_pred_rf)

    st.write("**Classification Report (Random Forest):**")
    st.dataframe(pd.DataFrame(rf_report).transpose())
    st.write("**Confusion Matrix (Random Forest):**")
    st.write(rf_cm)

    # Random Forest
    fig_rf_plotly = go.Figure(data=go.Heatmap(
        z=rf_cm,
        x=labels,
        y=labels,
        colorscale="Greens",
        showscale=True,
        text=rf_cm,
        texttemplate="%{text}",
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Jumlah: %{z}<extra></extra>"
    ))

    fig_rf_plotly.update_layout(
        title="Random Forest Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        width=400,
        height=400,
        xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=labels),
        yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=labels, autorange='reversed')
    )

    st.plotly_chart(fig_rf_plotly, use_container_width=True)





elif st.session_state.menu == "Input & Prediksi":
    import joblib
    st.header("Input & Prediksi Conversion")

    df, _, _, _, _, _, _, _, _ = load_and_preprocess_data()

    sample_raw = {
        "CampaignType": "Conversion",
        "Income": 30558,
        "AdSpend": 2076.535113910116,
        "ClickThroughRate": 0.0771545687008888,
        "ConversionRate": 0.0154771974827736,
        "WebsiteVisits": 9,
        "PagesPerVisit": 7.818844717795544,
        "TimeOnSite": 14.229981592378053,
        "EmailOpens": 11,
        "EmailClicks": 4,
        "PreviousPurchases": 2,
        "LoyaltyPoints": 951,
    }

    with st.form("input_form"):
        income = st.number_input("Income", min_value=1000, max_value=1000000, value=int(sample_raw["Income"]))
        ad_spend = st.number_input("AdSpend", min_value=0.0, max_value=100000.0, value=float(sample_raw["AdSpend"]))
        ctr = st.number_input("ClickThroughRate", min_value=0.0, max_value=1.0, value=float(sample_raw["ClickThroughRate"]), format="%.6f")
        conv_rate = st.number_input("ConversionRate", min_value=0.0, max_value=1.0, value=float(sample_raw["ConversionRate"]), format="%.6f")
        website_visits = st.number_input("WebsiteVisits", min_value=0, max_value=1000, value=int(sample_raw["WebsiteVisits"]))
        pages_per_visit = st.number_input("PagesPerVisit", min_value=0.0, max_value=50.0, value=float(sample_raw["PagesPerVisit"]))
        time_on_site = st.number_input("TimeOnSite", min_value=0.0, max_value=1000.0, value=float(sample_raw["TimeOnSite"]))
        email_opens = st.number_input("EmailOpens", min_value=0, max_value=100, value=int(sample_raw["EmailOpens"]))
        email_clicks = st.number_input("EmailClicks", min_value=0, max_value=100, value=int(sample_raw["EmailClicks"]))
        previous_purchases = st.number_input("PreviousPurchases", min_value=0, max_value=100, value=int(sample_raw["PreviousPurchases"]))
        loyalty_points = st.number_input("LoyaltyPoints", min_value=0, max_value=1000, value=int(sample_raw["LoyaltyPoints"]))
        campaign_type = st.selectbox("CampaignType", options=df['CampaignType'].unique(), index=list(df['CampaignType'].unique()).index(sample_raw["CampaignType"]))

        submitted = st.form_submit_button("Prediksi Conversion")

    if submitted:
        encoders = joblib.load('preprocessor.pkl')['encoders']
        scaler = joblib.load('preprocessor.pkl')['scaler']
        categorical_feat = joblib.load('preprocessor.pkl')['categorical_feat']
        numerical_feat = joblib.load('preprocessor.pkl')['numerical_feat']

        campaign_type_enc = encoders['CampaignType'].transform([campaign_type])[0]

        input_dict = {
            "CampaignType": campaign_type_enc,
            "Income": income,
            "AdSpend": ad_spend,
            "ClickThroughRate": ctr,
            "ConversionRate": conv_rate,
            "WebsiteVisits": website_visits,
            "PagesPerVisit": pages_per_visit,
            "TimeOnSite": time_on_site,
            "EmailOpens": email_opens,
            "EmailClicks": email_clicks,
            "PreviousPurchases": previous_purchases,
            "LoyaltyPoints": loyalty_points,
        }

        input_df = pd.DataFrame([input_dict])

        st.subheader("Input Data")
        st.dataframe(pd.DataFrame([{
            "CampaignType": campaign_type,
            "Income": income,
            "AdSpend": ad_spend,
            "ClickThroughRate": ctr,
            "ConversionRate": conv_rate,
            "WebsiteVisits": website_visits,
            "PagesPerVisit": pages_per_visit,
            "TimeOnSite": time_on_site,
            "EmailOpens": email_opens,
            "EmailClicks": email_clicks,
            "PreviousPurchases": previous_purchases,
            "LoyaltyPoints": loyalty_points,
        }]))

        input_df[numerical_feat] = scaler.transform(input_df[numerical_feat])

        X_input = input_df[categorical_feat + numerical_feat].values

        model_params = joblib.load('LinearSVM_Manual.joblib')
        w = model_params['w']
        b = model_params['b']

        y_pred_svm = (np.dot(X_input, w) + b >= 0).astype(int)[0]

        rf_model = joblib.load('RandomForest_Manual.joblib')
        y_pred_rf = rf_model.predict(X_input)[0]

        st.success(f"Prediksi Conversion (SVM): {'Yes' if y_pred_svm == 1 else 'No'}")
        st.success(f"Prediksi Conversion (Random Forest): {'Yes' if y_pred_rf == 1 else 'No'}")
else:
    st.write("Mulai aplikasi dengan memilih menu!")

st.markdown("---")
st.markdown("Developed by **Nadia, Rafly, Fadhil**")
