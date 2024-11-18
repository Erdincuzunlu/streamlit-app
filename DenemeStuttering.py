import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_excel("/Users/erdinc/PycharmProjects/pythonProject3/22/stuttering_dataset.xlsx")
print(df.head())  # İlk birkaç satırı görüntüleyerek kontrol edelim

# Kategorik ve sayısal sütunları ayırmak için fonksiyon
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

def stutering_data_prep(df):

    # MFCC
    mfcc_mean_columns = [col for col in df.columns if col.startswith("mfcc_") and col.endswith("mean")]
    df["mfcc_mean_sum"] = df[mfcc_mean_columns].sum(axis=1)

    mfcc_std_columns = [col for col in df.columns if col.startswith("mfcc_") and col.endswith("std")]
    df["mfcc_std_sum"] = df[mfcc_std_columns].sum(axis=1)

    mfcc_columns = [col for col in df.columns if col.startswith("mfcc_") and col.endswith(("_mean", "_std"))]
    df["mfcc_sum"] = df[mfcc_columns].sum(axis=1)

    # Chroma
    chroma_mean_columns = [col for col in df.columns if col.startswith("chroma_") and col.endswith("mean")]
    df["chroma_mean_sum"] = df[chroma_mean_columns].sum(axis=1)

    chroma_std_columns = [col for col in df.columns if col.startswith("chroma_") and col.endswith("std")]
    df["chroma_std_sum"] = df[chroma_std_columns].sum(axis=1)

    df["rms_zcr_mean"] = (df["rms_mean"] + df["zcr_mean"]) / 2
    df["sc_sr_mean"] = (df["spectral_centroid_mean"] + df["spectral_rolloff_mean"]) / 2
    df["rms_std*sc_std_"] = df["rms_std"] * df["spectral_centroid_std"]
    df["tempo_+_zcr"] = df["tempo"] + df["zcr_mean"]
    df["mfcc_*_rms_mean"] = (df["mfcc_mean_sum"] * df["rms_mean"]) / 13
    df["chroma+rolloff"] = df[chroma_mean_columns].sum(axis=1) + df["spectral_rolloff_mean"]
    df["tempo_*_zcr"] = df["tempo"] + df["rms_mean"]
    df["sc_sr_chroma_mean"] = df["spectral_centroid_mean"] + df["spectral_rolloff_mean"] + df["chroma_mean_sum"]

    df["mfcc_mean_avg"] = df[mfcc_mean_columns].mean(axis=1)
    df["mfcc_mean_var"] = df[mfcc_std_columns].mean(axis=1)

    # Additional feature transformations
    df["spectral_centroid_var"] = df["spectral_centroid_std"] ** 2
    df["spectral_rolloff_var"] = df["spectral_rolloff_std"] ** 2

    # Data Preparation
    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

    y = df["stutering"]
    X = df.drop(["stutering"], axis=1)

    return X, y

# Veri işleme fonksiyonu
def stutering_data_prep(df):
    mfcc_mean_columns = [col for col in df.columns if col.startswith("mfcc_") & col.endswith("mean")]
    df["mfcc_mean_sum"] = df[mfcc_mean_columns].sum(axis=1)

    mfcc_std_columns = [col for col in df.columns if col.startswith("mfcc_") & col.endswith("std")]
    df["mfcc_std_sum"] = df[mfcc_std_columns].sum(axis=1)

    mfcc_columns = [col for col in df.columns if col.startswith("mfcc_") & col.endswith(("_mean", "_std"))]
    df["mfcc_sum"] = df[mfcc_columns].sum(axis=1)

    chroma_mean_columns = [col for col in df.columns if col.startswith("chroma_") & col.endswith("mean")]
    df["chroma_mean_sum"] = df[chroma_mean_columns].sum(axis=1)

    chroma_std_columns = [col for col in df.columns if col.startswith("chroma_") & col.endswith("std")]
    df["chroma_std_sum"] = df[chroma_std_columns].sum(axis=1)

    df["rms_zcr_mean"] = (df["rms_mean"] + df["zcr_mean"]) / 2
    df["sc_sr_mean"] = (df["spectral_centroid_mean"] + df["spectral_rolloff_mean"]) / 2
    df["rms_std*sc_std_"] = df["rms_std"] * df["spectral_centroid_std"]
    df["tempo_+_zcr"] = df["tempo"] + df["zcr_mean"]
    df["mfcc_*_rms_mean"] = (df["mfcc_mean_sum"] * df["rms_mean"]) / 13
    df["chroma+rolloff"] = df[chroma_mean_columns].sum(axis=1) + df["spectral_rolloff_mean"]
    df["tempo_*_zcr"] = df["tempo"] + df["rms_mean"]
    df["sc_sr_chroma_mean"] = df["spectral_centroid_mean"] + df["spectral_rolloff_mean"] + df["chroma_mean_sum"]

    # Eksik sütunlara göre spectral_energy_variation sütununu ekleyin
    if 'spectral_centroid_var' in df.columns and 'spectral_rolloff_var' in df.columns:
        df["spectral_energy_variation"] = df["spectral_centroid_var"] + df["spectral_rolloff_var"]
    else:
        df["spectral_energy_variation"] = df["spectral_rolloff_std"]  # Alternatif hesaplama yöntemi

    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

    # Sayısal sütunları standartlaştırın
    X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

    y = df["stutering"]
    X = df.drop(["stutering"], axis=1)

    return X, y

# Veri hazırlık fonksiyonu
def stutering_data_prep(df):
    # Örnek veri hazırlık adımları
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop(["stutering"], axis=1))
    y = df["stutering"]
    return X, y

# Veriyi hazırla
X, y = stutering_data_prep(df)

from sklearn.model_selection import train_test_split
# Eğitim ve test verisini ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# En iyi parametreleri kullanarak modelleri tanımlayalım
best_rf = RandomForestClassifier(max_depth=None, max_features=7, min_samples_split=20, n_estimators=200)

best_lgbm = LGBMClassifier(colsample_bytree=0.7, learning_rate=0.01, n_estimators=500)

best_xgb = XGBClassifier(colsample_bytree=0.5, learning_rate=0.1, max_depth=8, n_estimators=200)


# Modelleri sırayla optimize eden fonksiyon
def model_optimization(model, params, X, y):
    print(f"Optimizing {model.__class__.__name__}...")
    grid_search = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    print(f"Best params for {model.__class__.__name__}: {grid_search.best_params_}\n")
    return grid_search.best_estimator_

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# Model değerlendirme fonksiyonu
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred):.4f}")
    print(f"Recall: {recall_score(y, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

# Random Forest Performansını Değerlendirelim
print("Random Forest Performance:")
best_rf.fit(X_train, y_train)
evaluate_model(best_rf, X_test, y_test)

# LightGBM Performansını Değerlendirelim
print("\nLightGBM Performance:")
best_lgbm.fit(X_train, y_train)
evaluate_model(best_lgbm, X_test, y_test)

# XGBoost Performansını Değerlendirelim
print("\nXGBoost Performance:")
best_xgb.fit(X_train, y_train)
evaluate_model(best_xgb, X_test, y_test)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Modelleri ve isimlerini tanımlayalım
modeller = [best_rf, best_lgbm, best_xgb]
model_isimleri = ['Random Forest', 'LightGBM', 'XGBoost']


import joblib

# Model parametrelerini kullanarak modelleri eğitelim
best_rf = RandomForestClassifier(max_depth=None, max_features=7, min_samples_split=20, n_estimators=200)
best_lgbm = LGBMClassifier(colsample_bytree=0.7, learning_rate=0.01, n_estimators=500)
best_xgb = XGBClassifier(colsample_bytree=0.5, learning_rate=0.1, max_depth=8, n_estimators=200)

# Eğitim işlemi
best_rf.fit(X_train, y_train)
best_lgbm.fit(X_train, y_train)
best_xgb.fit(X_train, y_train)

# Modelleri kaydedelim
joblib.dump(best_rf, '/Users/erdinc/PycharmProjects/pythonProject3/22/random_forest_model_v3.pkl')
joblib.dump(best_lgbm, '/Users/erdinc/PycharmProjects/pythonProject3/22/lightgbm_model_v3.pkl')
joblib.dump(best_xgb, '/Users/erdinc/PycharmProjects/pythonProject3/22/xgboost_model_v3.pkll')

print("Modeller başarıyla kaydedildi.")

joblib.dump(best_rf, '/Users/erdinc/PycharmProjects/pythonProject3/DONEMSONU/random_forest_model_v3.pkl')

rf_model = joblib.load("/Users/erdinc/PycharmProjects/pythonProject3/DONEMSONU/random_forest_model_v3.pkl")


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

# Verinizi hazırlayın (örneğin df)
df = pd.read_excel("/Users/erdinc/PycharmProjects/pythonProject3/22/stuttering_dataset.xlsx")
X, y = stutering_data_prep(df)

# Veriyi eğitim ve test verisine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturun ve eğitin
rf_model = RandomForestClassifier(max_depth=None, max_features=7, min_samples_split=20, n_estimators=200)
rf_model.fit(X_train, y_train)

# Modeli kaydedin
joblib.dump(rf_model, '/Users/erdinc/PycharmProjects/pythonProject3/22/random_forest_model_v3.pkl')

print("Model başarıyla kaydedildi.")

