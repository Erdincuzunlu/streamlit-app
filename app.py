import streamlit as st
import librosa
import numpy as np
import joblib
from sqlalchemy import create_engine, Table, MetaData, Column, Integer, String, Date, insert
from datetime import datetime
import pandas as pd
import speech_recognition as sr

# 1. Tema ve Stil Ayarları
st.set_page_config(
    page_title="Kekemelik Tespit ve Analiz",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS ile arka plan resmi ekleme
page_bg_img = '''
<style>
body {
    background-image: url("https://via.placeholder.com/800x600");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
'''

# CSS'i Streamlit uygulamasına ekleme
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>Kekemelik Tespit ve Analiz Kaydı</h1>",
    unsafe_allow_html=True
)

# Fonksiyonlar
def stuttering_data_prep(df):


    # MFCC
    mfcc_mean_columns = [col for col in df.columns if col.startswith("mfcc_") & col.endswith("mean")]
    df["mfcc_mean_sum"] = df[mfcc_mean_columns].sum(axis=1)

    mfcc_std_columns = [col for col in df.columns if col.startswith("mfcc_") & col.endswith("std")]
    df["mfcc_std_sum"] = df[mfcc_std_columns].sum(axis=1)

    mfcc_columns = [col for col in df.columns if col.startswith("mfcc_") & col.endswith(("_mean", "_std"))]
    df["mfcc_sum"] = df[mfcc_columns].sum(axis=1)

    # Chroma
    chroma_mean_columns = [col for col in df.columns if col.startswith("chroma_") & col.endswith("mean")]
    df["chroma_mean_sum"] = df[chroma_mean_columns].sum(axis=1)

    chroma_std_columns = [col for col in df.columns if col.startswith("chroma_") & col.endswith("std")]
    df["chroma_std_sum"] = df[chroma_std_columns].sum(axis=1)

    df["rms_zcr_mean"] = (df["rms_mean"] + df["zcr_mean"])/2

    df["sc_sr_mean"] = (df["spectral_centroid_mean"] + df["spectral_rolloff_mean"])/2

    df["rms_std*sc_std_"] = df["rms_std"] * df["spectral_centroid_std"]

    df["tempo_+_zcr"] = df["tempo"] + df["zcr_mean"]

    df["mfcc_*_rms_mean"] = (df["mfcc_mean_sum"] * df["rms_mean"])/13

    df["chroma+rolloff"] = df[chroma_mean_columns].sum(axis=1) + df["spectral_rolloff_mean"]

    df["tempo_*_zcr"] = df["tempo"] + df["rms_mean"]

    df["sc_sr_chroma_mean"] = df["spectral_centroid_mean"] + df["spectral_rolloff_mean"] + df["chroma_mean_sum"]

    # Log Transformation
    log_cols = ["mfcc_1_std", "mfcc_3_std", "mfcc_5_std", "mfcc_7_std", "mfcc_10_std", "rms_mean", "rms_std", "zcr_mean", "zcr_std", "spectral_centroid_mean", "spectral_rolloff_mean", "spectral_centroid_std", "spectral_rolloff_std", "tempo", "zcr_tempo_ratio", "rms_tempo_ratio", "rms_energy_fluctuation" ]

    df[log_cols] = np.log1p(df[log_cols])

    return df


# Sidebar
with st.sidebar:
    st.header("Navigasyon")
    st.write("Buradan farklı bölümlere ulaşabilirsiniz.")
    st.markdown("---")
    st.write("Uygulama Hakkında:")
    st.write("Bu uygulama, ses dosyalarından kekemelik analizi yapmaktadır.")

# Model dosyasını yükleme
model_path = "ANN.pkl"
rf_model = joblib.load(model_path)

# Veritabanı bağlantısı ve tablo oluşturma
engine = create_engine("sqlite:///analysis_results.db")
metadata = MetaData()

results_table = Table(
    'analysis_results', metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String),
    Column('surname', String),
    Column('date', Date),
    Column('analysis_result', String),
    Column('kekemelik_count', Integer)
)

# Tablo oluşturulmamışsa oluştur
with engine.connect() as connection:
    if not engine.dialect.has_table(connection, "analysis_results"):
        metadata.create_all(engine)
        st.write("Tablo başarıyla oluşturuldu!")

# 3. Özellik Çıkarma Fonksiyonu
def extract_features(audio_segment, sr):
    """
    Ses segmentlerinden özellikler çıkarır.
    """
    mfccs = np.mean(librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio_segment, sr=sr).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=audio_segment).T, axis=0).flatten()
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio_segment).T, axis=0).flatten()
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_segment, sr=sr).T, axis=0).flatten()
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_segment, sr=sr).T, axis=0).flatten()

    features = np.concatenate([mfccs, chroma, rms, zcr, spectral_centroid, spectral_rolloff])

    if len(features) < 99:
        padding = np.zeros(99 - len(features))
        features = np.concatenate([features, padding])

    return features

# 4. Kullanıcı Girişleri
st.header("Danışan Bilgileri")
name = st.text_input("Ad:")
surname = st.text_input("Soyad:")
date = st.date_input("Tarih:", datetime.now().date())

# 5. Ortak Metin
st.subheader("Okumanız Gereken Metin")
st.write("Lütfen aşağıdaki metni sesli olarak okuyun:\n\n")
st.markdown(
    "<div style='background-color:#FFFFFF; padding:10px; border-radius:5px;'>"
    "<h3>Metin:</h3>"
    "<p style='color: #000000; font-style: italic;'>"
    "'Bugün dünyada her geçen gün gelişen teknolojiler, insanların yaşamını büyük ölçüde değiştiriyor. "
    "Özellikle yapay zeka ve makine öğrenimi gibi alanlar, çeşitli endüstrilerde devrim yaratmakta. "
    "Bu teknolojiler, insanların daha verimli çalışmasına, çeşitli görevleri daha hızlı ve doğru bir şekilde tamamlamasına olanak tanımaktadır. "
    "Ancak, teknolojinin hızlı gelişimi, bazı toplumsal zorlukları da beraberinde getirebilir. "
    "Bu nedenle, teknolojiye dair hem fırsatlar hem de riskler dikkatle değerlendirilmelidir.'</p>"
    "</div>",
    unsafe_allow_html=True
)

# 6. Ses Yükleme ve Analiz
uploaded_file = st.file_uploader("Ses dosyasını yükleyin", type=["wav", "mp3"])
if uploaded_file is not None:
    # Ses dosyasını işleme
    audio, sr = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file, format="audio/wav")
    st.write(f"Ses kaydı başarıyla yüklendi. Örnekleme oranı: {sr} Hz")

    # Ses dosyasını 3 saniyelik parçalara ayırma
    segment_duration = 3
    segment_samples = sr * segment_duration
    segments = [audio[i:i + segment_samples] for i in range(0, len(audio), segment_samples)]

    st.write(f"Toplam {len(segments)} segment oluşturuldu.")

    # Segmentlerin analiz edilmesi
    results = []
    for i, segment in enumerate(segments):
        if len(segment) < segment_samples:
            continue  # Eksik segmenti atla

        features = extract_features(segment, sr)
        features = stuttering_data_prep(features)
        features = features.reshape(1, -1)
        prediction = rf_model.predict(features)[0]
        results.append(prediction)

        st.write(f"Segment {i + 1}: {'Kekemelik' if prediction == 1 else 'Normal'}")

    # Sonuçları Özetleme
    total_kekemelik = sum(results)
    st.subheader("Sonuçlar")
    if total_kekemelik > 0:
        st.markdown(
            f"<div style='background-color:#FFCDD2; padding:10px; border-radius:5px;'>"
            f"<h3>Kekemelik Tespit Edildi</h3>"
            f"<p>Toplam {total_kekemelik} segmentte kekemelik tespit edilmiştir.</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background-color:#C8E6C9; padding:10px; border-radius:5px;'>"
            f"<h3>Normal</h3>"
            f"<p>Kekemelik tespit edilmemiştir.</p>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Veritabanına Kaydetme
    if st.button("Sonuçları Kaydet"):
        with engine.connect() as connection:
            insert_stmt = insert(results_table).values(
                name=name,
                surname=surname,
                date=date,
                analysis_result="Kekemelik Tespit Edildi" if total_kekemelik > 0 else "Normal",
                kekemelik_count=total_kekemelik
            )
            connection.execute(insert_stmt)
            st.success("Analiz sonuçları başarıyla kaydedildi!")

# 7. Tüm Kayıtları Görüntüleme
st.header("Tüm Kayıtlar")
with engine.connect() as connection:
    results = connection.execute(results_table.select()).fetchall()

if results:
    st.write("Aşağıda kayıtlı tüm analiz sonuçları bulunmaktadır:")
    st.dataframe(results)
else:
    st.write("Henüz bir kayıt bulunmamaktadır.")

# 8. Mikrofon ile Ses Kaydı Alma
def record_audio():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        st.write("Lütfen konuşmaya başlayın...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        st.write("Konuşma algılandı, şimdi metne dönüştürülüyor...")
        text = recognizer.recognize_google(audio, language='tr-TR')  # Türkçe dil desteği
        return text
    except sr.UnknownValueError:
        st.write("Ses tanınamadı. Lütfen tekrar deneyin.")
    except sr.RequestError:
        st.write("Google Speech API servisine ulaşılamadı.")
    return None

# Sesle konuşma özelliği
if st.button("Sesle Konuşun"):
    result = record_audio()
    if result:
        st.write("Metne dönüştürülen konuşma:")
        st.write(result)