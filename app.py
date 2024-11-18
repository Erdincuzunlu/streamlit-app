import streamlit as st
import librosa
import numpy as np
import joblib
from sqlalchemy import create_engine, Table, MetaData, Column, Integer, String, Date, insert
from datetime import datetime

# Model ve veritabanı yükleme


model_path = "/app/voting_model.pkl"
rf_model = joblib.load(model_path)

# SQLite veritabanı bağlantısı
engine = create_engine("sqlite:///analysis_results.db")
metadata = MetaData()

# Tabloyu tanımlama ve kontrol
results_table = Table(
    'analysis_results', metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String),
    Column('surname', String),
    Column('date', Date),
    Column('analysis_result', String),
    Column('kekemelik_count', Integer)
)

# Tabloyu oluşturma
with engine.connect() as connection:
    if not engine.dialect.has_table(connection, "analysis_results"):
        metadata.create_all(engine)
        st.write("Tablo başarıyla oluşturuldu!")

# Özellik çıkarma fonksiyonu
def extract_features(audio_segment, sr):
    # Mevcut özellikleri çıkar
    mfccs = np.mean(librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio_segment, sr=sr).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=audio_segment).T, axis=0).flatten()  # 1D yapıya dönüştür
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio_segment).T, axis=0).flatten()  # 1D yapıya dönüştür
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_segment, sr=sr).T, axis=0).flatten()  # 1D yapıya dönüştür
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_segment, sr=sr).T, axis=0).flatten()  # 1D yapıya dönüştür

    # Özellikleri birleştir
    features = np.concatenate([mfccs, chroma, rms, zcr, spectral_centroid, spectral_rolloff])

    # Özellik sayısını 99'a tamamla
    if len(features) < 99:
        padding = np.zeros(99 - len(features))
        features = np.concatenate([features, padding])

    return features

# Streamlit başlık
st.title("Kekemelik Tespit ve Analiz Kaydı")

# Kullanıcıdan giriş alımı
st.header("Danışan Bilgileri")
name = st.text_input("Ad:")
surname = st.text_input("Soyad:")
date = st.date_input("Tarih:", datetime.now().date())

# Ses kaydı yükleme ve analiz
uploaded_file = st.file_uploader("Ses dosyasını yükleyin", type=["wav", "mp3"])
if uploaded_file is not None:
    # Ses dosyasını işleme
    audio, sr = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file, format="audio/wav")
    st.write(f"Ses kaydı başarıyla yüklendi. Örnekleme oranı: {sr} Hz")

    # Ses dosyasını 3 saniyelik parçalara ayırma
    segment_duration = 3  # saniye
    segment_samples = sr * segment_duration
    segments = [audio[i:i + segment_samples] for i in range(0, len(audio), segment_samples)]

    st.write(f"Toplam {len(segments)} segment oluşturuldu.")

    # Her bir segment için özellik çıkarma ve tahmin
    results = []
    for i, segment in enumerate(segments):
        if len(segment) < segment_samples:
            continue  # Eksik segmenti atla

        features = extract_features(segment, sr)
        features = features.reshape(1, -1)  # Model girişine uygun hale getir
        prediction = rf_model.predict(features)[0]
        results.append(prediction)

        st.write(f"Segment {i + 1}: {'Kekemelik' if prediction == 1 else 'Normal'}")

    # Sonuçları özetleme
    total_kekemelik = sum(results)
    st.subheader("Sonuçlar")
    st.write(f"Toplam {total_kekemelik} segmentte kekemelik tespit edildi.")

    # Veritabanına kaydetme
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

# Veritabanındaki verileri görüntüleme
st.header("Tüm Kayıtlar")
with engine.connect() as connection:
    results = connection.execute(results_table.select()).fetchall()

# Verileri tablo olarak gösterme
if results:
    st.write("Aşağıda kayıtlı tüm analiz sonuçları bulunmaktadır:")
    st.dataframe(results)
else:
    st.write("Henüz bir kayıt bulunmamaktadır.")