from sqlalchemy import create_engine, Column, Integer, String, Date, MetaData, Table
from sqlalchemy.exc import OperationalError

# SQLite veritabanı bağlantısı
engine = create_engine("sqlite:///analysis_results.db")
metadata = MetaData()

# Tablo tanımlama
results_table = Table(
    'analysis_results', metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String),
    Column('surname', String),
    Column('date', Date),
    Column('analysis_result', String),
    Column('kekemelik_count', Integer)
)

# Tabloyu kontrol ve oluşturma
try:
    with engine.connect() as connection:
        if not engine.dialect.has_table(connection, 'analysis_results'):
            metadata.create_all(engine)
            print("Veritabanı ve tablo başarıyla oluşturuldu!")
        else:
            print("Tablo zaten mevcut!")
except OperationalError as e:
    print("Veritabanı bağlantısında bir sorun oluştu:", str(e))