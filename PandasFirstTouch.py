#### Pandas Series


var =pd.Series([10, 20, 30, 40, 50])


type(var)
var.dtypes ### tipini verir.

var.index

var.size ## içinde bulunan eleman sayısı.

var.ndim ## boyutuna bakmak istersek kullanacağımız fonksiyon.

### pandas serileri tek boyutludur indexleri vardır..

var.values ## içindeki değerlere erişmek istersek...

type(var.values)

var.head()
var.tail(3)### sondan 3 indexi getirir....


#### pandas ile birlikte  Dış kaynaklı verileri okumak istersek..


import başlangıç.pandas as pd

# Dosyanın tam yolunu belirtin
df = pd.read_csv('/Users/erdinc/Desktop/olympics2024.csv')

# İlk birkaç satırı görüntüleyin
print(df.head())



olympics_path = '/Users/erdinc/Desktop/olympics2024.csv'

df1 = pd.read_csv(olympics_path)

print(df1.head())

print(df1.info())##### data frame'in yapısını kontrol eder.

print(df1.describe()) #### data frame'in temel istatistiklerini görüntüler.
import matplotlib.pyplot as plt
from başlangıç import seaborn as sns, pandas as pd

sns.set(style="whitegrid")   ## grafiklerin daha iyi görünmesi için.

print(df1.head())

fig, ax = plt.subplots(figsize=(10, 6))

df1[['Gold', 'Silver', 'Bronze']].mean().plot(kind='bar', ax=ax, color=['gold', 'silver', 'brown'])

ax.set_title('Ortalama Madalya Sayıları')
ax.set_ylabel('Ortalama Sayı')
ax.set_xlabel('Madalya Türü')

plt.show()

plt.savefig('ortalama_madalya_sayilari.pdf', format='pdf')


plt.figure(figsize=(10, 6))
sns.histplot(df1['Total'], bins=20, kde=True, color='blue')
plt.title('Toplam Madalya Sayısının Dağılımı')
plt.xlabel('Toplam Madalya Sayısı')
plt.ylabel('Frekans')

plt.show()

import matplotlib.pyplot as plt
import başlangıç.seaborn as sns
import başlangıç.pandas as pd

# CSV dosyasını yükleyin
olympics_path = '/Users/erdinc/Desktop/olympics2024.csv'
df1 = pd.read_csv(olympics_path)

# Altın madalya sayısına göre en yüksek 10 ülkeyi seçin
top_10_countries = df1.sort_values('Gold', ascending=False).head(10)

# Figür ve alt grafik oluşturma
plt.figure(figsize=(12, 8))

# Bar grafiği oluşturma
sns.barplot(x='Country', y='Gold', data=top_10_countries, palette='viridis')

# Başlık ve etiketler
plt.title('En Çok Altın Madalya Kazanan İlk 10 Ülke')
plt.xlabel('Ülke')
plt.ylabel('Altın Madalya Sayısı')

# Ülke isimlerini daha okunabilir hale getirmek için döndürme
plt.xticks(rotation=45)

# Grafiği gösterme
plt.show()


import matplotlib.pyplot as plt
import başlangıç.seaborn as sns
import başlangıç.pandas as pd

# CSV dosyasını yükleyin
olympics_path = '/Users/erdinc/Desktop/olympics2024.csv'
df1 = pd.read_csv(olympics_path)

# Altın madalya sayısına göre en yüksek 10 ülkeyi seçin
top_10_countries = df1.sort_values('Gold', ascending=False).head(10)

# Figür ve alt grafik oluşturma
plt.figure(figsize=(12, 8))

# Kendi renklerinizi belirleyin
colors = ['gold', 'orange', 'red', 'purple', 'blue', 'green', 'cyan', 'magenta', 'pink', 'brown']

# Bar grafiği oluşturma
sns.barplot(x='Country', y='Gold', data=top_10_countries, palette=colors)

# Başlık ve etiketler
plt.title('En Çok Altın Madalya Kazanan İlk 10 Ülke')
plt.xlabel('Ülke')
plt.ylabel('Altın Madalya Sayısı')

# Ülke isimlerini daha okunabilir hale getirmek için döndürme
plt.xticks(rotation=45)

# Grafiği gösterme
plt.show()

plt.savefig('en_cok_altin_madalya_kazanan_ilk_10_ulke.png', format='pdf')


import matplotlib.pyplot as plt
import başlangıç.seaborn as sns
import başlangıç.pandas as pd

# CSV dosyasını yükleyin
olympics_path = '/Users/erdinc/Desktop/olympics2024.csv'
df1 = pd.read_csv(olympics_path)

# Figür ve alt grafik oluşturma
plt.figure(figsize=(12, 8))

# Dağılım grafiği oluşturma
sns.scatterplot(x='Total', y='Gold', data=df1)

# Başlık ve etiketler
plt.title('Toplam Madalya ile Altın Madalya Arasındaki İlişki')
plt.xlabel('Toplam Madalya Sayısı')
plt.ylabel('Altın Madalya Sayısı')

# Grafiği gösterme
plt.show()

import matplotlib.pyplot as plt
import başlangıç.seaborn as sns
import başlangıç.pandas as pd

# CSV dosyasını yükleyin
olympics_path = '/Users/erdinc/Desktop/olympics2024.csv'
df1 = pd.read_csv(olympics_path)

# Figür ve alt grafik oluşturma
plt.figure(figsize=(12, 8))

# Kutu grafiği oluşturma
sns.boxplot(x='Gold', data=df1)

# Başlık ve etiketler
plt.title('Altın Madalya Sayısının Dağılımı')
plt.xlabel('Altın Madalya Sayısı')

# Grafiği gösterme
plt.show()


import matplotlib.pyplot as plt
import başlangıç.pandas as pd

# CSV dosyasını yükleyin
olympics_path = '/Users/erdinc/Desktop/olympics2024.csv'
df1 = pd.read_csv(olympics_path)

# Toplam madalyaları hesaplayın
total_medals = df1[['Gold', 'Silver', 'Bronze']].sum()

# Figür ve pasta grafiği oluşturma
plt.figure(figsize=(8, 8))

# Pasta grafiği oluşturma
plt.pie(total_medals, labels=total_medals.index, autopct='%1.1f%%', colors=['gold', 'silver', 'brown'])

# Başlık
plt.title('Madalya Türlerinin Oranı')



import matplotlib.pyplot as plt
import başlangıç.seaborn as sns
import başlangıç.pandas as pd

# CSV dosyasını yükleyin
olympics_path = '/Users/erdinc/Desktop/olympics2024.csv'
df1 = pd.read_csv(olympics_path)

# Figür ve histogram oluşturma
plt.figure(figsize=(12, 8))

# Histogram oluşturma
sns.histplot(df1['Total'], bins=20, kde=True, color='blue')

# Başlık ve etiketler
plt.title('Toplam Madalya Sayısının Dağılımı')
plt.xlabel('Toplam Madalya Sayısı')
plt.ylabel('Frekans')

# Grafiği gösterme
plt.show()




import matplotlib.pyplot as plt
import başlangıç.seaborn as sns
import başlangıç.pandas as pd

# CSV dosyasını yükleyin
olympics_path = '/Users/erdinc/Desktop/olympics2024.csv'
df1 = pd.read_csv(olympics_path)

# Korelasyon matrisini hesaplayın
corr = df1[['Gold', 'Silver', 'Bronze', 'Total']].corr()

# Figür ve ısı haritası oluşturma
plt.figure(figsize=(10, 8))

# Isı haritası oluşturma
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')

# Başlık
plt.title('Madalya Türleri Arasındaki Korelasyon')

# Grafiği gösterme
plt.show()