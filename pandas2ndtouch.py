from başlangıç import seaborn as sns, pandas as pd

# Dosyanın tam yolunu belirtin
df = sns.load_dataset("titanic")

df.head()

df.tail()

df.shape

df.info()

df.columns

df.index

df.describe().T

df.isnull().values.any() ### en az 1 tane eksik değer vardır...

df.isnull().sum()      ### değişkenler içinde ki eksik verileri veriyor...

df["sex"].head()

df["sex"].value_counts() ###

##### pandas'ta seçim işlemleri ( selection in pandas )

df.index

df[0:13]## belirli bir aralıkta yapılan seçinm işlemi ''slice''

df.drop(0 ,axis=0).head()   ### satırlardan bir index silme işlemi.

delete_indexes = [1, 3, 5, 7] ### birden fazla index silmek istersek eğer

df.drop(delete_indexes, axis=0).head(10)

#df.drop(delete_indexes, axis=0, inplace=True) # yapılan işlemleri kalıcı olarak siler...

#### değişkeni indexe çevirmek ********

df["age"].head()
df.age.head()

df.index

df.index

df.index = df["age"]



df.info

df.drop("age", axis=1, inplace=True)

#### indexi değişkene çevirmek

df.index

df["age"] = df.index
df.head()

df.drop("age", axis=1, inplace=True)
df.head()
df.reset_index().head()

df = df.reset_index().head()
df.head()

#### işlevsel çözümdür... Tam olarak değişkendeki bir elemanı indexe gönderdik daha sonra da indexteki değişkeni kalıcı olarak
## yerine şutladık...


#### Değişkenler üzerinde işlemler...

import numpy as np
import başlangıç.seaborn as sns
pd.set_option('display.max_rows', None)
df = sns.load_dataset("titanic")

df.head()

"age" in df

df["age"].head()

type(df["age"].head())

df[["age", "alive"]].head()

col_names = ["age", "adult_male", "alive"]
df[col_names].head()

### veri setine yeni bir değişken eklemek....

df["age2"] = df["age"] **  2
df.head()

df["age3"] = df["age"] / df["age2"]
df.head()

#### veri silmek istersek ne yapacağız ?

df.drop("age3", axis=1).head()

#### birden fazla değişken silmek istersek o zaman yapacağımız işlemler....


df.drop(col_names, axis=1).head()

### belirli bir string ifade barındıran ifadeleri silmek istersen o zaman ne yapacağız....

df.loc[:, ~df.columns.str.contains("age")].head()  #### ~~~~~ işareti ilgili df içinde "age" yazmayan diğer tüm ifadeleri çağırmak içindir

df.loc[:, df.columns.str.contains("age")].head() ### burada da ilgili df içindeki "age" olanları getir anlamı vardır...



#### iloc ve loc integer base selecetipon .... label base selecetion

df.iloc[0:3] ### ''' e kadar seçer..
df.iloc[0, 0]

df.head()

####
df.loc[0:3]  ### 0-1-2-3 alır...

df.iloc[0:3, "age"]   #### hata alırız  .çünkü iloc integer base olduğu için ama bu işlemi loc ile yapabiliriz

df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]

#### koşullu seçim işlemleri ( conditional selection )

#bu veri setinde 50 yaşından büyük olanlara erişmek istiyorum mesela....

df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count()  ## 50 den büyük age değişkenini sayalım dedik ve 64 sonucunu aldık...

#### class sınıfını merak ediyoruz mesela... bu değişkeni seçme...

df.loc[df["age"] > 50, "class"].head()
df.loc[df["age"] > 50, ["class", "age"]].head()      ### yaşı 50 den büyük olan ve class age değişkenkerini getirdik.
                       ## burada birden fazla değişkene bakmak istiyorsak eğer bir liste oluşturmalıyız.


#### Q1 .... yaşı 50 den büyük olan ve cinsiyeti erkek olanlara erişmek istesek ne yapacağız....

### DİKKAT!!! BURAADA BİR DEN FAZLA KOŞUL GİRECEKSEK EĞER () İÇİNE ALMAK ZORUNDAYIZ..

df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head()

### aynı anda iki koşul ve iki değişken seçtik

df_new = df.loc[(df["age"] > 50)
       & (df["sex"] == "male")
       & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
       ["age", "class", "embark_town"]].head()


### toplulaştırma ve gruplama ( Aggregation and Grouping)

### count()
## first()
# Last()
# Mean()
# Median..... ETC....

df.head()


df["age"].mean()

df.groupby("sex")["age"].mean()
### cinsiyete göre gruplama...

df.groupby("sex").agg({"age": "mean"})
### groupby alınan cinsiyet değişkeni üzerinde yaş değişkenlerine göre ortalama aldık

df.groupby("sex").agg({"age": ["mean", "sum"]})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})

(df.groupby(["sex", "embark_town", "class"]).agg
 ({"age": ["mean"],
"survived": "mean",
   "sex": "count"}))


#### pivot table

df.pivot_table("survived", "sex", "embarked")

df.pivot_table("survived", "sex", "embarked", aggfunc="std")

df.pivot_table("survived", "sex", ["embarked", "class"])

df.head()
#### yaş değişkeni kategorik değişkene çevirme... ******  tanımlayamıyorsan qcut foksiyonu kullanılır.*******

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])

df.head()


df.pivot_table("survived", "sex", "new_age")

### boyut eklemek istersek....

df.pivot_table("survived", "sex", ["new_age", "class"])



df.head()
pd.set_option('display.max_columns', None)
df.head()
pd.set_option('display.width', 500)
df.head()


#### Apply ve Lambda Kullan at fonksiyonlarıı....


#### bir DF satır ve sütunlarda istediğimiz fonk. uygularız... Lambda ile de yapabiliriz ama kullan at şeklindedir. Fonk tanımalamadan

df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 5
df.head()
pd.set_option('display.max_columns', 10)
df.head()


(df["age"]/10).head()
df["age2"]/10
df["age3"]/10

#Bunun daha kısa yazmak için... Döngü ile örneğin...

### değişkenlere bir fonksiyon uygulayacağız...

for col in df.columns:
    if "age" in col:
        df[col] = df[col] / 10
        print(col)

df[["age", "age2", "age3"]].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

print(df.loc[:, df.columns.str.contains("age")].dtypes)

df.drop("new_age", axis=1, inplace=True)
df.head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

#### standartlaştırma fonksiyonu kullanacağız.....

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()


def standart_scaler(col_name):
    return(col_name - col_name.mean() / col_name.std())

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)()

df.head()

##### birleştirme ( Join ) işlemleriiii

m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2], ignore_index=True)



