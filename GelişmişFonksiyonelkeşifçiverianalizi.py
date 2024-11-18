#### Gelimiş Fonksiyonel Keşifçi Veri analizi
##

###### 1. Genel resim...
### 2. Kategorik değişken analizi ( analysis of categorical veriables)
####3. sayısal değişken analizi ( analysis of Numerical veriables)
####4. Hedef değişken Analizi ( analysis of Target Veriable)
### 5. Korelasyon analizi ( analaysis of Correlation)


from başlangıç import seaborn as sns, pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df.tail()
df.describe()
df.shape
df.info()
df.index
df.isnull().sum()
df.isnull().values.any()


def check_df(dataframe, head=5):
    print("######### shape #######")
    print(dataframe.shape)
    print("######### Types #########")
    print(dataframe.dtypes)
    print("######### Tail ##########")
    print(dataframe.tail(head))
    print("######## NA ##############")
    print(dataframe.isna().sum())
    print("############ Quantiles ##########")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]))

check_df(df)

df = sns.load_dataset("tips")
check_df(df)


import başlangıç.pandas as pd
import başlangıç.seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["embarked"].value_counts()
df["sex"].unique()
df["class"].nunique()
df.head()






str(df["sex"].dtype) in ["object"]
str(df["alone"].dtype) in ["bool"]

df["survived"].value_counts()

cat_cols = [col for col in df.columns if str(df[col].dtype) in ["object", "category", "bool"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]


cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["int", "float"]]


cat_cols = cat_cols + num_but_cat + cat_but_car

cat_cols = [col for col in cat_cols if col not in cat_but_car]


df[cat_cols]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_but_car]

df["survived"].value_counts()
100 * df["survived"].value_counts() / len(df)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * df["survived"].value_counts() / len(dataframe) }))

    print("############")


cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)

df["adult_male"].astype(int)


import başlangıç.pandas as pd
import başlangıç.seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df[["age", "fare"]].describe().T

[col for col in df.columns if df[col].dtypes in ["int", "float"]]


import başlangıç.pandas as pd
import başlangıç.seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

### docstring

df.groupby("sex")["survived"].mean()

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))


target_summary_with_cat(df, "survived", "pclass")


for col in cat_cols:
    target_summary_with_cat(df,"survived", col)




