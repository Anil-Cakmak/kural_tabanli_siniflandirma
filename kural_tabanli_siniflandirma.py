import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ
# persona.csv dosyasını okutulması ve veri seti ile ilgili genel bilgileri inceleme.
df = pd.read_csv(r"persona.csv")


def check_df(dataframe, head=5, quantiles=(0, 0.05, 0.50, 0.95, 0.99, 1)):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Index ####################")
    print(dataframe.index)
    print("##################### Quantiles #####################")
    print(dataframe.describe(list(quantiles)).T)


check_df(df)


# Nümerik ve kategorik değişkenlerin tespit edilmesi.
def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """

    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes not in ["category", "object", "bool"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Kategorik Değiken Analizi


def cat_summary(dataframe, col_name, plot=False):

    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)


# Nümerik Değişken Analizi
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col)

# Hedef Değişken Analizi

target_col = "PRICE"


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    if col != target_col:
        target_summary_with_cat(df, target_col, col)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    if col != target_col:
        target_summary_with_num(df, target_col, col)

# MÜŞTERİ SEGMENTASYONU

df_group = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})

agg_df = df_group.sort_values(by="PRICE", ascending=False)

agg_df.reset_index(inplace=True)

cut_lst = [agg_df.AGE.min()-1, 18, 23, 30, 40, agg_df.AGE.max()]
agg_df["AGE_CAT"] = pd.cut(agg_df.AGE, cut_lst,
                           labels=[f"{cut_lst[i]+1}_{cut_lst[i+1]}" for i in range(len(cut_lst)-1)])


agg_df["customers_level_based"] = ["_".join(i).upper() for i in agg_df.drop(["AGE", "PRICE"], axis=1).values]
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
segment_descr = agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})

# YENİ MÜŞTERİ MUHTEMEL GETİRİ TAHMİNİ


def olasi_kazanc(new_user):
    user_segment = agg_df.loc[new_user, "SEGMENT"]
    getiri = segment_descr.loc[user_segment, ("PRICE", "mean")]
    return user_segment, getiri


# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
olasi_kazanc("TUR_ANDROID_FEMALE_31_40")

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
olasi_kazanc("FRA_IOS_FEMALE_31_40")
