# kural_tabanli_siniflandirma
MIUUL Data Science and Machine Learning Bootcamp Projesi

#### İş Problemi
Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak
seviye tabanlı (level based) yeni müşteri tanımları (persona)
oluşturmak ve bu yeni müşteri tanımlarına göre segmentler
oluşturup bu segmentlere göre yeni gelebilecek müşterilerin
şirkete ortalama ne kadar kazandırabileceğini tahmin etmek
istemektedir.

Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek
kullanıcının ortalama ne kadar kazandırabileceği belirlenmek
isteniyor.

#### Veri Seti Hikâyesi
Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu
ürünleri satın alan kullanıcıların bazı demografik bilgilerini barındırmaktadır. Veri
seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı
tablo tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir
kullanıcı birden fazla alışveriş yapmış olabilir.

PRICE | SOURCE | SEX | COUNTRY | AGE
------|--------|-----|---------|----
39 | android | male | bra | 17
39 | android | male | bra | 17
49 | android | male | bra | 17
29 | android | male | tur | 17
49 | android | male | tur | 17

#### Değişkenler
PRICE – Müşterinin harcama tutarı
SOURCE – Müşterinin bağlandığı cihaz türü
SEX – Müşterinin cinsiyeti
COUNTRY – Müşterinin ülkesi
AGE – Müşterinin yaşı
