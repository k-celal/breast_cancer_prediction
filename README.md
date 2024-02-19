# breast_cancer_prediction

Bu proje, makine öğrenimi algoritmalarını kullanarak meme kanserini tahmin etmeyi amaçlamaktadır. Bu projede kullanılan veri seti, Meme Kanseri Wisconsin (Tanısal) Veri Seti'dir.

## Açıklama

Proje, meme kanseri tümörlerini malign veya benign olarak sınıflandırmak için K-En Yakın Komşu (KNN), Destek Vektör Makinesi (SVM) ve Naif Bayes gibi makine öğrenimi algoritmalarını kullanmaktadır. Veri ön işleme adımları, veriyi temizleme, tanı etiketlerini kodlama ve veri kümesini eğitim ve test setlerine bölmeyi içerir.

## Streamlit Entegrasyonu

Streamlit, etkileşimli web uygulamaları oluşturmak için kullanılan bir Python kütüphanesidir. Bu proje, farklı sınıflandırıcıları ve veri setlerini keşfetmek için kullanıcı dostu bir arayüz oluşturmak için Streamlit'i kullanır. Kullanıcılar, kenar çubuğundan bir veri seti ve sınıflandırıcı seçebilir, veri kümesinin özet istatistiklerini, korelasyon matrisini ve dağılım grafiğini görüntüleyebilir ve ızgara arama kullanarak optimal parametrelere sahip makine öğrenimi modellerini eğitebilirler.

## Kurulum

Projeyi yerel olarak çalıştırmak için aşağıdaki adımları izleyin:

1. Deposunu klonlayın:

```bash
git clone https://github.com/k-celal/breast-cancer-prediction
```

2. Gerekli bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

3. Streamlit uygulamasını çalıştırın:

```bash
streamlit run app.py
```

4. Uygulamaya tarayıcınızda `http://localhost:8501` adresinden erişin.

## Kullanım

1. Kenar çubuğundan veri setini (Meme Kanseri) ve sınıflandırıcıyı (KNN, SVM veya Naif Bayes) seçin.
2. Veri kümesinin özet istatistiklerini, korelasyon matrisini ve dağılım grafiğini inceleyin.
3. Kenar çubuğundaki kaydırıcıları kullanarak parametreleri ayarlayın (örneğin, SVM için C, KNN için K).
4. Makine öğrenimi modelini eğitin ve değerlendirme metriklerini (doğruluk, hassasiyet, duyarlılık, F1 skoru ve karışıklık matrisi) görüntüleyin.
5. İsteğe bağlı olarak, seçilen sınıflandırıcı için optimal parametreleri bulmak için ızgara araması yapın.

## Katkıda Bulunanlar

- [`Celal Karahan`](https://github.com/k-celal)

</Good coding with MuyuX 👨‍💻>