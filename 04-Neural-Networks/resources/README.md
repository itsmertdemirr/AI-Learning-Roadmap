# 📚 Bölüm 4 Ek Kaynaklar

## Makaleler
- Glorot & Bengio (2010) — [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)
- He et al. (2015) — [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852)
- Srivastava et al. (2014) — [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)
- Ioffe & Szegedy (2015) — [Batch Normalization](https://arxiv.org/abs/1502.03167)
- Kingma & Ba (2014) — [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

## Videolar
- 3Blue1Brown — [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- deeplearning.ai — [Improving Deep Neural Networks](https://www.coursera.org/learn/deep-neural-network) (Andrew Ng)

## Kopya Kağıdı: Hangi Başlatmayı Ne Zaman Kullanmalı?

| Aktivasyon | Önerilen Başlatma |
|---|---|
| Sigmoid, Tanh | Xavier / Glorot |
| ReLU, Leaky ReLU, ELU | He |
| SELU | LeCun |

## Kopya Kağıdı: Aşırı Öğrenme Belirtileri ve Çözümleri

| Belirti | Olası Çözüm |
|---|---|
| Eğitim kaybı çok düşük, doğrulama kaybı yüksek | Dropout, L2, daha az parametre |
| Doğrulama kaybı bir noktadan sonra artmaya başlıyor | Early Stopping |
| Eğitim kararsız/salınımlı | Öğrenme oranını düşür, Batch Norm ekle |
| Eğitim çok yavaş ilerliyor | Öğrenme oranını artır, He/Xavier başlatma kullan |
