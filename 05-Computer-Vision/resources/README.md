# 📚 Bölüm 5 Ek Kaynaklar

## Makaleler
- Krizhevsky, Sutskever, Hinton (2012) — [ImageNet Classification with Deep CNNs (AlexNet)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- LeCun et al. (1998) — [Gradient-Based Learning Applied to Document Recognition (LeNet)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- He et al. (2015) — [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)
- Dalal & Triggs (2005) — [Histograms of Oriented Gradients (HOG)](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
- Redmon et al. (2016) — [You Only Look Once (YOLO)](https://arxiv.org/abs/1506.02640)

## Videolar
- 3Blue1Brown — [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA)
- Stanford CS231n — [Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)

## Kopya Kağıdı: Evrişim Çıktı Boyutu Hesaplama

```
çıktı_boyutu = (girdi_boyutu - çekirdek_boyutu + 2*padding) / stride + 1
```

| Padding | Stride | Etki |
|---|---|---|
| 0 ("valid") | 1 | Çıktı, girdiden küçülür |
| (k-1)/2 ("same") | 1 | Çıktı, girdiyle aynı boyutta kalır |
| Herhangi | >1 | Çıktı, stride oranında küçülür |

## Kopya Kağıdı: Klasik CNN Mimarileri

| Mimari | Yıl | Önemi |
|---|---|---|
| LeNet-5 | 1998 | İlk başarılı CNN, el yazısı rakam tanıma |
| AlexNet | 2012 | ImageNet'i büyük farkla kazandı, derin öğrenme çağını başlattı |
| VGGNet | 2014 | Basit, tekrarlayan 3x3 çekirdek yapısı |
| ResNet | 2015 | "Residual connections" ile 100+ katmanlı ağları eğitilebilir kıldı |
