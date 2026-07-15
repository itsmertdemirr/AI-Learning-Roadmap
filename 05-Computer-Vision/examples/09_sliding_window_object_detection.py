"""
09 - Kaydırmalı Pencere (Sliding Window) ile Basit Nesne Tespiti
=====================================================================

Modern nesne tespit sistemleri (YOLO, Faster R-CNN) çok daha
gelişmiştir, ama hepsinin temelinde yatan fikir buradan başlar: bir
görüntüyü küçük pencerelere bölüp, HER pencerede "burada aradığım nesne
var mı?" sorusunu sormak. Bu örnek, şablon eşleştirme (template
matching) kullanarak basit bir kaydırmalı pencere dedektörü uygular.

Gereksinimler: numpy, matplotlib, scikit-image
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def create_synthetic_scene(size: int = 100, seed: int = 3) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """İçinde birkaç 'X' işareti bulunan sentetik bir sahne (görüntü) üretir.
    Gerçek konumları da (doğrulama için) döndürür."""
    rng = np.random.default_rng(seed)
    scene = rng.normal(0.1, 0.03, (size, size))  # hafif gürültülü arka plan
    scene = np.clip(scene, 0, 1)

    true_positions: list[tuple[int, int]] = []
    template_size = 10
    min_separation = template_size * 2  # örtüşmeyi önlemek için minimum mesafe

    attempts = 0
    while len(true_positions) < 3 and attempts < 200:
        attempts += 1
        top = rng.integers(0, size - template_size)
        left = rng.integers(0, size - template_size)
        if all(np.hypot(top - t, left - left_pos) >= min_separation for t, left_pos in true_positions):
            true_positions.append((int(top), int(left)))

    for top, left in true_positions:
        # Bir "X" deseni çiz (köşegenler)
        for i in range(template_size):
            scene[top + i, left + i] = 1.0
            scene[top + i, left + template_size - 1 - i] = 1.0

    return scene, true_positions


def create_template(size: int = 10) -> np.ndarray:
    """Aranacak 'X' şablonunu oluşturur."""
    template = np.zeros((size, size))
    for i in range(size):
        template[i, i] = 1.0
        template[i, size - 1 - i] = 1.0
    return template


def sliding_window_detect(
    scene: np.ndarray, template: np.ndarray, stride: int = 2, threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    """Kaydırmalı pencere ile şablon eşleştirmesi uygular.

    Her pencere konumunda, o bölge ile şablon arasındaki normalize
    edilmiş korelasyonu (benzerlik skorunu) hesaplar.

    Returns:
        (satır, sütun, skor) üçlülerinden oluşan, eşik değerini aşan
        tespit listesi.
    """
    t_h, t_w = template.shape
    s_h, s_w = scene.shape
    detections = []

    template_flat = template.flatten()
    template_norm = template_flat / (np.linalg.norm(template_flat) + 1e-8)

    for i in range(0, s_h - t_h + 1, stride):
        for j in range(0, s_w - t_w + 1, stride):
            window = scene[i:i + t_h, j:j + t_w].flatten()
            window_norm = window / (np.linalg.norm(window) + 1e-8)
            # Kosinüs benzerliği: iki vektörün "ne kadar aynı yöne
            # işaret ettiğini" ölçer -- şablon eşleştirmede yaygın kullanılır
            score = float(np.dot(window_norm, template_norm))
            if score > threshold:
                detections.append((i, j, score))

    return detections


def non_max_suppression(
    detections: list[tuple[int, int, float]], distance_threshold: int = 8
) -> list[tuple[int, int, float]]:
    """Birbirine çok yakın, aynı nesneyi işaret eden çoklu tespitleri
    tek bir en iyi tespide indirger (gerçek nesne tespit sistemlerinde
    'Non-Max Suppression' olarak bilinen standart bir son işleme adımı)."""
    if not detections:
        return []

    sorted_detections = sorted(detections, key=lambda d: d[2], reverse=True)
    kept: list[tuple[int, int, float]] = []

    for candidate in sorted_detections:
        too_close = any(
            np.hypot(candidate[0] - k[0], candidate[1] - k[1]) < distance_threshold
            for k in kept
        )
        if not too_close:
            kept.append(candidate)

    return kept


def main() -> None:
    scene, true_positions = create_synthetic_scene()
    template = create_template()

    print(f"Sahne boyutu: {scene.shape}")
    print(f"Gerçek 'X' konumları: {true_positions}\n")

    raw_detections = sliding_window_detect(scene, template, stride=1, threshold=0.6)
    print(f"Non-Max Suppression ÖNCESİ tespit sayısı: {len(raw_detections)}")

    final_detections = non_max_suppression(raw_detections, distance_threshold=8)
    print(f"Non-Max Suppression SONRASI tespit sayısı: {len(final_detections)}")
    print("\nNihai tespitler (satır, sütun, güven skoru):")
    for row, col, score in final_detections:
        print(f"  ({row}, {col})  güven={score:.3f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(scene, cmap="gray")
    for row, col, score in final_detections:
        rect = patches.Rectangle(
            (col, row), template.shape[1], template.shape[0],
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(col, row - 2, f"{score:.2f}", color="lime", fontsize=9)
    ax.set_title(f"Tespit Sonuçları ({len(final_detections)} nesne bulundu)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("object_detection_demo.png", dpi=100)
    print("\nGörsel object_detection_demo.png dosyasına kaydedildi.")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Bu basit örnek, modern nesne tespit sistemlerinin (YOLO, SSD, Faster
R-CNN) çözdüğü temel problemi gösterir: "Bu görüntüde nesne NEREDE?"
Kaydırmalı pencere yaklaşımı kavramsal olarak basittir ama hesaplama
açısından çok pahalıdır (her pencere boyutu, her konum için ayrı bir
kontrol). Modern sistemler bunun yerine TEK bir ileri geçişte tüm
görüntüyü analiz edip konum+sınıf tahminlerini aynı anda üretir --
ama "birden fazla aday tespiti tek bir en iyi tespide indirgeme"
fikri (Non-Max Suppression) hâlâ neredeyse tüm modern nesne tespit
sistemlerinin son adımıdır.

İyileştirme Fikirleri
-----------------------
1. Farklı boyutlardaki nesneleri yakalamak için "image pyramid"
   (görüntüyü farklı ölçeklerde tarama) ekleyin.
2. `threshold` ve `distance_threshold` değerlerini değiştirip
   kesinlik/duyarlılık (precision/recall) dengesinin nasıl değiştiğini
   gözlemleyin.
3. `stride` değerini 1'e düşürüp (daha yavaş ama daha hassas) sonuçları
   karşılaştırın.
"""
