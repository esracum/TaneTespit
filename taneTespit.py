import cv2
import numpy as np

# Görüntüyü yükle ve boyutlandır
image = cv2.imread('misir.png')
image = cv2.resize(image, (600, 600))
original = image.copy()

# HSV renk uzayına çevir 
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Sarı renk aralığı
lower_yellow = np.array([10, 60, 60])
upper_yellow = np.array([25, 255, 255])
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Morfolojik işlemler
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

# Kenarları netleştir
edges = cv2.Canny(opened, 50, 150)
edges = cv2.dilate(edges, None, iterations=1)
enhanced_mask = cv2.subtract(opened, edges)

# Distance transform
dist_transform = cv2.distanceTransform(enhanced_mask, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# Arka plan ve bilinmeyen alanlar
sure_bg = cv2.dilate(opened, kernel, iterations=3)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker etiketleme
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Watershed
markers = cv2.watershed(image, markers)
output = image.copy()
output[markers == -1] = [255, 0, 0]  # sınırları mavi yapti

# Kontur çizimi ve sayma
count = 0
for marker in np.unique(markers):
    if marker <= 1:
        continue
    mask_marker = np.uint8(markers == marker)
    contours, _ = cv2.findContours(mask_marker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 45:
            count += 1
            cv2.drawContours(output, [cnt], -1, (255, 225, 0), 1)
            x, y, w, h = cv2.boundingRect(cnt)
            text_pos = (x, y - 5 if y - 5 > 10 else y + 15)
            cv2.putText(output, str(count), text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Toplam tane sayısı
cv2.putText(output, f"Tane Sayisi: {count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Görüntüler arası boşluk
gap = 10
white_gap = 255 * np.ones((600, gap, 3), dtype=np.uint8)

# İki görüntüyü yatay birleştir
combined_body = np.hstack((original, white_gap, output))

# Başlık çubuğu (yüksekliği 40 piksel, genişliği otomatik)
title_height = 40
total_width = combined_body.shape[1]
title_bar = 255 * np.ones((title_height, total_width, 3), dtype=np.uint8)

# Başlık konumlarını dinamik hesapla
original_width = original.shape[1]
output_width = output.shape[1]

original_title_x = original_width // 2 - 80
output_title_x = original_width + gap + output_width // 2 - 80

cv2.putText(title_bar, "Orijinal Goruntu", (original_title_x, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
cv2.putText(title_bar, "Tespit Sonucu", (output_title_x, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Başlık ve içerik birleştir
combined = np.vstack((title_bar, combined_body))

# Sonucu göster
cv2.imshow("Misir Tanesi Tespiti", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
