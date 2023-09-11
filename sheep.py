import cv2
import numpy as np

# Video dosyasının adı ve yolunu belirt
video_path = 'video.mp4' #video konumu yazmak için gerekli kısım
# Video dosyasını okuması yapıldı
cap = cv2.VideoCapture(video_path)

# Videonun genişliği ve yüksekliği belirt
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Video oynatma hızını ayarla (x0.5) video hızı ayarlanarak daha net takip
frame_rate = 30  # Örnek bir çerçeve hızı, videonun çerçeve hızına göre değiştirin
new_frame_rate = int(frame_rate * 0.5)

# Video kaydı için çıktı dosyası oluşturma 
output_path = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, new_frame_rate, (frame_width, frame_height))

# Koyun sayacı
sheep_count = 0

# Çizginin solundaki sınırları belirleme
line_x = frame_width // 4
# Önceki çerçeve üzerindeki koyunların konumlarını saklayacak bir dizi
prev_sheep_positions = []

# Videoyu çerçeve çerçeve işleme
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Videoyu yavaşlatmak için 
    cv2.waitKey(1000 // new_frame_rate)

    # Çerçeveyi siyah-beyaz yap
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Kenarları algıla
    edges = cv2.Canny(gray_frame, threshold1=50, threshold2=150)

    # Algılanan kenarları çiz
    cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 0, 255), 2)

    # Koyunları sayma işlemi gerçekleştir
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.pointPolygonTest(contour, (line_x, frame_height // 2), False) >= 0:
            # Koyunun konumunu belirle
            x, y, w, h = cv2.boundingRect(contour)
            sheep_center = (x + w // 2, y + h // 2)

            # Önceki çerçevede bu koyun var mı kontrol et
            prev_sheep_found = False
            for prev_sheep_position in prev_sheep_positions:
                if np.linalg.norm(np.array(sheep_center) - np.array(prev_sheep_position)) < 20:
                    prev_sheep_found = True
                    break

            # Eğer bu koyun önceki çerçevede yoksa say
            if not prev_sheep_found:
                sheep_count += 1
                prev_sheep_positions.append(sheep_center)

    # Koyun sayısını çerçeve üzerine yaz
    cv2.putText(frame, f"Koyun Sayaci: {sheep_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Çerçeveyi kaydet
    out.write(frame)

    # Çerçeveyi ekranda göster
    cv2.imshow('Frame', frame)

    # Çıkış için 'q' tuşuna basarak programı durdur
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Koyun sayısını enson terminal kısmında yazdır
print(f"Toplam Koyun Sayısı: {sheep_count}")
