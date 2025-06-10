# Sistema de Reconhecimento Facial
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Um sistema completo de reconhecimento facial implementando quatro métodos diferentes de detecção:

- Haar cascade (usando OpenCV)
- HOG+SVM (Dlib)
- MMOD/CNN (Dlib)
- SSD (módulo DNN do OpenCV)

## 📌Funcionalidades

- Múltiplos algoritmos de detecção facial
- Capacidade de detecção em tempo real
- Comparação de desempenho entre métodos
- API fácil de usar
- Suporte para imagens estáticas e vídeo
- Detecção de múltiplas faces na mesma imagem

## ![icons8-book-shelf-48](https://github.com/user-attachments/assets/2c01421a-afcf-47f9-b555-416ff2f0cbb3) Biblioteca 
``` python
import cv2 # OpenCV
import numpy as np
from google.colab.patches import cv2_imshow
```

# ![icons8-facial-60](https://github.com/user-attachments/assets/f0617e03-2dae-40e8-9714-e4c084e6356f) Detecção de faces com Haar cascade (OpenCV)

Este código realiza detecção facial usando o método Haar Cascade do OpenCV. Aqui está o resumo passo a passo:

## Carrega a imagem:
```python
imagem = cv2.imread('/content/5.jpg')
```
- Lê a imagem do caminho especificado

## Converte para escala de cinza:
```python
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
```
- Converte a imagem colorida (BGR) para tons de cinza (necessário para o Haar Cascade)

## Inicializa o detector facial:
```python
detector_facial = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')
```
- Carrega o classificador pré-treinado para detecção de faces frontais

## Executa a detecção:
```python
detector_facial = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')
```
- Detecta faces na imagem retornando:

   - x, y: coordenadas do canto superior esquerdo do retângulo

   - w, h: largura e altura do retângulo
     
## Desenha retângulos nas faces detectadas e mostra resultado:
```python
for (x, y, w, h) in deteccoes:
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 255), 4)
cv2_imshow(imagem)
```
- Para cada face detectada, desenha um retângulo amarelo (BGR: 0,255,255) com espessura 4
- Mostra a imagem com as detecções (função específica do Google Colab)

  ![download](https://github.com/user-attachments/assets/49e7901e-0caa-488b-8cdb-356517664d31)


# ![icons8-eye-94](https://github.com/user-attachments/assets/f94109f3-6b0f-4f86-b0a4-427188195f1c)Detecção de olhos

Este código utiliza o OpenCV com o classificador Haar Cascade para detectar olhos em uma imagem.
```python
detector_olhos = cv2.CascadeClassifier('/content/haarcascade_eye.xml')
imagem = cv2.imread('/content/2.jpg')
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

deteccoes_faces = detector_facial.detectMultiScale(imagem_cinza)
for (x, y, w, h) in deteccoes_faces:
  cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

deteccoes_olhos = detector_olhos.detectMultiScale(imagem_cinza, minNeighbors=2, minSize = (25,25))
for (x, y, w, h) in deteccoes_olhos:
  print(w, h)
  cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2_imshow(imagem)
```
![download (1)](https://github.com/user-attachments/assets/2500fdbe-371f-44f0-893a-3dcd3c5a2f25)

## Saída Esperada:
- Uma imagem com retângulos verdes destacando o rosto e retangulos vermelhos todos os olhos detectados.
- Se precisar melhorar a detecção, ajuste parâmetros como scaleFactor, minNeighbors ou minSize no detectMultiScale().

# ![download (2)](https://github.com/user-attachments/assets/45081d16-bb5a-4a33-b75a-a9d0b5d49747) Detecção de Sorriso
Este código utiliza o OpenCV com o classificador Haar Cascade para detectar sorrisos em uma imagem.
```python
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
image = cv2.imread("/content/5.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = smile_detector.detectMultiScale(image_gray, scaleFactor = 1.3, minNeighbors=15)

for (x, y, w, h) in detections:
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
cv2_imshow(image)
```
![download (3)](https://github.com/user-attachments/assets/1a279ba0-5468-4ce7-93fa-f4dfd19f9e67)


# ![icons8-facial-60](https://github.com/user-attachments/assets/f0617e03-2dae-40e8-9714-e4c084e6356f) Detecção de faces com SSD (OpenCV's DNN module)
Este código utiliza a biblioteca OpenCV e um modelo pré-treinado baseado em ResNet SSD para detectar rostos em uma imagem. Ele carrega os arquivos do modelo Caffe (.prototxt e .caffemodel), processa a imagem de entrada convertendo-a em um blob adequado para a rede neural, executa a inferência e, para cada detecção com confiança superior a 50%, desenha uma caixa delimitadora verde ao redor do rosto e exibe a porcentagem de confiança. Ao final, a imagem com os rostos identificados é exibida com as marcações.

```python
for i in range(0, deteccoes.shape[2]):
  confianca = deteccoes[0, 0, i, 2]
  if confianca > conf_min:
    #print(confianca)
    text_conf = "{:.2f}%".format(confianca * 100)
    box = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
    #print(box)
    (start_x, start_y, end_x, end_y) = box.astype(int)
    #print(start_x, start_y, end_x, end_y)
    cv2.rectangle(imagem, (start_x, start_y), (end_x, end_y), (0,255,0), 2)
    cv2.putText(imagem, text_conf, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
cv2_imshow(imagem)
```
![download (4)](https://github.com/user-attachments/assets/a1693802-f243-4b13-b65b-4fe7b0380da1)

# ![download (6)](https://github.com/user-attachments/assets/2545cf55-60ba-432a-83a7-d3f93b271c92) HAAR CASCADES ![download (5)](https://github.com/user-attachments/assets/202fe877-478c-46da-b40e-5325cbae6065) HOG + SVM ![download (5)](https://github.com/user-attachments/assets/79504cf5-8464-4fcb-8226-ebcf3b121809) SSD 

## HAAR CASCADES
```python
image = cv2.imread(test_image)
start = time.time()
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
haarcascade_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detections = haarcascade_detector.detectMultiScale(image_gray, scaleFactor = 1.3, minNeighbors=3, minSize = (5,5))

for (x, y, w, h) in detections:
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)

end = time.time()
cv2_imshow(image)
print("Detection time: {:.2f} s".format(end - start))
```
![download (7)](https://github.com/user-attachments/assets/d588b3b3-7dc9-4b6c-b526-009814e5b9b9)

## HOG + SVM
```python
image = cv2.imread(test_image)
start = time.time()
face_detector_hog = dlib.get_frontal_face_detector()
detections = face_detector_hog(image, 2)
for face in detections:
    l, t, r, b = (face.left(), face.top(), face.right(), face.bottom())
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 255), 2)
end = time.time()
cv2_imshow(image)
print("Detection time: {:.2f} s".format(end - start))
```
![download (8)](https://github.com/user-attachments/assets/c9717090-2889-4d1a-803e-52d0ce0c82fa)

## SSD
```python
image = cv2.imread(test_image)
(h, w) = image.shape[:2]

start = time.time()
network = cv2.dnn.readNetFromCaffe(arquivo_prototxt, arquivo_modelo)
blob = cv2.dnn.blobFromImage(cv2.resize(image, (900, 900)), 1.0, (900, 900), (104.0, 117.0, 123.0))
network.setInput(blob)
detections = network.forward()
for i in range(0, detections.shape[2]):
  confidence = detections[0, 0, i, 2]
  if confidence > conf_min:
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (start_x, start_y, end_x, end_y) = box.astype("int")
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

end = time.time()
cv2_imshow(image)
print("Detection time: {:.2f} s".format(end - start))
```
![download (9)](https://github.com/user-attachments/assets/391cf2cb-2b43-4a87-90af-c081a0be38b5)

## 🥇 SSD (Single Shot Detector) – Vencedor Geral
Por que ganha?

- Usa redes neurais profundas, como ResNet ou MobileNet.
- Alta precisão, mesmo em imagens com múltiplos rostos, em diferentes posições, tamanhos e iluminações.
- É rápido o suficiente para aplicações em tempo real (principalmente com aceleração por GPU).
- Funciona bem em ambientes complexos e não controlados.


