import cv2

# Carregar imagens
imagem1 = cv2.imread('imagem3.jpg', cv2.IMREAD_GRAYSCALE)
imagem2 = cv2.imread('imagem4.jpg', cv2.IMREAD_GRAYSCALE)

# Inicializar o detector e descritor ORB
orb = cv2.ORB_create()

# Encontrar pontos-chave e descritores com ORB
pontos_chave1, descritores1 = orb.detectAndCompute(imagem1, None)
pontos_chave2, descritores2 = orb.detectAndCompute(imagem2, None)

# Inicializar o matcher (correspondência) BF (Brute Force)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Realizar correspondência usando o matcher BF
correspondencias = bf.match(descritores1, descritores2)

# Ordenar correspondências com base na distância
correspondencias = sorted(correspondencias, key=lambda x: x.distance)

# Desenhar correspondências na imagem
imagem_correspondencias = cv2.drawMatches(imagem1, pontos_chave1, imagem2, pontos_chave2, correspondencias[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Salvar a imagem com correspondências
cv2.imwrite('imagem_correspondencias.jpg', imagem_correspondencias)

# Exibir a mensagem de sucesso
print('Imagem com correspondências salva com sucesso!')

