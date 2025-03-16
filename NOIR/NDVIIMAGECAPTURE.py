import cv2
import numpy as np
from picamera import PiCamera
from time import sleep


def capture_image(save_path):
    camera = PiCamera()
    camera.resolution = (1024, 768)  # Ajusta la resolución según tus necesidades
    sleep(2)  # Tiempo para inicializar la cámara
    camera.capture(save_path)
    camera.close()


def display(image, image_name):
    image = np.array(image, dtype=float) / float(255)
    shape = image.shape
    height = int(shape[0] / 2)
    width = int(shape[1] / 2)
    image = cv2.resize(image, (width, height))
    cv2.namedWindow(image_name)
    cv2.imshow(image_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contrast_stretch(im):
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = (im - in_min) * ((out_max - out_min) / (in_max - in_min))
    out = np.clip(out, out_min, out_max)  # Asegurarse de que los valores estén dentro del rango
    return out.astype(np.uint8)


def calc_ndvi(image):
    # Separar las bandas
    b, g, r = cv2.split(image)

    # Usar el canal rojo (R) como NIR y el canal azul (B) como RED
    nir = r.astype(float)
    red = b.astype(float)

    # Calcular NDVI
    ndvi = (nir - red) / (nir + red + 1e-6)  # Evitar divisiones por cero

    # Escalar NDVI a rango 0-255 para visualizar
    ndvi_normalized = ((ndvi + 1) / 2) * 255
    return ndvi_normalized.astype(np.uint8)


# Captura la imagen
image_path = '/home/username/park.png'
capture_image(image_path)

# Procesa la imagen
original = cv2.imread(image_path)
display(original, 'Original')

contrasted = contrast_stretch(original)
display(contrasted, 'Contrasted original')
cv2.imwrite('contrasted.png', contrasted)

ndvi = calc_ndvi(contrasted)
display(ndvi, 'NDVI')

ndvi_contrasted = contrast_stretch(ndvi)
display(ndvi_contrasted, 'NDVI Contrasted')
cv2.imwrite('ndvi_contrasted.png', ndvi_contrasted)
