import cv2 as cv
import os
import imutils

ruta_completa = 'Data/Usuario'
if not os.path.exists(ruta_completa):
    os.makedirs(ruta_completa)


camara = cv.VideoCapture(0)
ruidos = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
id = 0
while True:
    respuesta, captura = camara.read()
    if respuesta == False:
        break
    captura = imutils.resize(captura, width=640)

    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    id_captura = captura.copy()

    cara = ruidos.detectMultiScale(grises, 1.3, 5)

    for (x, y, e1, e2) in cara:
        cv.rectangle(captura, (x, y), (x+e1, y+e2), (0, 255, 0), 2)
        rostro_capturado = id_captura[y:y+e2, x:x+e1]
        rostro_capturado = cv.resize(
            rostro_capturado, (160, 160), interpolation=cv.INTER_CUBIC)
        cv.imwrite(ruta_completa+'/imagen_{}.jpg'.format(id), rostro_capturado)
        id = id+1

    cv.imshow("Resultado rostro", captura)

    if id == 351:
        break
camara.release()
cv.destroyAllWindows()
