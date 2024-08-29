import cv2 as cv
import os
import imutils

data = 'Data'
lista_data = os.listdir(data)
entrenamiento_EigenFaceRecognizer = cv.face.EigenFaceRecognizer_create()
entrenamiento_EigenFaceRecognizer.read('EntrenamientoEigenFaceRecognizer.xml')
ruidos = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
camara = cv.VideoCapture(0)
while True:
    respuesta, captura = camara.read()
    if respuesta == False:
        break
    captura = imutils.resize(captura, width=640)
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    id_captura = grises.copy()
    cara = ruidos.detectMultiScale(grises, 1.3, 5)
    for (x, y, e1, e2) in cara:
        rostro_capturado = id_captura[y:y+e2, x:x+e1]
        rostro_capturado = cv.resize(
            rostro_capturado, (160, 160), interpolation=cv.INTER_CUBIC)
        resultado = entrenamiento_EigenFaceRecognizer.predict(rostro_capturado)
        cv.putText(captura, '{}'.format(resultado), (x, y-5),
                   1, 1.3, (0, 255, 0), 1, cv.LINE_AA)
        if resultado[1] < 8000:
            cv.putText(captura, '{}'.format(
                lista_data[resultado[0]]), (x, y-20), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x, y), (x+e1, y+e2), (255, 0, 0), 2)
        else:
            cv.putText(captura, "No encontrado", (x, y-20),
                       2, 0.7, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x, y), (x+e1, y+e2), (255, 0, 0), 2)

    cv.imshow("Resultados", captura)
    if cv.waitKey(1) == ord('q'):
        break
camara.release()
cv.destroyAllWindows()
