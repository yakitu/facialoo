import cv2 as cv

ruidos = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
camara = cv.VideoCapture(0)
while True:
    _, captura = camara.read()
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    cara = ruidos.detectMultiScale(grises, 1.3, 5)
    for (x, y, e1, e2) in cara:
        cv.rectangle(captura, (x, y), (x+e1, y+e2), (0, 255, 0), 2)
    cv.imshow('Resultado rostro', captura)
    if cv.waitKey(1) == ord('q'):
        break
camara.release()
cv.destroyAllWindows()
