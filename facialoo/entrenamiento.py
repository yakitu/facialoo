import cv2 as cv
import os
import numpy as np
from time import time

data = 'Data'
lista_data = os.listdir(data)
ids = []
rostros_data = []
id = 0
tiempo_inicial = time()
for fila in lista_data:
    ruta_completa = data + '/' + fila
    print('Iniciando lectura...')
    for archivo in os.listdir(ruta_completa):

        print('Imagenes: ', fila + '/'+archivo)

        ids.append(id)
        rostros_data.append(cv.imread(ruta_completa + '/' + archivo, 0))

    id = id+1
    tiempo_final_lectura = time()
    tiempo_total_lectura = tiempo_final_lectura - tiempo_inicial
    print('Tiempo total lectura: ', tiempo_total_lectura)

entrenamiento_EigenFaceRecognizer = cv.face.EigenFaceRecognizer_create()
print('Iniciando el entrenamiento...espere')
entrenamiento_EigenFaceRecognizer.train(rostros_data, np.array(ids))
Tiempo_final_entrenamiento = time()
tiempo_total_entrenamiento = Tiempo_final_entrenamiento - tiempo_total_lectura
print('Tiempo entrenamiento total: ', tiempo_total_entrenamiento)
entrenamiento_EigenFaceRecognizer.write('EntrenamientoEigenFaceRecognizer.xml')
print('Entrenamiento concluido')
