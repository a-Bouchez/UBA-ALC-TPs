import numpy as np
import scipy.linalg

def construye_adyacencia(D,m):
    """
    Función que construye la matriz de adyacencia del grafo de museos
    D matriz de distancias, m cantidad de links por nodo
    Retorna la matriz de adyacencia como un numpy.
    """
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

def elim_gaussiana(A):
  """
  Realiza la eliminación gaussiana para triangularizar la matriz A.
  Devuelve la matriz L (triangular inferior) y U (triangular superior) resultantes.
  También devuelve la cantidad de operaciones realizadas.
  """
  cant_op = 0
  m=A.shape[0]
  n=A.shape[1]
  Ac = np.copy(A)

  if m!=n:
    print('Matriz no cuadrada')
    return
  
  for k in range(m-1): # para el k-ésimo paso
    for i in range(k+1, n): # para cada fila
      coeficiente = Ac[i,k] / Ac[k,k]
      cant_op += 1
      
      for j in range(m): # para cada columna

        if k < j:
          Ac[i,j] -= coeficiente * Ac[k,j] # triangularización
          cant_op += 2

        if k == j: Ac[i,j] = coeficiente # guardamos el coeficiente
      
    #print("La matriz A^(" + str(k + 1) + ") es:\n", Ac)
  
  #print("Matriz final con L en donde van los ceros:\n", A)
  L = np.tril(Ac,-1) + np.eye(A.shape[0])
  U = np.triu(Ac)

  return L, U, cant_op

def calculaLU(matriz):
    """
    Función para calcular la descomposición LU de una matriz cuadrada.
    matriz: Matriz cuadrada a descomponer
    Retorna: L, U.
    """
    L, U, _ = elim_gaussiana(matriz)
    return L, U

def calcula_matriz_C(A):
    """
    Función para calcular la matriz de trancisiones C.
    Usamos la ecuación C = A^T @ K^(-1).
    A: Matriz de adyacencia
    K: Matriz de grado (de salida), tiene en su diagonal la suma por filas de A
    Retorna la matriz C
    """
    Atrans = np.transpose(A) # Transponemos la matriz A
    
    # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    Kinv = np.zeros(A.shape) # Inicializamos Kinv con ceros
    for i in range(A.shape[0]): # Recorremos las filas de A
        suma = np.sum(A[i,:]) # Suma de la fila i-ésima de A
        if suma != 0: # Si la suma no es cero, asignamos el inverso a la diagonal de Kinv
            Kinv[i,i] = 1/suma
        else: # Si la suma es cero, asignamos cero a la diagonal de Kinv
            Kinv[i,i] = 0

    C = Atrans @ Kinv # Calcula C multiplicando Kinv y A
    return C

def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping (¿asumimos que se refieren al alfa?, no hay d como parámetro de entrada...)
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = A.shape[0] # Obtenemos el número de museos N a partir de la estructura de la matriz A
    M = N/alfa * (np.eye(N) - ((1-alfa) * C)) # Calculamos la matriz M con la ecuación del PDF
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    b = np.ones(N) * alfa/N # Vector de 1s, multiplicado por el coeficiente correspondiente usando d (¿alfa?) y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return 1

def calcula_matriz_C_continua(D): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0)
    Kinv = ... # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F 
    C = ... # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    for i in range(cantidad_de_visitas-1):
        # Sumamos las matrices de transición para cada cantidad de pasos
        return # para que no se queje python
    return B
