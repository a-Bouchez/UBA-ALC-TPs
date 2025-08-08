def calcula_K(A):
  K = np.zeros(A.shape) # Inicializamos Kinv con ceros
  for i in range(A.shape[0]): # Recorremos las filas de A
    suma = np.sum(A[i,:]) # Suma de la fila i-ésima de A
    K[i,i] = suma # Asignamos la suma a la diagonal de K
  return K

def calcula_L(A):
  return calcula_K(A) - A

def calcula_P(A):
  K = calcula_K(A) # Calculamos la matriz de grado K
  doubleE = np.sum(K) # Cantidad de aristas por dos
  P = np.zeros(A.shape) # Inicializamos P con ceros
  for i in range(A.shape[0]): # Recorremos las filas de A
    for j in range(A.shape[1]): # Recorremos las columnas de A
      if doubleE != 0: # Si la cantidad de aristas por dos no es cero, asignamos el valor correspondiente
        P[i,j] = (K[i,i] * K[j,j]) / (2 * doubleE) # Calculamos el valor de P_ij
  return P

def calcula_R(A):
  return A - calcula_P(A)

def signo(v_i):
  return -1 if v_i < 0 else 1  # Retorna -1 si el elemento es negativo, 1 si es positivo o cero

def calcula_s(v):
  s = np.zeros(v.shape) # Inicializamos s con ceros
  for i in range(len(v)): # Recorremos el vector v
    s[i] = signo(v[i]) # Asignamos el signo correspondiente a cada elemento de v
  return s

def calcula_lambda(L, v):
  s = calcula_s(v)  # Calculamos el vector s a partir de v
  return (1/4) * s.transpose() @ L @ s  # Calculamos Lambda

def calcula_Q(R, v):
  s = calcula_s(v)  # Calculamos el vector s a partir de v
  return s.transpose() @ R @ s  # Calculamos Q

def metpot1(A, tol=1e-8, maxrep=np.inf):
  # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
  v = np.random.rand(A.shape[0]) # Generamos un vector de partida aleatorio, entre -1 y 1
  v = v / np.linalg.norm(v) # Lo normalizamos
  v1 = A @ v # Aplicamos la matriz una vez
  v1 = v1 / np.linalg.norm(v1) # Normalizamos
  l =  v @ A @ v # Calculamos el autovalor estimado
  l1 = v1 @ A @ v1 # Y el estimado en el siguiente paso
  nrep = 0 # Contador
  while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
    v = v1 # Actualizamos v y repetimos
    l = l1
    v1 = A @ v # Calculamos nuevo v1
    v1 = v1 / np.linalg.norm(v1) # Normalizamos
    l1 = v1 @ A @ v1 # Calculamos autovalor
    nrep += 1 # Un pasito mas
  if not nrep < maxrep:
    print('MaxRep alcanzado')
  l = v1 @ A @ v1 # Calculamos el autovalor
  return v1, l, nrep<maxrep

def deflaciona(A, tol=1e-8, maxrep=np.inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1, l1, _ = metpot1(A, tol, maxrep) # Buscamos primer autovector con método de la potencia
    deflA = A - l1 * np.linalg.outer(v1, v1) # Sugerencia, usar la funcion outer de numpy
    return deflA

def inversa(A):
  # Calcula la inversa de la matriz A usando LU
  L,U = calculaLU(A)
  I = np.eye(A.shape[0])
  Y = scipy.linalg.solve_triangular(L,I,lower=True) # Resuelvo el sistema LY = I con Y = UX
  A_inv = scipy.linalg.solve_triangular(U,Y) # Resuelvo el sistema UA_inv = Y
  return A_inv

def metpotI(A, mu, tol=1e-8, maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    M = A + mu * np.eye(A.shape[0])
    M_inv = inversa(M) # Calculamos la inversa de M
    return metpot1(M_inv, tol, maxrep)

def metpot2(A, v1, l1, tol=1e-8, maxrep=np.inf):
  # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
  # v1 y l1 son los primeors autovectores y autovalores de A
  # Have fun!
  deflA = A - l1 * np.linalg.outer(v1, v1) # Deflacionamos la matriz A
  return metpot1(deflA, tol, maxrep)

def metpotI2(A, mu, tol=1e-8, maxrep=np.inf):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   X = A + mu * np.eye(A.shape[0]) # Calculamos la matriz A shifteada en mu
   iX = inversa(X) # La invertimos
   defliX = deflaciona(iX, tol, maxrep) # La deflacionamos
   v,l,_ =  metpot1(defliX, tol, maxrep) # Buscamos su segundo autovector y autovalor
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_

def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        v,_,_ = metpotI2(L, mu) # Encontramos el segundo autovector de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        Ap = A[:,v>0][v>0,:] # Asociado al signo positivo
        Am = A[:,v<0][v<0,:] # Asociado al signo negativo
        
        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )        


def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return([[nombres_s[0]]]) # Retornamos el único nodo en una lista
    else:
        v,_,_ = metpot1(R) # Primer autovector y autovalor de R
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]]) # Retornamos la partición actual
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            Rp = R[v>0,:][:,v>0] # Parte asociada a los valores positivos de v
            Rm = R[v<0,:][:,v<0] # Parte asociada a los valores negativos de v
            vp,_,_ = metpot1(Rp) # Autovector principal de Rp
            vm,_,_ = metpot1(Rm) # Autovector principal de Rm
        
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles
                return(
                    modularidad_iterativo(A,Rp,[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                    modularidad_iterativo(A,Rm,[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )