#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np

def elim_gaussiana(A):
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


def main_gaussiana():
  n = 7
  B = np.eye(n) - np.tril(np.ones((n,n)),-1)
  B[:n,n-1] = 1
  #B = np.array([[2, 1, 2, 3],
  #              [4, 3, 3, 4],
  #              [-2, 2, -4, -12],
  #              [4, 1, 8, -3]])
  print('Matriz B \n', B)

  L, U, cant_oper = elim_gaussiana(B)

  print('Matriz L \n', L)
  print('Matriz U \n', U)
  print('Cantidad de operaciones: ', cant_oper)
  print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
  print('Norma infinito de U:', np.max(np.sum(np.abs(U), axis=1)) )

def resolver_sistema_triangular_inferior(L, b):
  """
  Resuelve un sistema de ecuaciones lineales Ly = b, donde L es una matriz triangular inferior.
  Utiliza forward substitution para resolver el sistema.
  """
  n = L.shape[0]
  y = np.zeros(n)
  
  for i in range(n): # para cada fila, de arriba hacia abajo
    suma = 0
    for j in range(i): # para cada columna, asumiendo triangular inferior
      suma += L[i, j] * y[j]
      
    y[i] = (b[i] - suma) / L[i, i]
  return y

def resolver_sistema_triangular_superior(U, y):
  """
  Resuelve un sistema de ecuaciones lineales Ux = y, donde U es una matriz triangular superior.
  Utiliza backward substitution para resolver el sistema.
  """
  n = U.shape[0]
  x = np.zeros(n)
  
  for i in range(n-1, -1, -1): # para cada fila, de abajo hacia arriba
    suma = 0
    for j in range(i+1, n): # para cada columna, asumiendo triangular superior
      suma += U[i, j] * x[j]
      
    x[i] = (y[i] - suma) / U[i, i]
  return x

def resolver_sistema(A, b):
  """
  Resuelve un sistema de ecuaciones lineales Ax = b utilizando la descomposición LU.
  """
  L, U, _ = elim_gaussiana(A)
  
  y = resolver_sistema_triangular_inferior(L, b)
  x = resolver_sistema_triangular_superior(U, y)
  
  return x
  
def main():
  B = np.array([[2, 1, 2, 3],
                [4, 3, 3, 4],
                [-2, 2, -4, -12],
                [4, 1, 8, -3]])
  L, U, _ = elim_gaussiana(B)
  
  b = np.array([1, 2, 3 ,4])
  
  y = resolver_sistema_triangular_inferior(L, b)
  x = resolver_sistema_triangular_superior(U, y)
  
  print('La solucion del sistema Ly = b es: ', y)
  print('La solucion del sistema Ux = y es: ', x)
  print('La solucion del sistema Ax = b es (hecho con NumPy, debería dar lo mismo que el print de arriba): ', np.linalg.solve(B, b))

if __name__ == "__main__":
  main()
