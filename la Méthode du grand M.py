# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:26:06 2024

@author: delll
"""

import numpy as np

# Fonction pour résoudre le problème d'optimisation linéaire
def solve_linear_program(c, A, b):
    """
    Résout un problème d'optimisation linéaire simple avec NumPy.
    max z = c @ x sous les contraintes A @ x <= b et x >= 0.
    
    Arguments :
    - c : Coefficients de la fonction objectif (1D array)
    - A : Matrice des contraintes (2D array)
    - b : Limites supérieures des contraintes (1D array)
    
    Retourne :
    - Solution optimale x et valeur optimale z, ou None si aucune solution.
    """
    # Dimensions
    num_vars = len(c)
    num_constraints = len(b)
    
    # Construire la matrice étendue pour les contraintes d'égalité (tableau simplex)
    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
    
    # Fonction objectif (ligne 0)
    tableau[0, :num_vars] = -c  # Maximiser en mettant le négatif de c
    
    # Contraintes
    tableau[1:, :num_vars] = A
    tableau[1:, num_vars:num_vars + num_constraints] = np.eye(num_constraints)
    tableau[1:, -1] = b
    
    # Indices des variables de base
    basis = list(range(num_vars, num_vars + num_constraints))
    
    # Méthode Simplex
    while True:
        # Identifier la colonne pivot (coefficient le plus négatif de la ligne 0)
        col_pivot = np.argmin(tableau[0, :-1])
        if tableau[0, col_pivot] >= 0:
            # Optimalité atteinte (aucun coefficient négatif)
            break
        
        # Identifier la ligne pivot (test du rapport minimal)
        ratios = tableau[1:, -1] / tableau[1:, col_pivot]
        ratios[ratios <= 0] = np.inf  # Ignorer les rapports négatifs ou nuls
        row_pivot = np.argmin(ratios) + 1  # +1 car tableau[1:]
        
        if ratios[row_pivot - 1] == np.inf:
            # Problème non borné
            return None, None
        
        # Pivotage
        pivot_value = tableau[row_pivot, col_pivot]
        tableau[row_pivot, :] /= pivot_value
        for i in range(len(tableau)):
            if i != row_pivot:
                tableau[i, :] -= tableau[i, col_pivot] * tableau[row_pivot, :]
        
        # Mise à jour de la base
        basis[row_pivot - 1] = col_pivot
    
    # Extraire les solutions
    x = np.zeros(num_vars)
    for i, var_index in enumerate(basis):
        if var_index < num_vars:
            x[var_index] = tableau[i + 1, -1]
    
    # Valeur optimale
    z = tableau[0, -1]
    return x, z

# Paramètres du problème
c = np.array([2, 3])  # Coefficients de la fonction objectif
A = np.array([[1, 2], [2, 1]])  # Matrice des contraintes
b = np.array([8, 6])  # Limites des contraintes

# Résolution
solution, optimal_value = solve_linear_program(c, A, b)

# Affichage des résultats
if solution is not None:
    print("Solution optimale trouvée :")
    print("x =", solution)
    print("Valeur optimale de z =", optimal_value)
else:
    print("Aucune solution optimale trouvée.")
