
'''
Visualização e interpretação do comportamento de algumas funções através do Método de Newton no plano complexo

-> Se df(z) = 0 ou f(z) for muito grande em algum momento, não continua -> sem cor
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def f(z):
    return np.sin(1j*z)

def df(z):
    return 1j*np.cos(1j*z)

def newton_step(z):

    if not np.isfinite(z):
        return np.nan

    if abs(z) > 100:
        return np.nan

    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):

            fp = f(z)
            dfp = df(z)

            if not np.isfinite(fp) or not np.isfinite(dfp):
                return np.nan

            if abs(dfp) <= 1e-4:
                return np.nan

            if abs(fp) > 1e6:
                return np.nan

            z_new = z - fp / dfp

            if not np.isfinite(z_new):
                return np.nan

            if abs(z_new) > 100:
                return np.nan

            return z_new

    except FloatingPointError:
        return np.nan
    


def loop_chutes(chute):
    thresh = 0.01
    max_epocas = 100

    for epoca in range(max_epocas):

        if not np.isfinite(chute):
            return np.nan

        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                backwards_error = abs(f(chute))
        except FloatingPointError:
            return np.nan

        if not np.isfinite(backwards_error):
            return np.nan

        if backwards_error <= thresh:
            return chute

        chute = newton_step(chute)

    return np.nan



def main():

    N = 500        # Resolução
    range_n = 5.0

    convs = np.full((N, N), np.nan)

    # Raízes escolhidas para representar as 3 faixas
    raiz_cima = complex(0, np.pi)
    raiz_meio = complex(0, 0)
    raiz_baixo = complex(0, -np.pi)

    raizes = [raiz_cima, raiz_meio, raiz_baixo]

    for x in range(N):
        for y in range(N):

            xn = (2*range_n) * x / (N - 1) - range_n
            yn = (2*range_n) * y / (N - 1) - range_n

            chute = complex(xn, yn)
            conv = loop_chutes(chute)

            if not np.isfinite(conv):
                convs[y, x] = np.nan
            else:
                distancias = [abs(conv - r) for r in raizes]
                indice = np.argmin(distancias)

                if distancias[indice] < 0.5:
                    convs[y, x] = indice + 1
                else:
                    convs[y, x] = np.nan

    # Alemanha: preto em cima, vermelho no meio, amarelo embaixo
    cores = ['black', 'red', 'gold']
    cmap = ListedColormap(cores)
    cmap.set_bad(color='white')

    bounds = [0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.imshow(
        convs,
        origin='lower',
        extent=[-range_n, range_n, -range_n, range_n],
        cmap=cmap,
        norm=norm
    )

    # Marca as raízes
    plt.scatter(0, np.pi, color='black', edgecolors = 'purple', s=40, label=r'$i\pi$')
    plt.scatter(0, 0, color='red', s=40, edgecolors = 'purple', label=r'$0$')
    plt.scatter(0, -np.pi, color='gold', s=40, edgecolors = 'purple', label=r'$-i\pi$')

    plt.scatter(0, np.pi, color='purple', s=40)
    plt.scatter(0, 0, color='purple', s=40)
    plt.scatter(0, -np.pi, color='purple', s=40)

    plt.xlabel('Eixo Real')
    plt.ylabel('Eixo Imaginário')
    plt.title(r'Mapa de Convergência para $f(z)=\sin(iz)$')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
