import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def f(z):
    return z / (0.6 - z)

def df(z):
    return 0.6 / (0.6 - z)**2


RAIZ = 0 + 0j
POLO = 0.6 + 0j


def newton_step(z):

    if not np.isfinite(z):
        return np.nan

    if abs(z) > 100:
        return np.nan

    # evita avaliar exatamente no polo
    if abs(z - POLO) <= 1e-8:
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
    backwards_error = abs(f(chute)) if abs(chute - POLO) > 1e-8 else np.inf

    thresh = 0.01

    epoca = 0
    max_epocas = 100

    while backwards_error > thresh and epoca < max_epocas:
        epoca += 1

        chute = newton_step(chute)

        if not np.isfinite(chute):
            return np.nan, False

        if abs(chute - POLO) <= 1e-8:
            return np.nan, False

        backwards_error = abs(f(chute))

    return chute, backwards_error <= thresh


def main():

    N = 350
    range_n = 2.0
    convs = np.zeros((N, N))

    for x in range(N):
        for y in range(N):

            xn = (2 * range_n) * x / (N - 1) - range_n
            yn = (2 * range_n) * y / (N - 1) - range_n

            chute = complex(xn, yn)
            val, conv = loop_chutes(chute)

            if not np.isfinite(val) or not conv:
                convs[y, x] = 0   # não convergiu / problema numérico
            else:
                if abs(val - RAIZ) < 0.01:
                    convs[y, x] = 1   # convergiu para 0

    # 0 -> preto
    # 1 -> vermelho
    cores = ['white', 'red']
    cmap = ListedColormap(cores)

    bounds = [-0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # Plotagem =======================================
    plt.imshow(
        convs,
        origin='lower',
        extent=[-range_n, range_n, -range_n, range_n],
        cmap=cmap,
        norm=norm
    )

    # raiz
    plt.scatter(0, 0, color='red', s=60, edgecolors='black', label='raiz $0$')

    plt.xlabel('Eixo Real')
    plt.ylabel('Eixo Imaginário')
    plt.legend()
    plt.title(r'Mapa de Convergência para $f(z)=\dfrac{z}{0.6-z}$')
    plt.show()


if __name__ == '__main__':
    main()