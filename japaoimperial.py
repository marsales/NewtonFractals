import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def f(z):
    return z**9 - z

def df(z):
    return 9*z**8 - 1


# Raízes de z^9 - z = z(z^8 - 1)
RAIZES = [0 + 0j] + [np.exp(2j*np.pi*k/8) for k in range(8)]


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
    backwards_error = abs(f(chute))

    thresh = 0.01

    epoca = 0
    max_epocas = 100

    while backwards_error > thresh and epoca < max_epocas:
        epoca += 1

        chute = newton_step(chute)

        if not np.isfinite(chute):
            return np.nan, False

        backwards_error = abs(f(chute))

    return chute, backwards_error <= thresh


def main():

    N = 350
    range_n = 2.0
    convs = np.zeros((N, N))

    for x in range(N):
        for y in range(N):

            xn = (2*range_n) * x / (N - 1) - range_n
            yn = (2*range_n) * y / (N - 1) - range_n

            chute = complex(xn, yn)
            val, conv = loop_chutes(chute)

            if not np.isfinite(val) or not conv:
                convs[y, x] = 0   # preto: não convergiu
            else:
                # Descobre para qual raiz o ponto convergiu
                distancias = [abs(val - raiz) for raiz in RAIZES]
                indice_raiz = np.argmin(distancias)

                if indice_raiz == 0:
                    convs[y, x] = 1   # vermelho: raiz 0
                else:
                    convs[y, x] = indice_raiz + 1
                    # raízes não nulas recebem valores 2,3,...,9

    # 0 -> preto
    # 1 -> vermelho
    # 2 até 9 -> tons graduais de amarelo claro
    cores = [
        'black',     # 0: não convergiu
        'red',       # 1: raiz 0

        '#fffde7',   # 2
        '#fff9c4',   # 3
        '#fff59d',   # 4
        '#fff176',   # 5
        '#ffee58',   # 6
        '#ffeb3b',   # 7
        '#fdd835',   # 8
        '#fbc02d'    # 9
    ]

    cmap = ListedColormap(cores)

    bounds = np.arange(-0.5, 10.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    # Plotagem =======================================
    plt.imshow(
        convs,
        origin='lower',
        extent=[-range_n, range_n, -range_n, range_n],
        cmap=cmap,
        norm=norm
    )

    # Marca a raiz 0
    plt.scatter(0, 0, color='red', s=60, edgecolors='black', label='$0$')

    # Marca as 8 raízes não nulas
    for k in range(8):
        raiz = RAIZES[k + 1]
        plt.scatter(
            raiz.real,
            raiz.imag,
            color=cores[k + 2],
            s=50,
            edgecolors='black'
        )

    plt.xlabel('Eixo Real')
    plt.ylabel('Eixo Imaginário')
    plt.legend()
    plt.title('Mapa de Convergência para $f(z) = z^{9}-z$')
    plt.show()


if __name__ == '__main__':
    main()