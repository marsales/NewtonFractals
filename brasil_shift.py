import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def f(z):
    return ((z**3 + 1) * (z - 1)) / (z - 2)


def df(z):
    return (3*z**4 - 10*z**3 + 6*z**2 - 1) / ((z - 2)**2)


def safe_abs_f(z):
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            valor = f(z)

            if not np.isfinite(valor):
                return np.inf

            return abs(valor)

    except FloatingPointError:
        return np.inf


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
            return np.nan

        backwards_error = abs(f(chute))

    if backwards_error <= thresh:
        return chute

    return np.nan


def main():

    N = 350
    range_n = 5.0

    # Códigos:
    # 0 -> raiz -1
    # 1 -> raiz 1
    # 2 -> raiz 1/2 + i sqrt(3)/2
    # 3 -> raiz 1/2 - i sqrt(3)/2
    # 4 -> não convergiu / inválido
    convs = np.full((N, N), 4)

    roots = np.array([
        complex(-1, 0),
        complex(1, 0),
        complex(0.5, np.sqrt(3) / 2),
        complex(0.5, -np.sqrt(3) / 2)
    ])

    for x in range(N):
        for y in range(N):

            xn = (2 * range_n) * x / (N - 1) - range_n
            yn = (2 * range_n) * y / (N - 1) - range_n

            chute = complex(xn, yn)
            conv = loop_chutes(chute)

            if not np.isfinite(conv):
                convs[y, x] = 4
            else:
                distancias = np.abs(conv - roots)
                raiz_mais_proxima = np.argmin(distancias)

                convs[y, x] = raiz_mais_proxima

    cores = [
        'blue',     # raiz -1
        'red',      # raiz 1
        'green',    # raiz complexa superior
        'yellow',   # raiz complexa inferior
        'black'     # não convergiu / inválido
    ]

    cmap = ListedColormap(cores)

    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.imshow(
        convs,
        origin='lower',
        extent=[-range_n, range_n, -range_n, range_n],
        cmap=cmap,
        norm=norm
    )

    # Raízes
    plt.scatter(-1, 0, color='blue', s=60, label='$-1$')
    plt.scatter(1, 0, color='red', s=60, label='$1$')
    plt.scatter(
        0.5,
        np.sqrt(3) / 2,
        color='green',
        s=60,
        label=r'$\frac{1}{2} + i\frac{\sqrt{3}}{2}$'
    )
    plt.scatter(
        0.5,
        -np.sqrt(3) / 2,
        color='yellow',
        s=60,
        label=r'$\frac{1}{2} - i\frac{\sqrt{3}}{2}$'
    )

    # Contorno preto nas raízes
    plt.scatter(-1, 0, color='black', s=15)
    plt.scatter(1, 0, color='black', s=15)
    plt.scatter(0.5, np.sqrt(3) / 2, color='black', s=15)
    plt.scatter(0.5, -np.sqrt(3) / 2, color='black', s=15)

    # Polo da função
    '''plt.scatter(
        2,
        0,
        color='white',
        edgecolor='black',
        marker='x',
        s=80,
        label='polo $z=2$'
    )'''

    plt.xlabel('Eixo Real')
    plt.ylabel('Eixo Imaginário')
    plt.legend()
    plt.title(r'Mapa de Convergência para $f(z)=\dfrac{(z^3+1)(z-1)}{z-2}$')
    plt.show()


if __name__ == '__main__':
    main()