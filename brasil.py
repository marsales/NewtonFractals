import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def f(z):
    return (z**3+1)*(z-1) / (z-1)

def df(z):
    return (3*z**2)



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
    max_epocas = 7

    while backwards_error > thresh and epoca < max_epocas:
        epoca += 1

        chute = newton_step(chute)
        if (not np.isfinite(chute)):
          return np.nan, False

        backwards_error = abs(f(chute))

    return chute, backwards_error <= thresh


def main():

    N = 1000
    range_n = 2.0
    convs = np.zeros((N, N))

    for x in range(N):
        for y in range(N):

            xn = (2*range_n) * x / (N - 1) - range_n
            yn = (2*range_n) * y / (N - 1) - range_n

            chute = complex(xn, yn)
            val, conv = loop_chutes(chute)

            if (not np.isfinite(val) or not conv):
              convs[y, x] = 0
            else:
                if abs(val-complex(-1)) < min(abs(val-complex(0.5, np.sqrt(3)*0.5)),abs(val-complex(0.5, -np.sqrt(3)*0.5))):
                    convs[y,x] = -1
                if abs(val-complex(0.5, np.sqrt(3)*0.5)) < min(abs(val-complex(-1)),abs(val-complex(0.5, -np.sqrt(3)*0.5))):
                    convs[y,x] = 1
                if abs(val-complex(0.5, -np.sqrt(3)*0.5)) < min(abs(val-complex(-1)),abs(val-complex(0.5, np.sqrt(3)*0.5))):
                    convs[y,x] = 2
              

    cores = ['blue', 'white', 'green', 'yellow', 'black']
    cmap = ListedColormap(cores)

    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # Plotagem =======================================
    plt.imshow(
        convs,
        origin='lower',
        extent=[-range_n, range_n, -range_n, range_n],
        cmap = cmap,
        norm = norm
    )

    plt.scatter(-1, 0, color='blue', edgecolors='black',  s=50, label='$-1$')
    plt.scatter(0.5, np.sqrt(3)/2, color='green', edgecolors='black', s=50, label='$0.5 + i \\dfrac{\\sqrt{3}}{2}$')
    plt.scatter(0.5, -np.sqrt(3)/2, color='yellow', s=50, edgecolors='black', label='$0.5 - i \\dfrac{\\sqrt{3}}{2}$')
    

    plt.xlabel('Eixo Real')
    plt.ylabel('Eixo Imaginário')
    plt.legend()
    plt.title('Mapa de Convergência para $f(z) = z^{3}+1$')
    plt.show()

if __name__ == '__main__':
    main()





