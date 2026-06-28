import numpy as np
import matplotlib.pyplot as plt

resolucao = 600
window_size = 5.0

# parâmetro da função f(z) = z^3 + 1 - 3az
a = 0.63


def f(z):
    return z**3 + 1 - 3*a*z


def df(z):
    return 3*z**2 - 3*a


# raízes de z^3 - 3az + 1 = 0
RAIZES = np.roots([1, 0, -3*a, 1])

# polos do mapa de Newton: f'(z) = 0
# 3z^2 - 3a = 0  =>  z = +-sqrt(a)
POLOS = [np.sqrt(a), -np.sqrt(a)]


def plot_newton_step_size(xmin, xmax, ymin, ymax):
    x = np.linspace(xmin, xmax, resolucao)
    y = np.linspace(ymin, ymax, resolucao)

    X, Y = np.meshgrid(x, y)
    z = X + 1j * Y

    with np.errstate(all='ignore'):
        step = np.abs(f(z) / df(z))

    # evita valores infinitos atrapalhando a escala visual
    step = np.where(np.isfinite(step), step, np.nan)

    # escala logarítmica para visualizar melhor
    step_log = np.log10(1 + step)

    plt.figure(figsize=(8, 8))

    plt.imshow(
        step_log,
        origin='lower',
        extent=[xmin, xmax, ymin, ymax],
        interpolation='nearest'
    )

    plt.colorbar(label=r'$\log_{10}(1+|f(z)/f\'(z)|)$')

    # marca as raízes
    for i, raiz in enumerate(RAIZES):
        plt.scatter(
            raiz.real,
            raiz.imag,
            color='red',
            edgecolors='black',
            s=50,
            label=f'raiz {i+1}: {raiz:.3f}'
        )

    # marca os polos
    

    plt.xlabel('Eixo Real')
    plt.ylabel('Eixo Imaginário')
    plt.title(rf'Tamanho do passo de Newton para $f(z)=z^3+1-3az$, com $a={a}$')
    plt.legend(loc='upper right', fontsize=8)
    plt.show()


def main():
    plot_newton_step_size(
        xmin=-window_size,
        xmax=window_size,
        ymin=-window_size,
        ymax=window_size
    )


if __name__ == '__main__':
    main()