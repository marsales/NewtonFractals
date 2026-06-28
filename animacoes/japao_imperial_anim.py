import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation, PillowWriter


def f(z):
    return z**9 - z

def df(z):
    return 9*z**8 - 1

RAIZES = [0 + 0j] + [np.exp(2j*np.pi*k/8) for k in range(8)]

def gerar_frames(
    N=1000,
    range_n=2.0,
    max_epocas=50,
    thresh=0.01,
    df_tol=1e-4,
    z_max=100,
    f_max=1e6
):
    roots = np.array(
    [0 + 0j] + [np.exp(2j * np.pi * k / 8) for k in range(8)],
    dtype=complex
)

    x = np.linspace(-range_n, range_n, N)
    y = np.linspace(-range_n, range_n, N)

    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Códigos:
    # 0 -> raiz 1
    # 1 -> raiz 2
    # 2 -> raiz 3
    # 3 -> raiz 4
    # 4 -> raiz 5
    # 5 -> raiz 6
    # 6 -> raiz 7
    # 7 -> raiz 8
    # 8 -> raiz 9
    # 9 -> não convergiu ainda / inválido
    convs = np.full((N, N), 10, dtype=np.int8)

    ativo = np.ones((N, N), dtype=bool)

    frames = []

    for epoca in range(max_epocas + 1):

        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            FZ = f(Z)

        finito = np.isfinite(Z) & np.isfinite(FZ)
        convergiu = ativo & finito & (np.abs(FZ) <= thresh)

        if np.any(convergiu):
            Z_conv = Z[convergiu]

            distancias = np.abs(Z_conv[:, None] - roots[None, :])
            indices_raizes = np.argmin(distancias, axis=1)

            convs[convergiu] = indices_raizes
            ativo[convergiu] = False

        frames.append(convs.copy())

        if epoca == max_epocas:
            break

        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            FZ = f(Z)
            DFZ = df(Z)

            valido = (
                ativo
                & np.isfinite(Z)
                & np.isfinite(FZ)
                & np.isfinite(DFZ)
                & (np.abs(DFZ) > df_tol)
                & (np.abs(FZ) < f_max)
                & (np.abs(Z) < z_max)
            )

            Z_novo = np.empty_like(Z)
            Z_novo[:] = np.nan

            Z_novo[valido] = Z[valido] - FZ[valido] / DFZ[valido]

            valido_depois = (
                valido
                & np.isfinite(Z_novo)
                & (np.abs(Z_novo) < z_max)
            )

            Z[valido_depois] = Z_novo[valido_depois]

            ativo[ativo & ~valido_depois] = False

    return frames


def main():
    N = 1000
    range_n = 2.0
    max_epocas = 20

    frames = gerar_frames(
        N=N,
        range_n=range_n,
        max_epocas=max_epocas,
        thresh=0.01
    )

    cores = [
        'red',         # 1: raiz 0
        '#fffde7',   # 2
        '#fff9c4',   # 3
        '#fff59d',   # 4
        '#fff176',   # 5
        '#ffee58',   # 6
        '#ffeb3b',   # 7
        '#fdd835',   # 8
        '#fbc02d',   # 9
        'black'        # não convergiu
    ]

    cmap = ListedColormap(cores)
    bounds = np.arange(-0.5, 10.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 7))

    imagem = ax.imshow(
        frames[0],
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
            color=cores[k + 1],
            s=50,
            edgecolors='black',
            label=fr'$e^{{{k}\pi i/4}}$'
        )

    ax.set_xlabel('Eixo Real')
    ax.set_ylabel('Eixo Imaginário')
    ax.legend(loc='upper right')

    titulo = ax.set_title('')

    def atualizar(frame):
        imagem.set_data(frames[frame])
        titulo.set_text(
            rf'Mapa de Convergência para $f(z) = z^{9}-z$ — max_epocas = {frame}'
        )
        return imagem, titulo

    animacao = FuncAnimation(
        fig,
        atualizar,
        frames=len(frames),
        interval=250,
        blit=False
    )

    animacao.save(
        'animacao_newton.gif',
        writer=PillowWriter(fps=4),
        dpi=120
    )

    plt.close(fig)

    print('Animação salva como animacao_newton.gif')


if __name__ == '__main__':
    main()