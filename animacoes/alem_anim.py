import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation, PillowWriter


def f(z):
    return np.sin(1j*z)

def df(z):
    return 1j*np.cos(1j*z)


def gerar_frames(
    N=1000,
    range_n=5.0,
    max_epocas=50,
    thresh=0.01,
    df_tol=1e-4,
    z_max=100,
    f_max=1e6
):
    roots = np.array([
        complex(0, np.pi),
        complex(0, 0),
        complex(0, -np.pi)
    ])

    x = np.linspace(-range_n, range_n, N)
    y = np.linspace(-range_n, range_n, N)

    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Códigos:
    # 0 -> raiz (0, pi)
    # 1 -> raiz (0, 0)
    # 2 -> raiz (0, -pi)
    # 3 -> não convergiu ainda / inválido
    convs = np.full((N, N), 3, dtype=np.int8)

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
    range_n = 5.0
    max_epocas = 15

    frames = gerar_frames(
        N=N,
        range_n=range_n,
        max_epocas=max_epocas,
        thresh=0.01
    )

    cores = [
        'black',    # raiz (0, pi)
        'red',      # raiz (0, 0)
        'gold',     # raiz (0, -pi)
        'white'     # não convergiu / inválido
    ]

    cmap = ListedColormap(cores)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 7))

    imagem = ax.imshow(
        frames[0],
        origin='lower',
        extent=[-range_n, range_n, -range_n, range_n],
        cmap=cmap,
        norm=norm
    )

    ax.scatter(
        0,
        np.pi,
        color='black',
        edgecolors='white',
        s=40,
        label=r'$i\pi$'
    )

    ax.scatter(
        0,
        0,
        color='red',
        edgecolors='black',
        s=40,
        label=r'$0$'
    )

    ax.scatter(
        0,
        -np.pi,
        color='gold',
        edgecolors='black',
        s=40,
        label=r'$-i\pi$'
    )

    ax.set_xlabel('Eixo Real')
    ax.set_ylabel('Eixo Imaginário')
    ax.legend(loc='upper right')

    titulo = ax.set_title('')

    def atualizar(frame):
        imagem.set_data(frames[frame])
        titulo.set_text(
            rf'Mapa de Convergência para $f(z)=\sin(iz)$ — max_epocas = {frame}'
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
        'alem_anim.gif',
        writer=PillowWriter(fps=4),
        dpi=120
    )

    plt.close(fig)

    print('Animação salva como alem_anim.gif')


if __name__ == '__main__':
    main()
