import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

resolucao = 350
iter_max = 100
threshold = 0.01       
tolerancia_root = 0.5       
window_size = 5.0  #isso aqui é a janela inicial    


RAIZES = [complex(0, np.pi), complex(0, 0), complex(0, -np.pi)]
CORES = ['black', 'red', 'gold']




def f(z):
    return np.sin(1j * z)

def df(z):
    return 1j * np.cos(1j * z)





def compute_fractal(xmin, xmax, ymin, ymax, resolucao=resolucao):
    x = np.linspace(xmin, xmax, resolucao)
    y = np.linspace(ymin, ymax, resolucao)
    X, Y = np.meshgrid(x, y)
    z = X + 1j * Y

    result = np.full(z.shape, np.nan)
    active = np.ones(z.shape, dtype=bool)  # pontos que ainda estão sendo iterados

    for _ in range(iter_max):
        if not active.any():
            break

        with np.errstate(all='ignore'):
            fz = f(z)
            dfz = df(z)

        valid = np.isfinite(fz) & np.isfinite(dfz) & (np.abs(z) <= 100)
        active &= valid
        if not active.any():
            break

        err = np.abs(fz)
        converged_now = active & (err <= threshold)

        if converged_now.any():

            # distância de cada ponto convergente até cada raiz
            dists = np.stack([np.abs(z - r) for r in RAIZES], axis=0)
            best_idx = np.argmin(dists, axis=0)
            best_dist = np.min(dists, axis=0)
            assign = converged_now & (best_dist < tolerancia_root)
            result[assign] = best_idx[assign] + 1

            # pontos convergentes (com raiz aceitável ou não) saem da iteração
            active &= ~converged_now

        if not active.any():
            break

        denom_ok = active & (np.abs(dfz) > 1e-4) & (np.abs(fz) <= 1e6)
        active &= denom_ok
        if not active.any():
            break

        with np.errstate(all='ignore'):
            z_new = z - fz / dfz

        step_ok = np.isfinite(z_new) & (np.abs(z_new) <= 100)
        active &= step_ok
        z = np.where(active, z_new, z)

    return result




# visualização (barra pesada)
class FractalViewer:
    def __init__(self):
        self.cmap = ListedColormap(CORES)
        self.cmap.set_bad(color='white')
        self.norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], self.cmap.N)

        self.xlim0 = (-window_size, window_size)
        self.ylim0 = (-window_size, window_size)

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self._busy = False  # evita recálculo recursivo quando setamos xlim/ylim nós mesmos

        self.redraw(*self.xlim0, *self.ylim0)

        self.ax.callbacks.connect('xlim_changed', self.on_lims_changed)
        self.ax.callbacks.connect('ylim_changed', self.on_lims_changed)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        plt.show()

    def redraw(self, xmin, xmax, ymin, ymax):
        data = compute_fractal(xmin, xmax, ymin, ymax)

        self.ax.clear()
        self.ax.imshow(
            data,
            origin='lower',
            extent=[xmin, xmax, ymin, ymax],
            cmap=self.cmap,
            norm=self.norm,
            interpolation='nearest',
        )

        # marca as raízes (só aparecem quando estão dentro da vista atual)
        for raiz, cor, label in zip(
            RAIZES, CORES, [r'$i\pi$', r'$0$', r'$-i\pi$']
        ):
            self.ax.scatter(
                raiz.real, raiz.imag, color=cor, edgecolors='purple', s=40, label=label
            )

        self.ax.set_xlabel('Eixo Real')
        self.ax.set_ylabel('Eixo Imaginário')
        self.ax.set_title(r'Mapa de Convergência para $f(z)=\sin(iz)$')
        self.ax.legend(loc='upper right')

        self._busy = True
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self._busy = False

        self.fig.canvas.draw_idle()

    def on_lims_changed(self, _ax):
        if self._busy:
            return
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        self.redraw(xmin, xmax, ymin, ymax)

    def on_key(self, event):
        if event.key == 'r':
            self.redraw(*self.xlim0, *self.ylim0)

    def on_scroll(self, event):
        if event.xdata is None or event.ydata is None:
            return  # scroll fora da área do gráfico

        scale = 0.8 if event.button == 'up' else 1.25  # "up" significa zoom-in

        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata

        new_w = (xmax - xmin) * scale
        new_h = (ymax - ymin) * scale
        relx = (xdata - xmin) / (xmax - xmin)
        rely = (ydata - ymin) / (ymax - ymin)

        new_xmin = xdata - relx * new_w
        new_xmax = new_xmin + new_w
        new_ymin = ydata - rely * new_h
        new_ymax = new_ymin + new_h

        self.redraw(new_xmin, new_xmax, new_ymin, new_ymax)






def main():
    FractalViewer()


if __name__ == '__main__':
    main()