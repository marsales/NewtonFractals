import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Usaremos cores vibrantes para destacar bem o contraste na fronteira
CORES = ['#000000', '#FF0044', '#00CC66', '#0066FF'] # Preto (não convergiu), Vermelho, Verde, Azul
RAIZES = [1 + 0j, -0.5 + 1j*np.sqrt(3)/2, -0.5 - 1j*np.sqrt(3)/2]

def f(z):
    return z**3 - 1

def df(z):
    return 3*z**2

def compute_fractal_patch(center_x, center_y, size, resolucao=600):
    """Calcula um pedaço específico do fractal com alta resolução."""
    xmin, xmax = center_x - size, center_x + size
    ymin, ymax = center_y - size, center_y + size
    
    x = np.linspace(xmin, xmax, resolucao)
    y = np.linspace(ymin, ymax, resolucao)
    X, Y = np.meshgrid(x, y)
    z = X + 1j * Y
    
    result = np.zeros(z.shape) # Começa tudo como 0 (preto)
    active = np.ones(z.shape, dtype=bool)
    
    iter_max = 80
    threshold = 0.001
    
    for _ in range(iter_max):
        if not active.any():
            break
            
        with np.errstate(all='ignore'):
            fz = f(z)
            dfz = df(z)
            # Previne divisão por zero exata na origem (polo)
            z_new = np.where(dfz != 0, z - fz / dfz, np.nan)
            
        step_ok = np.isfinite(z_new) & (np.abs(z_new) <= 1e10)
        active &= step_ok
        z = np.where(active, z_new, z)
        
        # Verifica convergência apenas para os pontos ativos
        for i, raiz in enumerate(RAIZES):
            convergiu = active & (np.abs(z - raiz) < threshold)
            result[convergiu] = i + 1  # Salva a cor da raiz (1, 2 ou 3)
            active &= ~convergiu # Desativa os que já chegaram no destino

    return result, (xmin, xmax, ymin, ymax)

def main():
    # Ponto de interesse: a origem (0,0) é um polo e ponto central de fronteira
    cx, cy = 0.0, 0.0
    
    # Níveis de zoom: Macro, Médio e Micro
    zooms = [2.0, 0.2, 0.02]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    cmap = ListedColormap(CORES)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    
    for ax, size in zip(axs, zooms):
        print(f"Calculando zoom com janela de tamanho {size*2}...")
        data, extent = compute_fractal_patch(cx, cy, size)
        
        ax.imshow(data, origin='lower', extent=extent, cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(f'Janela: {size*2} x {size*2}')
        ax.set_xlabel('Eixo Real')
        if ax == axs[0]:
            ax.set_ylabel('Eixo Imaginário')
            
    plt.suptitle('Propriedade dos Lagos de Wada: A Fronteira Infinita', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

'''
Nosso código implementa o método de Newton-Raphson no plano complexo, iterando a fórmula 
para mapear as raízes de polinômios, como f(z) = z^3 - 1. 
Cada pixel atua como um ponto de partida e é colorido de acordo com a raiz final para a qual converge, 
formando as Bacias de Atração (Conjunto de Fatou). A teoria que sustenta nossa análise vem dos sistemas dinâmicos, 
focando no comportamento caótico nas fronteiras dessas bacias, chamadas de Conjunto de Julia. Através das janelas 
de zoom sucessivas criadas no algoritmo, conseguimos provar visualmente a propriedade topológica dos Lagos de Wada. 
Essa propriedade garante que a fronteira de uma bacia é simultaneamente a fronteira de todas as outras, 
impossibilitando uma linha divisória simples e revelando um padrão tricolor auto-similar e infinitamente complexo.
Concluímos que o método apresenta extrema sensibilidade às condições iniciais, uma assinatura da Teoria do Caos, 
onde variações infinitesimais alteram completamente a trajetória do ponto. Por fim, demonstramos que compreender os polos 
(onde a derivada zera e lança o ponto ao infinito) é vital: ao ajustarmos os limites de tolerância computacional do código, 
eliminamos artefatos artificiais (o "buraco negro") e revelamos a verdadeira natureza matemática e fractal do plano.

 '''