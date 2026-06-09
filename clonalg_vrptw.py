import random
import numpy as np
import matplotlib.pyplot as plt
from AG import (ler_instancia, decoder_max_fill, fitness_vrptw,
                avaliar_vrptw, gerar_pop_vrptw)


def avaliar_pop(populacao, customers, dist, capacity, peso_veiculo):
    fitnesses = []
    for ind in populacao:
        rotas = decoder_max_fill(ind, customers, dist, capacity)
        fitnesses.append(fitness_vrptw(rotas, customers, dist, peso_veiculo))
    return fitnesses


def selecionar(populacao, fitnesses, n):
    ordenados = sorted(zip(fitnesses, populacao), key=lambda x: x[0])
    melhores_fit = [f for f, _ in ordenados[:n]]
    melhores_pop = [ind for _, ind in ordenados[:n]]
    return melhores_pop, melhores_fit


def clonar(selecionados, fitnesses, tam_pop, beta=1.0):
    clones      = []
    clones_fit  = []
    for rank, (ind, fit) in enumerate(zip(selecionados, fitnesses), start=1):
        n_clones = max(1, int(beta * tam_pop / rank))
        clones.extend([ind[:] for _ in range(n_clones)])
        clones_fit.extend([fit] * n_clones)
    return clones, clones_fit


def hipermutacao(clones, clones_fit, p=5.0):
    """
    Hipermutação por inversão de segmento (2-opt).
    Alta afinidade (fitness baixo) → taxa de mutação baixa.
    """
    arr = np.array(clones_fit, dtype=float)
    f_min, f_max = arr.min(), arr.max()

    if f_max == f_min:
        afinidades = np.ones(len(clones))
    else:
        # normaliza: menor fitness → maior afinidade (mais próximo de 1)
        afinidades = 1.0 - (arr - f_min) / (f_max - f_min)

    taxas = np.exp(-p * afinidades)

    mutados = []
    for clone, taxa in zip(clones, taxas):
        novo = clone[:]
        n = len(novo)
        for i in range(n):
            if random.random() < taxa:
                j = random.randint(i, n - 1)
                novo[i:j+1] = novo[i:j+1][::-1]
        mutados.append(novo)
    return mutados


def re_selecionar(populacao, fitnesses, mutados, customers, dist, capacity,
                  peso_veiculo, tam_pop):
    fit_mutados = avaliar_pop(mutados, customers, dist, capacity, peso_veiculo)

    pool     = populacao + mutados
    pool_fit = fitnesses + fit_mutados

    ordenados  = sorted(zip(pool_fit, pool), key=lambda x: x[0])
    nova_pop   = [ind for _, ind in ordenados[:tam_pop]]
    novas_fit  = [f   for f, _  in ordenados[:tam_pop]]
    return nova_pop, novas_fit


def clonalg_vrptw(arquivo, tam_pop=150, n_sel=25, n_geracoes=100,
                  beta=1.0, p=5.0, d=20, peso_veiculo=2000.0,
                  frac_nn=0.2, seed=42,
                  caminho_grafico="convergencia_clonalg_vrptw.png"):

    random.seed(seed)
    np.random.seed(seed)

    customers, dist, num_vehicles, capacity = ler_instancia(arquivo)

    pop  = gerar_pop_vrptw(customers, dist, tam_pop, frac_nn)
    fits = avaliar_pop(pop, customers, dist, capacity, peso_veiculo)

    melhor_global_fit = float('inf')
    melhor_global_ind = None
    historico_fit     = []
    historico_veic    = []
    historico_dist    = []

    for geracao in range(n_geracoes):
        # 1. Selecionar os n_sel melhores anticorpos
        sel, sel_fit = selecionar(pop, fits, n_sel)

        # 2. Clonar proporcionalmente ao ranking
        clones, clones_fit = clonar(sel, sel_fit, tam_pop, beta)

        # 3. Hipermutação por inversão
        mutados = hipermutacao(clones, clones_fit, p)

        # 4. Re-selecionar melhores entre população atual e clones mutados
        pop, fits = re_selecionar(pop, fits, mutados, customers, dist,
                                  capacity, peso_veiculo, tam_pop)

        # 5. Substituir os d piores por anticorpos aleatórios (diversidade)
        if d > 0:
            novos     = gerar_pop_vrptw(customers, dist, d, frac_nn=0.0)
            novos_fit = avaliar_pop(novos, customers, dist, capacity, peso_veiculo)
            pop[-d:]  = novos
            fits[-d:] = novos_fit

        melhor_fit = min(fits)
        melhor_ind = pop[fits.index(melhor_fit)]

        if melhor_fit < melhor_global_fit:
            melhor_global_fit = melhor_fit
            melhor_global_ind = melhor_ind[:]

        rotas_melhor = decoder_max_fill(melhor_ind, customers, dist, capacity)
        dist_melhor, veic_melhor = avaliar_vrptw(rotas_melhor, customers, dist)

        historico_fit.append(melhor_global_fit)
        historico_veic.append(veic_melhor)
        historico_dist.append(dist_melhor)

        print(f"Geração {geracao+1:03d} | Veículos: {veic_melhor} | "
              f"Distância: {dist_melhor:.4f} | Fitness: {melhor_fit:.4f}")

    rotas_finais = decoder_max_fill(melhor_global_ind, customers, dist, capacity)
    dist_final, veic_final = avaliar_vrptw(rotas_finais, customers, dist)

    print("\n========================================")
    print(f"Melhor solução: {veic_final} veículos | Distância: {dist_final:.4f}")
    for i, rota in enumerate(rotas_finais, 1):
        print(f"  Rota {i}: {' '.join(map(str, rota))}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(historico_dist, color='b')
    ax1.set_xlabel("Geração")
    ax1.set_ylabel("Distância total")
    ax1.set_title(f"CLONALG-VRPTW | pop={tam_pop} sel={n_sel} β={beta} p={p} seed={seed}")
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(historico_veic, color='r')
    ax2.set_xlabel("Geração")
    ax2.set_ylabel("Número de veículos")
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
    plt.close()

    return rotas_finais, dist_final, veic_final


if __name__ == '__main__':
    clonalg_vrptw("Instancias_teste/rc208.txt", n_geracoes=200, seed=42)
