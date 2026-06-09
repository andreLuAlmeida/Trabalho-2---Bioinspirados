import random
import matplotlib.pyplot as plt
from AG import ler_instancia, avaliar_vrptw

alpha = 1.0
beta  = 2
rho   = 1.2
Q     = 100000
tau_inicial  = 1e-6
n_formigas   = 10
peso_veiculo = 2000.0  # penalidade por veículo extra no custo


def gera_feromonios(n):
    return [[tau_inicial if i != j else 0.0 for j in range(n)] for i in range(n)]


def _clientes_viaveis(visitados, prev, tempo, carga, customers, dist, capacity):
    """Retorna lista de (cid, arrival, inicio_servico) factíveis para inserir na rota atual."""
    viaveis = []
    depot_due = customers[0].due_date
    for cid in range(1, len(customers)):
        if cid in visitados:
            continue
        c = customers[cid]
        arrival = tempo + dist[prev][cid]
        inicio  = max(arrival, float(c.ready_time))
        if (carga + c.demand <= capacity
                and inicio <= c.due_date
                and inicio + c.service_time + dist[cid][0] <= depot_due):
            viaveis.append((cid, arrival, inicio))
    return viaveis


def _escolher_proximo(prev, viaveis, feromonios, dist):
    """Roleta sobre clientes viáveis usando feromônio × heurística."""
    pesos = []
    for cid, _, _ in viaveis:
        tau_ij = feromonios[prev][cid] ** alpha
        eta_ij = (1.0 / dist[prev][cid]) ** beta if dist[prev][cid] > 0 else 1.0
        pesos.append(tau_ij * eta_ij)

    soma = sum(pesos)
    if soma == 0:
        return random.choice(viaveis)

    r = random.random()
    acum = 0.0
    for (cid, arrival, inicio), p in zip(viaveis, pesos):
        acum += p / soma
        if r <= acum:
            return cid, arrival, inicio
    return viaveis[-1][0], viaveis[-1][1], viaveis[-1][2]


def constroi_solucao(customers, dist, feromonios, capacity):
    """
    Uma formiga constrói uma solução completa:
    abre rotas sucessivas até alocar todos os clientes.
    Retorna lista de rotas (cada rota = lista de IDs de clientes).
    """
    visitados = set()
    rotas = []
    n_clientes = len(customers) - 1  # exclui depósito

    while len(visitados) < n_clientes:
        rota   = []
        prev   = 0
        tempo  = 0.0
        carga  = 0

        while True:
            viaveis = _clientes_viaveis(visitados, prev, tempo, carga,
                                        customers, dist, capacity)
            if not viaveis:
                break

            cid, _, inicio = _escolher_proximo(prev, viaveis, feromonios, dist)

            rota.append(cid)
            visitados.add(cid)
            carga  += customers[cid].demand
            tempo   = inicio + customers[cid].service_time
            prev    = cid

        if rota:
            rotas.append(rota)
        else:
            # Segurança: cliente não alocável isoladamente — força rota unitária
            restantes = [c for c in range(1, len(customers)) if c not in visitados]
            if restantes:
                rotas.append([restantes[0]])
                visitados.add(restantes[0])

    return rotas


def custo_solucao(rotas, customers, dist):
    """Custo = distância total + peso_veiculo × número de rotas."""
    dist_total, n_veic = avaliar_vrptw(rotas, customers, dist)
    return dist_total + peso_veiculo * n_veic, dist_total, n_veic


def atualiza_feromonios(feromonios, solucoes, custos):
    n = len(feromonios)

    # Evaporação
    for i in range(n):
        for j in range(n):
            feromonios[i][j] *= (1 - rho)

    # Depósito proporcional à qualidade
    for rotas, custo in zip(solucoes, custos):
        deposito = Q / custo
        for rota in rotas:
            prev = 0
            for cid in rota:
                feromonios[prev][cid] += deposito
                feromonios[cid][prev] += deposito
                prev = cid
            # retorno ao depósito
            feromonios[prev][0] += deposito
            feromonios[0][prev] += deposito


def ACO_VRPTW(arquivo, iteracoes=100):
    customers, dist, num_vehicles, capacity = ler_instancia(arquivo)
    n = len(customers)

    feromonios = gera_feromonios(n)

    melhor_custo    = float('inf')
    melhor_rotas    = None
    historico_custo = []
    historico_veic  = []
    historico_dist  = []

    for it in range(iteracoes):
        solucoes = []
        custos   = []

        for _ in range(n_formigas):
            rotas = constroi_solucao(customers, dist, feromonios, capacity)
            custo, _, _ = custo_solucao(rotas, customers, dist)
            solucoes.append(rotas)
            custos.append(custo)

        idx_melhor = custos.index(min(custos))
        custo_iter = custos[idx_melhor]
        rotas_iter = solucoes[idx_melhor]
        _, dist_iter, veic_iter = custo_solucao(rotas_iter, customers, dist)

        if custo_iter < melhor_custo:
            melhor_custo = custo_iter
            melhor_rotas = rotas_iter

        historico_custo.append(melhor_custo)
        historico_veic.append(veic_iter)
        historico_dist.append(dist_iter)

        print(f"Iter {it+1:03d} | Veículos: {veic_iter} | Distância: {dist_iter:.4f} | Fitness: {custo_iter:.4f}")

        atualiza_feromonios(feromonios, solucoes, custos)

    _, dist_final, veic_final = custo_solucao(melhor_rotas, customers, dist)
    print("\n========================================")
    print(f"Melhor solução: {veic_final} veículos | Distância: {dist_final:.4f}")
    for i, rota in enumerate(melhor_rotas, 1):
        print(f"  Rota {i}: {' '.join(map(str, rota))}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(historico_dist, color='b')
    ax1.set_xlabel("Iteração")
    ax1.set_ylabel("Distância total")
    ax1.set_title(f"ACO-VRPTW | formigas={n_formigas} α={alpha} β={beta} ρ={rho}")
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(historico_veic, color='r')
    ax2.set_xlabel("Iteração")
    ax2.set_ylabel("Número de veículos")
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("convergencia_aco_vrptw.png", dpi=300, bbox_inches='tight')
    plt.close()

    return melhor_rotas, dist_final, veic_final


if __name__ == '__main__':
    ACO_VRPTW("Instancias_teste/c1_2_1.txt", iteracoes=500)
