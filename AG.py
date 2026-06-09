import random
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
import math
from dataclasses import dataclass


@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: int
    ready_time: int
    due_date: int
    service_time: int


def ler_instancia(arquivo: str) -> tuple:
    customers = []
    num_vehicles = 0
    capacity = 0

    with open(arquivo, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == 'VEHICLE':
            i += 1
            # skip header line ("NUMBER     CAPACITY")
            while i < len(lines) and not lines[i].strip():
                i += 1
            i += 1  # skip "NUMBER     CAPACITY"
            while i < len(lines) and not lines[i].strip():
                i += 1
            parts = lines[i].split()
            num_vehicles, capacity = int(parts[0]), int(parts[1])

        elif line == 'CUSTOMER':
            i += 1
            # skip header line ("CUST NO.   XCOORD. ...")
            while i < len(lines) and not lines[i].strip():
                i += 1
            i += 1  # skip column header
            while i < len(lines):
                row = lines[i].strip()
                if row:
                    parts = row.split()
                    customers.append(Customer(
                        id=int(parts[0]),
                        x=float(parts[1]),
                        y=float(parts[2]),
                        demand=int(parts[3]),
                        ready_time=int(parts[4]),
                        due_date=int(parts[5]),
                        service_time=int(parts[6]),
                    ))
                i += 1
            break

        i += 1

    n = len(customers)
    dist = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            dx = customers[a].x - customers[b].x
            dy = customers[a].y - customers[b].y
            dist[a][b] = round(math.sqrt(dx * dx + dy * dy), 4)

    return customers, dist, num_vehicles, capacity


def decoder(giant_tour: list, customers: list, dist: np.ndarray, capacity: int) -> list:
    """
    Converte um giant tour (permutação de IDs de clientes) em uma lista de rotas.
    Cada rota é uma lista de IDs de clientes (sem o depósito).
    Quebra a rota atual quando capacidade ou janela de tempo seria violada.
    """
    rotas = []
    rota_atual = []
    carga_atual = 0
    tempo_atual = 0.0
    prev = 0  # depósito é índice 0

    for cid in giant_tour:
        c = customers[cid]
        arrival = tempo_atual + dist[prev][cid]
        # Espera se chegou antes da janela abrir
        inicio_servico = max(arrival, float(c.ready_time))

        capacidade_ok = (carga_atual + c.demand) <= capacity
        janela_ok = inicio_servico <= c.due_date
        retorno_ok = (inicio_servico + c.service_time + dist[cid][0]) <= customers[0].due_date

        if capacidade_ok and janela_ok and retorno_ok:
            rota_atual.append(cid)
            carga_atual += c.demand
            tempo_atual = inicio_servico + c.service_time
            prev = cid
        else:
            # Fecha rota atual e abre nova
            if rota_atual:
                rotas.append(rota_atual)
            rota_atual = [cid]
            carga_atual = c.demand
            # Recalcula saindo do depósito
            arrival = dist[0][cid]
            inicio_servico = max(arrival, float(c.ready_time))
            tempo_atual = inicio_servico + c.service_time
            prev = cid

    if rota_atual:
        rotas.append(rota_atual)

    return rotas


def decoder_max_fill(giant_tour: list, customers: list, dist: np.ndarray, capacity: int) -> list:
    """
    Decoder que maximiza o preenchimento de cada rota antes de abrir a próxima.
    Para cada rota, varre todos os clientes ainda não alocados e insere os que
    couberem (capacidade + janelas + retorno), na ordem em que aparecem no tour.
    Minimiza número de veículos antes de considerar distância.
    """
    nao_alocados = list(giant_tour)
    rotas = []
    depot_due = customers[0].due_date

    while nao_alocados:
        rota_atual = []
        carga_atual = 0
        tempo_atual = 0.0
        prev = 0
        restantes = []

        for cid in nao_alocados:
            c = customers[cid]
            arrival = tempo_atual + dist[prev][cid]
            inicio_servico = max(arrival, float(c.ready_time))

            capacidade_ok = (carga_atual + c.demand) <= capacity
            janela_ok = inicio_servico <= c.due_date
            retorno_ok = (inicio_servico + c.service_time + dist[cid][0]) <= depot_due

            if capacidade_ok and janela_ok and retorno_ok:
                rota_atual.append(cid)
                carga_atual += c.demand
                tempo_atual = inicio_servico + c.service_time
                prev = cid
            else:
                restantes.append(cid)

        rotas.append(rota_atual)
        nao_alocados = restantes

    return rotas


def avaliar_vrptw(rotas: list, customers: list, dist: np.ndarray) -> tuple:
    """
    Calcula distância total e número de veículos de uma solução decodificada.
    Retorna (distancia_total, num_veiculos).
    """
    distancia_total = 0.0
    for rota in rotas:
        prev = 0
        for cid in rota:
            distancia_total += dist[prev][cid]
            prev = cid
        distancia_total += dist[prev][0]  # retorno ao depósito
    return round(distancia_total, 4), len(rotas)


def fitness_vrptw(rotas: list, customers: list, dist: np.ndarray, peso_veiculo: float = 1000.0) -> float:
    """
    Função objetivo combinada: minimiza distância + penalidade por veículo extra.
    peso_veiculo deve ser maior que a distância máxima possível para garantir
    que reduzir um veículo sempre supere qualquer ganho em distância.
    """
    distancia, num_veiculos = avaliar_vrptw(rotas, customers, dist)
    return distancia + peso_veiculo * num_veiculos


def _rota_valida(rota: list, customers: list, dist: np.ndarray, capacity: int) -> bool:
    """Verifica se uma rota respeita capacidade, janelas de tempo e retorno ao depósito."""
    carga = 0
    t = 0.0
    prev = 0
    depot_due = customers[0].due_date
    for cid in rota:
        c = customers[cid]
        carga += c.demand
        if carga > capacity:
            return False
        arrival = t + dist[prev][cid]
        t = max(arrival, float(c.ready_time))
        if t > c.due_date:
            return False
        t += c.service_time
        prev = cid
    return (t + dist[prev][0]) <= depot_due


def or_opt(rotas: list, customers: list, dist: np.ndarray, capacity: int) -> list:
    """
    Busca local: tenta eliminar a menor rota relocando cada um de seus clientes
    para a melhor posição factível em outra rota. Repete até não conseguir eliminar
    mais nenhuma rota.
    """
    rotas = [r[:] for r in rotas]  # cópia para não modificar o original

    melhorou = True
    while melhorou:
        melhorou = False
        # Ordena por tamanho para tentar eliminar as menores primeiro
        rotas.sort(key=len)

        for idx_menor in range(len(rotas)):
            rota_menor = rotas[idx_menor]
            clientes_relocados = []

            for cid in rota_menor:
                melhor_rota_idx = None
                melhor_pos = None
                melhor_custo = float('inf')

                for idx_outra in range(len(rotas)):
                    if idx_outra == idx_menor:
                        continue
                    rota_outra = rotas[idx_outra]

                    # Tenta inserir cid em cada posição da rota_outra
                    for pos in range(len(rota_outra) + 1):
                        candidata = rota_outra[:pos] + [cid] + rota_outra[pos:]
                        if _rota_valida(candidata, customers, dist, capacity):
                            # Custo de inserção: distância adicionada
                            prev_id = 0 if pos == 0 else rota_outra[pos - 1]
                            next_id = 0 if pos == len(rota_outra) else rota_outra[pos]
                            custo = (dist[prev_id][cid] + dist[cid][next_id]
                                     - dist[prev_id][next_id])
                            if custo < melhor_custo:
                                melhor_custo = custo
                                melhor_rota_idx = idx_outra
                                melhor_pos = pos

                if melhor_rota_idx is not None:
                    clientes_relocados.append((cid, melhor_rota_idx, melhor_pos))

            # Só elimina a rota se TODOS os clientes encontraram destino
            if len(clientes_relocados) == len(rota_menor):
                # Insere em ordem reversa de posição para não deslocar índices
                inserções = sorted(clientes_relocados, key=lambda x: (x[1], -x[2]))
                for cid, idx_outra, pos in inserções:
                    rotas[idx_outra].insert(pos, cid)
                rotas.pop(idx_menor)
                melhorou = True
                break  # recomeça com as rotas atualizadas

    return rotas


def nearest_neighbor_tour(customers: list, dist: np.ndarray) -> list:
    """
    Gera um giant tour greedy: sempre escolhe o cliente não visitado mais próximo.
    Serve como semente de qualidade para a população inicial.
    """
    nao_visitados = list(range(1, len(customers)))  # exclui depósito (0)
    tour = []
    atual = 0
    while nao_visitados:
        proximo = min(nao_visitados, key=lambda c: dist[atual][c])
        tour.append(proximo)
        nao_visitados.remove(proximo)
        atual = proximo
    return tour


def gerar_pop_vrptw(customers: list, dist: np.ndarray, tam_pop: int, frac_nn: float = 0.2) -> list:
    """
    Gera população inicial mista:
      - frac_nn * tam_pop indivíduos via nearest neighbor (com perturbação aleatória)
      - restante aleatório
    Cada indivíduo é um giant tour: lista com IDs dos clientes 1..N.
    """
    ids_clientes = list(range(1, len(customers)))
    populacao = []

    n_nn = int(tam_pop * frac_nn)
    base_nn = nearest_neighbor_tour(customers, dist)

    for _ in range(n_nn):
        tour = base_nn[:]
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
        populacao.append(tour)

    for _ in range(tam_pop - n_nn):
        tour = ids_clientes[:]
        random.shuffle(tour)
        populacao.append(tour)

    return populacao


### caixeiro viajante

def avaliar(df_dist, populacao, tam_pop):
    distancias_pop = []
    
    for r in range(tam_pop):
        individuo = populacao[r]
        dist = 0
        n = len(individuo)
        
        # Percorre o indivíduo calculando a distância entre cidades adjacentes
        for i in range(n - 1):
            cidade_atual = individuo[i]
            proxima_cidade = individuo[i+1]
            
            dist += df_dist.at[cidade_atual, proxima_cidade]
        #adicionar volta para primeira cidade
        dist += df_dist.at[individuo[-1], individuo[0]]
        
        distancias_pop.append(dist)   
    
    return distancias_pop

def gerar_pop(n_cidades, tam=50):
    pop = []
    for i in range(tam):
        vetor_ = list(range(1, n_cidades + 1))
        random.shuffle(vetor_)
        pop.append(vetor_)
    return pop

def ler_matriz(arquivo, n_cidades=15):
    with open(arquivo, 'r') as f:
        lines = [line for line in f if not line.strip().startswith('#')]
        content = " ".join(lines)

    numbers = re.findall(r'\b\d+\b', content)
    numbers = [int(n) for n in numbers]

    total = n_cidades ** 2
    if len(numbers) > total:
        matrix_data = numbers[1:total + 1]
    else:
        matrix_data = numbers[:total]

    matrix_array = np.array(matrix_data).reshape(n_cidades, n_cidades)

    matrix_df = pd.DataFrame(
        matrix_array,
        index=range(1, n_cidades + 1),
        columns=range(1, n_cidades + 1)
    )

    return matrix_df

import random

def roleta(individuos, distancias_pop, tam_pop):
    vetorPais = []
    
    # Inversão: Cidades com menor distância ganham valores maiores
    inversao = [1.0 / (d) for d in distancias_pop]
    
    # Cálculo das probabilidades (pesos)
    soma_inversao = sum(inversao)
    pesos = [inv / soma_inversao for inv in inversao]
    
    # Construção da Roleta (Soma acumulada)
    vetor_roleta = []
    soma_acumulada = 0
    for peso in pesos:
        soma_acumulada += peso
        vetor_roleta.append(soma_acumulada)
    
    # Garantir que o último valor seja exatamente 1.0 para evitar erros de precisão
    vetor_roleta[-1] = 1.0

    for _ in range(tam_pop):
        r = random.random() # Gera valor entre 0 e 1
        
        # Encontra o primeiro índice onde o valor acumulado é maior que o sorteado
        for idx, limite in enumerate(vetor_roleta):
            if r <= limite:
                vetorPais.append(individuos[idx])
                break
                
    return vetorPais

def ox_crossover(vetor_pais):
    nova_populacao = []
    n = len(vetor_pais[0])

    for i in range(0, len(vetor_pais), 2):
        pai1 = vetor_pais[i]
        pai2 = vetor_pais[i + 1] if i + 1 < len(vetor_pais) else vetor_pais[0]

        ponto1 = random.randint(0, n - 2)
        ponto2 = random.randint(ponto1 + 1, n - 1)

        for p1, p2 in [(pai1, pai2), (pai2, pai1)]:
            filho = [None] * n
            filho[ponto1:ponto2 + 1] = p1[ponto1:ponto2 + 1]

            segmento = set(filho[ponto1:ponto2 + 1])

            ordem_p2 = p2[ponto2 + 1:] + p2[:ponto2 + 1]
            cidades_restantes = [c for c in ordem_p2 if c not in segmento]

            posicoes = [(ponto2 + 1 + k) % n for k in range(n - (ponto2 - ponto1 + 1))]

            for pos, cidade in zip(posicoes, cidades_restantes):
                filho[pos] = cidade

            nova_populacao.append(filho)

    return nova_populacao

def cx_crossover(vetor_pais):
    nova_populacao = []
    n = len(vetor_pais[0])

    for i in range(0, len(vetor_pais), 2):
        pai1 = vetor_pais[i]
        pai2 = vetor_pais[i + 1] if i + 1 < len(vetor_pais) else vetor_pais[0]

        pos_em_pai2 = {cidade: idx for idx, cidade in enumerate(pai2)}

        for p1, p2 in [(pai1, pai2), (pai2, pai1)]:
            pos_em_p2 = {cidade: idx for idx, cidade in enumerate(p2)}

            filho = [None] * n
            ciclo = set()
            pos = 0
            while pos not in ciclo:
                ciclo.add(pos)
                pos = pos_em_p2[p1[pos]]

            for idx in range(n):
                filho[idx] = p1[idx] if idx in ciclo else p2[idx]

            nova_populacao.append(filho)

    return nova_populacao

def bcrc_crossover(vetor_pais, customers, dist, capacity):
    """
    Best Cost Route Crossover: herda uma rota inteira de cada pai.
    Opera sobre rotas decodificadas, preservando agrupamentos viáveis.
    """
    nova_populacao = []

    for i in range(0, len(vetor_pais), 2):
        pai1 = vetor_pais[i]
        pai2 = vetor_pais[(i + 1) % len(vetor_pais)]

        rotas1 = decoder_max_fill(pai1, customers, dist, capacity)
        rotas2 = decoder_max_fill(pai2, customers, dist, capacity)

        for rota_herdada, rotas_restante in [(random.choice(rotas1), rotas2),
                                             (random.choice(rotas2), rotas1)]:
            herdados = set(rota_herdada)

            # Mantém a ordem dos clientes restantes conforme aparecem nas rotas do outro pai
            restante = [cid for rota in rotas_restante for cid in rota if cid not in herdados]

            filho = rota_herdada + restante
            nova_populacao.append(filho)

    return nova_populacao


def mutacao(populacao, taxa=0.1):
    for individuo in populacao:
        if random.random() < taxa:
            i, j = sorted(random.sample(range(len(individuo)), 2))
            individuo[i:j+1] = individuo[i:j+1][::-1]
    return populacao

def elitismo(populacao, distancias, nova_populacao, novas_distancias, n_elite=2):
    pares = sorted(zip(distancias, populacao), key=lambda x: x[0])
    elite = [ind for _, ind in pares[:n_elite]]

    pares_novos = sorted(zip(novas_distancias, nova_populacao), key=lambda x: x[0], reverse=True)
    nova_populacao_final = [ind for _, ind in pares_novos[n_elite:]]

    return elite + nova_populacao_final


def main(arquivo, n_cidades, tam_pop, n_geracoes, taxa_mutacao, n_elite, cruzamento, seed, caminho_grafico):
    random.seed(seed)

    df_dist   = ler_matriz(arquivo, n_cidades)
    populacao = gerar_pop(n_cidades, tam_pop)

    fn_crossover = ox_crossover if cruzamento == "OX" else cx_crossover

    historico = []

    for geracao in range(n_geracoes):
        distancias  = avaliar(df_dist, populacao, len(populacao))
        melhor_dist = min(distancias)
        historico.append(melhor_dist)
        print(f"  [{cruzamento}] Geração {geracao + 1:03d} | Melhor: {melhor_dist}")

        pais   = roleta(populacao, distancias, tam_pop)
        filhos = fn_crossover(pais)
        filhos = mutacao(filhos, taxa_mutacao)
        filhos = filhos[:tam_pop]

        novas_distancias = avaliar(df_dist, filhos, len(filhos))
        populacao        = elitismo(populacao, distancias, filhos, novas_distancias, n_elite)

    distancias_finais = avaliar(df_dist, populacao, len(populacao))
    melhor_idx        = distancias_finais.index(min(distancias_finais))
    melhor_distancia  = distancias_finais[melhor_idx]

    plt.figure()
    plt.plot(historico)
    plt.xlabel("Geração")
    plt.ylabel("Melhor distância")
    plt.title(f"Convergência | pop={tam_pop} mut={taxa_mutacao} elite={n_elite} {cruzamento} seed={seed}")
    plt.tight_layout()
    plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
    plt.close()

    return melhor_distancia


def torneio_vrptw(populacao, fitnesses, tam_pop, pv=0.8):
    """Seleção por torneio para minimização."""
    n = len(populacao)
    pais = []
    for _ in range(tam_pop):
        i1 = random.randrange(n)
        i2 = random.randrange(n)
        while i2 == i1:
            i2 = random.randrange(n)
        melhor, pior = (i1, i2) if fitnesses[i1] < fitnesses[i2] else (i2, i1)
        vencedor = melhor if random.random() < pv else pior
        pais.append(populacao[vencedor][:])
    return pais


def main_vrptw(
    arquivo,
    tam_pop=200,
    n_geracoes=200,
    taxa_mutacao=0.05,
    n_elite=2,
    cruzamento="OX",
    peso_veiculo=2000.0,
    frac_nn=0.2,
    seed=42,
    decoder_fn=None,  # decoder (padrão) ou decoder_max_fill
    usar_or_opt=False,
    caminho_grafico="convergencia_vrptw.png",
):
    random.seed(seed)

    customers, dist, num_vehicles, capacity = ler_instancia(arquivo)

    fn_decoder = decoder_fn if decoder_fn is not None else decoder

    def decodificar(ind):
        rotas = fn_decoder(ind, customers, dist, capacity)
        if usar_or_opt:
            rotas = or_opt(rotas, customers, dist, capacity)
        return rotas
    populacao = gerar_pop_vrptw(customers, dist, tam_pop, frac_nn)
    if cruzamento == "OX":
        fn_crossover = ox_crossover
    elif cruzamento == "CX":
        fn_crossover = cx_crossover
    else:  # BCRC
        fn_crossover = lambda pais: bcrc_crossover(pais, customers, dist, capacity)

    historico_fitness = []
    historico_veiculos = []
    historico_distancia = []

    melhor_global_tour = None
    melhor_global_fitness = float('inf')

    for geracao in range(n_geracoes):
        # Avalia toda a população (decoder + or-opt + fitness)
        fitnesses = []
        rotas_populacao = []
        for ind in populacao:
            rotas = decodificar(ind)
            rotas_populacao.append(rotas)
            fitnesses.append(fitness_vrptw(rotas, customers, dist, peso_veiculo))

        melhor_idx = fitnesses.index(min(fitnesses))
        melhor_fit = fitnesses[melhor_idx]
        rotas_melhor = rotas_populacao[melhor_idx]
        dist_melhor, veic_melhor = avaliar_vrptw(rotas_melhor, customers, dist)

        historico_fitness.append(melhor_fit)
        historico_veiculos.append(veic_melhor)
        historico_distancia.append(dist_melhor)

        if melhor_fit < melhor_global_fitness:
            melhor_global_fitness = melhor_fit
            melhor_global_tour = populacao[melhor_idx][:]

        print(f"Geração {geracao + 1:03d} | Veículos: {veic_melhor} | Distância: {dist_melhor:.4f} | Fitness: {melhor_fit:.4f}")

        pais = torneio_vrptw(populacao, fitnesses, tam_pop)
        filhos = fn_crossover(pais)
        filhos = mutacao(filhos, taxa_mutacao)
        filhos = filhos[:tam_pop]

        novas_fitnesses = []
        for ind in filhos:
            rotas = decodificar(ind)
            novas_fitnesses.append(fitness_vrptw(rotas, customers, dist, peso_veiculo))

        populacao = elitismo(populacao, fitnesses, filhos, novas_fitnesses, n_elite)

    # Resultado final com or-opt
    rotas_finais = decodificar(melhor_global_tour)
    dist_final, veic_final = avaliar_vrptw(rotas_finais, customers, dist)

    print("\n========================================")
    print(f"Melhor solução: {veic_final} veículos | Distância: {dist_final:.4f}")
    for i, rota in enumerate(rotas_finais, 1):
        print(f"  Rota {i}: {' '.join(map(str, rota))}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(historico_distancia, color='b')
    ax1.set_xlabel("Geração")
    ax1.set_ylabel("Distância total")
    ax1.set_title(f"Convergência VRPTW | pop={tam_pop} mut={taxa_mutacao} elite={n_elite} {cruzamento} seed={seed}")
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(historico_veiculos, color='r')
    ax2.set_xlabel("Geração")
    ax2.set_ylabel("Número de veículos")
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
    plt.close()

    return dist_final, veic_final, rotas_finais


if __name__ == '__main__':
    main_vrptw(
        arquivo="Instancias_teste/c1_2_1.txt",
        tam_pop=100,
        n_geracoes=200,
        taxa_mutacao=0.05,
        n_elite=2,
        cruzamento="OX",
        seed=42,
        decoder_fn=decoder_max_fill
    )
