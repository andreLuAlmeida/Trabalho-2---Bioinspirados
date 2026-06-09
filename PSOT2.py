import random
import matplotlib.pyplot as plt
import math
import numpy as np

# PARÂMETROS
N = 50
MAX_GERACOES = 500
c1 = 2.05
c2 = 2.05
w = 0.4
vmax = 2.0

#Ler instância do arquivo
def ler_instancia(arquivo):
    clientes = []
    capacidade = 0
    num_veiculos = 0

    # Lê o arquivo linha por linha
    with open(arquivo, "r") as f:
        linhas = f.readlines()
    lendo_clientes = False
    #Percorre as linhas do arquivo para extrair informações
    for linha in linhas:
        linha = linha.strip()
        # Ignora linhas vazias ou comentários
        if not linha:
            continue
        partes = linha.split()
        # Verifica se a linha contém informações sobre número de veículos e capacidade
        if len(partes) == 2 and partes[0].isdigit():
            if capacidade == 0:
                num_veiculos = int(partes[0])
                capacidade = int(partes[1])
                continue
        # início da tabela de clientes
        if linha.startswith("CUSTOMER"):
            lendo_clientes = True
            continue
        # pula cabeçalhos
        if ("CUST" in linha or "XCOORD" in linha or "NUMBER" in linha or "CAPACITY" in linha):
            continue
        # lê os clientes
        if lendo_clientes:
            if len(partes) >= 7:
                cliente = [
                    int(partes[0]),   # id
                    float(partes[1]), # x
                    float(partes[2]), # y
                    int(partes[3]),   # demanda
                    int(partes[4]),   # ready
                    int(partes[5]),   # due
                    int(partes[6])    # service
                ]
                clientes.append(cliente)
    return clientes, capacidade, num_veiculos

#matriz de distância entre os clientes
def gerar_matriz_distancia(clientes):
    n = len(clientes)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dx = clientes[i][1] - clientes[j][1]
            dy = clientes[i][2] - clientes[j][2]
            dist[i][j] = math.sqrt(dx*dx + dy*dy)
    return dist

# Inicialização da população de particulas e velocidade, PSO
def inicializar_populacao(n_particulas, n_clientes):
    populacao = []
    velocidade = []

    for _ in range(n_particulas):
        particula = [ random.random() for _ in range(n_clientes)]
        vel = [random.uniform(-0.2,0.2)for _ in range(n_clientes)]
        populacao.append(particula)
        velocidade.append(vel)
    return populacao, velocidade

#Decodifica as chaves aleatórias de particulas para obter a sequência de clientes
#Adapta PSO para VRPTW
def decode_random_keys(keys):
    return sorted(range(len(keys)), key=lambda i: keys[i])

#Calcula a quantidade de veículos e suas rotas
#VRPTW
def decoder(rota, clientes, dist, capacidade):
    rotas = []
    atual = []
    carga = 0
    tempo = 0
    anterior = 0  # depósito

    for cliente in rota:
        c = clientes[cliente]
        chegada = tempo + dist[anterior][cliente]
        inicio = max(chegada, c[4])  
        if carga + c[3] <= capacidade and inicio <= c[5]:
            atual.append(c[0])
            carga += c[3]
            tempo = inicio + c[6]
            anterior = cliente
        else:
            if atual:
                rotas.append(atual)

            atual = [cliente]
            carga = c[3]
            tempo = c[6]
            anterior = cliente
    if atual:
        rotas.append(atual)
    return rotas

#Faz o calculo do fitness considerando a distância total e penalidades por violação de capacidade e janela de tempo
#VRPTW
def fitness_vrptw(rota_total, clientes, dist, capacidade):
    rotas = decoder(rota_total, clientes, dist, capacidade)
    distancia_total = 0
    penalidade = 0

    for r in rotas:
        if len(r) == 0:
            continue
        carga = 0
        tempo = 0
        anterior = 0 

        for c in r:
            idx = c-1
            cliente = clientes[idx]
            distancia_total += dist[anterior][idx]
            carga += cliente[3]
            chegada = tempo + dist[anterior][idx]
            inicio = max(chegada, cliente[4])
            #penalidade por violação de janela de tempo
            if inicio > cliente[5]:
                penalidade += len(rotas)*500  
            tempo = inicio + cliente[6]
            anterior = idx
        distancia_total += dist[anterior][0]  # volta depósito
        #penalidade por violação de capacidade
        if carga > capacidade:
            penalidade += len(rotas)*500  
    return distancia_total + penalidade

#Cacula o fitness de cada partícula, VRPTW
def avaliar(particula, clientes, dist, capacidade):
    rota_total = decode_random_keys(particula)
    return fitness_vrptw(rota_total, clientes, dist, capacidade)

#Inicializa PBest e GBest, praticamento PSO original
def inicializar_melhores(populacao, clientes, dist, capacidade):
    pbest = [particula[:] for particula in populacao]
    fitness_pbest = [avaliar(particula, clientes, dist, capacidade) for particula in populacao]
    indice_melhor = np.argmin(fitness_pbest)
    gbest = pbest[indice_melhor][:]
    fitness_gbest = fitness_pbest[indice_melhor]
    return pbest, fitness_pbest, gbest, fitness_gbest

#Atualiza PBest e GBest, praticamento PSO original
def atualizar_melhores(populacao, pbest, fitness_pbest, gbest, fitness_gbest, clientes, dist, capacidade):
    melhorou = False

    for i in range(len(populacao)):
        fitness_atual = avaliar(populacao[i], clientes, dist, capacidade)
        if fitness_atual < fitness_pbest[i]:
            fitness_pbest[i] = fitness_atual
            pbest[i] = populacao[i][:]
            if fitness_atual < fitness_gbest:
                fitness_gbest = fitness_atual
                gbest = populacao[i][:]
                melhorou = True
    return pbest, fitness_pbest, gbest, fitness_gbest, melhorou

#atualiza velocida, PSO original
def atualizar_velocidade(populacao, velocidade, pbest, gbest):
    for i in range(len(populacao)):
        for j in range(len(populacao[i])):
            r1 = random.random()
            r2 = random.random()
            velocidade[i][j] = (w * velocidade[i][j] + c1 * r1 * (pbest[i][j] - populacao[i][j]) + c2 * r2 * (gbest[j] - populacao[i][j]))
            velocidade[i][j] = max( -vmax, min(vmax, velocidade[i][j]))

#atualiza posições, PSO original
def atualizar_posicoes(populacao, velocidade):
    for i in range(len(populacao)):
        for j in range(len(populacao[i])):
            populacao[i][j] += velocidade[i][j]
            populacao[i][j] = max(-10, min(10, populacao[i][j]))

# PSO
def pso(clientes, dist, cap):
    pop, vel = inicializar_populacao(N, len(clientes))

    pbest = pop[:]
    fit_pbest = [avaliar(p, clientes, dist, cap) for p in pop]

    idx = np.argmin(fit_pbest)
    gbest = pop[idx][:]
    fit_gbest = fit_pbest[idx]

    for gen in range(MAX_GERACOES):

        for i in range(len(pop)):
            f = avaliar(pop[i], clientes, dist, cap)

            if f < fit_pbest[i]:
                fit_pbest[i] = f
                pbest[i] = pop[i][:]

                if f < fit_gbest:
                    fit_gbest = f
                    gbest = pop[i][:]

        for i in range(len(pop)):
            for j in range(len(pop[i])):

                r1, r2 = random.random(), random.random()

                vel[i][j] = (
                    w * vel[i][j]
                    + c1 * r1 * (pbest[i][j] - pop[i][j])
                    + c2 * r2 * (gbest[j] - pop[i][j])
                )

                vel[i][j] = max(-vmax, min(vmax, vel[i][j]))
                pop[i][j] = max(0.0, min(1.0, pop[i][j] + vel[i][j]))

        print(f"Geração {gen+1} | melhor = {fit_gbest}")

    return gbest, fit_gbest

def imprimir_solucao(nome, gbest, fitness_value, clientes, dist, cap):
    rota = decode_random_keys(gbest)
    rotas = decoder(rota, clientes, dist, cap)

    print(f"Nome da instância: {nome}")
    print(f"Número de veículos: {len(rotas)}")
    print(f"Distância total: {fitness_value:.4f}")
    print("Rotas:")

    for i, r in enumerate(rotas, 1):
        print(f"Rota {i}: " + " ".join(map(str, r)))

# MAIN
instancia = "C101.txt"
clientes, cap, nveh = ler_instancia(instancia)
dist = gerar_matriz_distancia(clientes)

gbest, best = pso(clientes, dist, cap)

imprimir_solucao("c1_2_1", gbest, best, clientes, dist, cap)
