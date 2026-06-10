import random
import math
import time
import matplotlib.pyplot as plt


# =========================
# PARÂMETROS DO PSO
# =========================

N_PARTICULAS = 50
MAX_GERACOES = 300
W_INICIAL = 0.9
W_FINAL = 0.4
C1 = 1.6
C2 = 1.9
VMAX = 0.20
SEM_MELHORA_LIMITE = 60
SEMENTE = 42

PESO_VEICULOS = 1000000.0
PESO_INFEASIVEL = 1000000000.0


# =========================
# LEITURA DA INSTÂNCIA
# =========================

def ler_instancia(nome_arquivo):
    clientes = []
    capacidade = 0
    numero_veiculos = 0
    lendo_clientes = False

    with open(nome_arquivo, "r", encoding="utf-8") as arquivo:
        linhas = arquivo.readlines()

    for linha in linhas:
        linha = linha.strip()

        if linha == "":
            continue

        partes = linha.split()

        if len(partes) == 2 and partes[0].isdigit() and partes[1].isdigit() and capacidade == 0:
            numero_veiculos = int(partes[0])
            capacidade = int(partes[1])
            continue

        if linha.startswith("CUSTOMER"):
            lendo_clientes = True
            continue

        if "CUST" in linha or "XCOORD" in linha or "NUMBER" in linha or "CAPACITY" in linha:
            continue

        if lendo_clientes and len(partes) >= 7:
            clientes.append([
                int(partes[0]),
                float(partes[1]),
                float(partes[2]),
                int(partes[3]),
                float(partes[4]),
                float(partes[5]),
                float(partes[6])
            ])

    return clientes, capacidade, numero_veiculos


# =========================
# MATRIZ DE DISTÂNCIAS
# =========================

def gerar_matriz_distancia(clientes):
    n = len(clientes)
    dist = []

    for i in range(n):
        linha = []
        xi = clientes[i][1]
        yi = clientes[i][2]

        for j in range(n):
            dx = xi - clientes[j][1]
            dy = yi - clientes[j][2]
            linha.append(math.sqrt(dx * dx + dy * dy))

        dist.append(linha)

    return dist


# =========================
# DECODIFICAÇÃO RANDOM KEYS
# =========================

def decodificar_random_keys(particula):
    indices = sorted(range(len(particula)), key=lambda i: particula[i])
    return [i + 1 for i in indices]


# =========================
# AVALIAÇÃO DE ROTA
# =========================

def avaliar_rota(rota, clientes, dist, capacidade):
    carga = 0
    tempo = 0.0
    distancia = 0.0
    anterior = 0

    for cliente in rota:
        carga += clientes[cliente][3]

        if carga > capacidade:
            return False, distancia

        chegada = tempo + dist[anterior][cliente]
        inicio = max(chegada, clientes[cliente][4])

        if inicio > clientes[cliente][5]:
            return False, distancia

        distancia += dist[anterior][cliente]
        tempo = inicio + clientes[cliente][6]
        anterior = cliente

    retorno = tempo + dist[anterior][0]

    if retorno > clientes[0][5]:
        return False, distancia

    distancia += dist[anterior][0]

    return True, distancia


# =========================
# CONSTRUÇÃO DAS ROTAS
# =========================

def construir_rotas(permutacao, clientes, dist, capacidade):
    rotas = []

    for cliente in permutacao:
        melhor_rota = -1
        melhor_posicao = -1
        melhor_aumento = float("inf")

        for r in range(len(rotas)):
            rota = rotas[r]
            viavel_antiga, dist_antiga = avaliar_rota(rota, clientes, dist, capacidade)

            if not viavel_antiga:
                continue

            for pos in range(len(rota) + 1):
                candidata = rota[:pos] + [cliente] + rota[pos:]
                viavel_nova, dist_nova = avaliar_rota(candidata, clientes, dist, capacidade)

                if viavel_nova:
                    aumento = dist_nova - dist_antiga

                    if aumento < melhor_aumento:
                        melhor_aumento = aumento
                        melhor_rota = r
                        melhor_posicao = pos

        if melhor_rota == -1:
            rotas.append([cliente])
        else:
            rota = rotas[melhor_rota]
            rotas[melhor_rota] = rota[:melhor_posicao] + [cliente] + rota[melhor_posicao:]

    return rotas


# =========================
# 2-OPT FINAL
# =========================

def melhorar_2opt_simples(rotas, clientes, dist, capacidade):
    novas_rotas = []

    for rota in rotas:
        melhor = rota[:]
        viavel, melhor_dist = avaliar_rota(melhor, clientes, dist, capacidade)

        if len(rota) <= 3 or not viavel:
            novas_rotas.append(melhor)
            continue

        melhorou = True

        while melhorou:
            melhorou = False

            for i in range(len(melhor) - 1):
                for j in range(i + 2, len(melhor) + 1):
                    candidata = melhor[:i] + list(reversed(melhor[i:j])) + melhor[j:]
                    viavel_cand, dist_cand = avaliar_rota(candidata, clientes, dist, capacidade)

                    if viavel_cand and dist_cand + 0.000001 < melhor_dist:
                        melhor = candidata
                        melhor_dist = dist_cand
                        melhorou = True
                        break

                if melhorou:
                    break

        novas_rotas.append(melhor)

    return novas_rotas


# =========================
# DISTÂNCIA TOTAL
# =========================

def distancia_total(rotas, clientes, dist, capacidade):
    total = 0.0
    penalidade = 0.0

    for rota in rotas:
        viavel, distancia = avaliar_rota(rota, clientes, dist, capacidade)
        total += distancia

        if not viavel:
            penalidade += 1.0

    return total, penalidade


# =========================
# FUNÇÃO OBJETIVO
# =========================

def fitness(particula, clientes, dist, capacidade, max_veiculos):
    permutacao = decodificar_random_keys(particula)
    rotas = construir_rotas(permutacao, clientes, dist, capacidade)

    distancia, penalidade = distancia_total(rotas, clientes, dist, capacidade)
    excesso = max(0, len(rotas) - max_veiculos)

    valor = (
        len(rotas) * PESO_VEICULOS
        + distancia
        + excesso * PESO_INFEASIVEL
        + penalidade * PESO_INFEASIVEL
    )

    return valor, rotas, distancia


# =========================
# POPULAÇÃO
# =========================

def criar_populacao(n_particulas, n_clientes):
    populacao = []
    velocidades = []

    for i in range(n_particulas):
        particula = []
        velocidade = []

        for j in range(n_clientes):
            particula.append(random.random())
            velocidade.append(random.uniform(-VMAX, VMAX))

        populacao.append(particula)
        velocidades.append(velocidade)

    return populacao, velocidades


# =========================
# ATUALIZAÇÃO DO PSO
# =========================

def atualizar(populacao, velocidades, pbest, gbest, geracao):
    w = W_INICIAL - ((W_INICIAL - W_FINAL) * geracao / MAX_GERACOES)

    for i in range(len(populacao)):
        particula = populacao[i]
        velocidade = velocidades[i]
        melhor_pessoal = pbest[i]

        for j in range(len(particula)):
            r1 = random.random()
            r2 = random.random()

            velocidade[j] = (
                w * velocidade[j]
                + C1 * r1 * (melhor_pessoal[j] - particula[j])
                + C2 * r2 * (gbest[j] - particula[j])
            )

            if velocidade[j] > VMAX:
                velocidade[j] = VMAX
            elif velocidade[j] < -VMAX:
                velocidade[j] = -VMAX

            particula[j] += velocidade[j]

            if particula[j] < 0.0:
                particula[j] = 0.0
            elif particula[j] > 1.0:
                particula[j] = 1.0


# =========================
# MUTAÇÃO
# =========================

def mutacao(populacao, taxa):
    for i in range(len(populacao)):
        particula = populacao[i]

        if random.random() < taxa:
            a = random.randrange(len(particula))
            b = random.randrange(len(particula))
            particula[a], particula[b] = particula[b], particula[a]

        if random.random() < taxa:
            a = random.randrange(len(particula))
            particula[a] = random.random()


# =========================
# PSO VRPTW
# =========================

def pso_vrptw(clientes, capacidade, max_veiculos):
    n_clientes = len(clientes) - 1
    dist = gerar_matriz_distancia(clientes)

    populacao, velocidades = criar_populacao(N_PARTICULAS, n_clientes)

    pbest = [p[:] for p in populacao]
    pbest_fit = []

    gbest = None
    gbest_fit = float("inf")
    gbest_rotas = []
    gbest_dist = float("inf")

    historico = []

    for i in range(N_PARTICULAS):
        valor, rotas, distancia = fitness(populacao[i], clientes, dist, capacidade, max_veiculos)
        pbest_fit.append(valor)

        if valor < gbest_fit:
            gbest = populacao[i][:]
            gbest_fit = valor
            gbest_rotas = rotas
            gbest_dist = distancia

    sem_melhora = 0

    for geracao in range(MAX_GERACOES):
        melhorou = False

        for i in range(N_PARTICULAS):
            valor, rotas, distancia = fitness(populacao[i], clientes, dist, capacidade, max_veiculos)

            if valor < pbest_fit[i]:
                pbest[i] = populacao[i][:]
                pbest_fit[i] = valor

            if valor < gbest_fit:
                gbest = populacao[i][:]
                gbest_fit = valor
                gbest_rotas = rotas
                gbest_dist = distancia
                melhorou = True

        historico.append(gbest_dist)

        if geracao % 50 == 0:
            print("Geração:", geracao, "| Veículos:", len(gbest_rotas), "| Distância:", round(gbest_dist, 2))

        if melhorou:
            sem_melhora = 0
        else:
            sem_melhora += 1

        if sem_melhora >= SEM_MELHORA_LIMITE:
            print("Parada antecipada na geração", geracao)
            break

        atualizar(populacao, velocidades, pbest, gbest, geracao)

        if sem_melhora > 30:
            mutacao(populacao, 0.08)
        else:
            mutacao(populacao, 0.02)

    gbest_rotas = melhorar_2opt_simples(gbest_rotas, clientes, dist, capacidade)
    gbest_dist, penalidade = distancia_total(gbest_rotas, clientes, dist, capacidade)

    return gbest_rotas, gbest_dist, historico


# =========================
# VERIFICAÇÃO
# =========================

def verificar(rotas, clientes, capacidade):
    dist = gerar_matriz_distancia(clientes)
    visitados = []

    print()
    print("Verificação:")

    for i in range(len(rotas)):
        viavel, distancia = avaliar_rota(rotas[i], clientes, dist, capacidade)
        carga = sum(clientes[c][3] for c in rotas[i])

        print("Rota", i + 1, "| viável:", viavel, "| carga:", carga, "| distância:", round(distancia, 2))

        for c in rotas[i]:
            visitados.append(c)

    esperado = set(range(1, len(clientes)))
    obtido = set(visitados)

    print("Clientes atendidos:", len(obtido), "de", len(esperado))
    print("Clientes repetidos:", len(visitados) - len(obtido))
    print("Clientes faltando:", sorted(list(esperado - obtido)))


# =========================
# IMPRESSÃO
# =========================

def imprimir(nome, rotas, distancia, max_veiculos):
    print()
    print("Nome da instância:", nome)
    print("Número de veículos usado:", len(rotas))
    print("Número de veículos máximo:", max_veiculos)
    print("Distância total:", round(distancia, 2))
    print("Rotas:")

    for i in range(len(rotas)):
        print("Rota " + str(i + 1) + ":", " ".join(str(c) for c in rotas[i]))


# =========================
# SALVAR RESULTADO
# =========================

def salvar_resultado(nome_instancia, rotas, distancia, tempo_execucao):
    nome_saida = "resultado_" + nome_instancia + ".txt"

    with open(nome_saida, "w", encoding="utf-8") as arquivo:
        arquivo.write("Numero de Veiculos: " + str(len(rotas)) + "\n")
        arquivo.write("Distancia Total: " + str(round(distancia, 2)) + "\n")
        arquivo.write("Tempo de Execucao (segundos): " + str(round(tempo_execucao, 4)) + "\n")
        arquivo.write("Semente: " + str(SEMENTE) + "\n")
        arquivo.write("Rotas:\n")

        for i in range(len(rotas)):
            rota_texto = " ".join(str(c) for c in rotas[i])
            arquivo.write("Rota " + str(i + 1) + ": " + rota_texto + "\n")

    print("Arquivo salvo:", nome_saida)


# =========================
# GRÁFICO
# =========================

def plotar_convergencia(nome_instancia, historico):
    if len(historico) == 0:
        return

    plt.figure()
    plt.plot(range(len(historico)), historico)
    plt.xlabel("Geração")
    plt.ylabel("Melhor distância encontrada")
    plt.title("Convergência PSO - " + nome_instancia)
    plt.grid(True)

    nome_figura = "convergencia_" + nome_instancia + ".png"
    plt.savefig(nome_figura, dpi=150, bbox_inches="tight")
    plt.close()

    print("Gráfico salvo:", nome_figura)


# =========================
# MAIN
# =========================

def main():
    random.seed(SEMENTE)

    instancias = [
        "c1_2_1.txt",
        "c101.txt",
        "r209.txt",
        "rc2_4_9.txt",
        "rc208.txt"
    ]

    for nome_arquivo in instancias:
        nome_instancia = nome_arquivo.replace(".txt", "")

        print()
        print("=" * 60)
        print("Executando instância:", nome_instancia)
        print("=" * 60)

        inicio = time.time()

        clientes, capacidade, max_veiculos = ler_instancia(nome_arquivo)
        rotas, distancia, historico = pso_vrptw(clientes, capacidade, max_veiculos)

        fim = time.time()
        tempo_execucao = fim - inicio

        imprimir(nome_instancia, rotas, distancia, max_veiculos)
        verificar(rotas, clientes, capacidade)
        salvar_resultado(nome_instancia, rotas, distancia, tempo_execucao)
        plotar_convergencia(nome_instancia, historico)

        print("Tempo de execução:", round(tempo_execucao, 4), "segundos")


main()