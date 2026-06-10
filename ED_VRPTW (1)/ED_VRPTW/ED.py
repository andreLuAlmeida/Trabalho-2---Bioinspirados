import random
import math
import time
import matplotlib.pyplot as plt


# =========================
# PARÂMETROS DO ED
# =========================

# Tamanho da população.
NP = 50

# Número máximo de gerações.
MAX_GERACOES = 300

# Fator de mutação diferencial.
F = 0.6

# Taxa de crossover binomial.
CR = 0.85

# Critério de parada: encerra se ficar muitas gerações sem melhorar.
SEM_MELHORA_LIMITE = 60

# Semente fixa para permitir repetibilidade dos resultados.
SEMENTE = 42

# Para acelerar o decoder, testa apenas as rotas mais promissoras.
MAX_ROTAS_TESTADAS = 8


# =========================
# LEITURA DA INSTÂNCIA
# =========================

def ler_instancia(nome_arquivo):
    # Lista que armazenará depósito e clientes.
    clientes = []

    # Capacidade dos veículos.
    capacidade = 0

    # Número máximo de veículos informado no arquivo.
    numero_veiculos = 0

    # Controle para saber quando começou a tabela de clientes.
    lendo_clientes = False

    # Lê todas as linhas do arquivo da instância.
    with open(nome_arquivo, "r", encoding="utf-8") as arquivo:
        linhas = arquivo.readlines()

    # Percorre linha por linha.
    for linha in linhas:
        linha = linha.strip()

        # Ignora linhas vazias.
        if linha == "":
            continue

        partes = linha.split()

        # Captura a linha do número de veículos e capacidade.
        if len(partes) == 2 and partes[0].isdigit() and partes[1].isdigit() and capacidade == 0:
            numero_veiculos = int(partes[0])
            capacidade = int(partes[1])
            continue

        # Detecta o início da seção CUSTOMER.
        if linha.startswith("CUSTOMER"):
            lendo_clientes = True
            continue

        # Ignora linhas de cabeçalho.
        if "CUST" in linha or "XCOORD" in linha or "NUMBER" in linha or "CAPACITY" in linha:
            continue

        # Lê os dados dos clientes.
        if lendo_clientes and len(partes) >= 7:
            clientes.append([
                int(partes[0]),      # número do cliente
                float(partes[1]),    # coordenada x
                float(partes[2]),    # coordenada y
                int(partes[3]),      # demanda
                float(partes[4]),    # início da janela de tempo
                float(partes[5]),    # fim da janela de tempo
                float(partes[6])     # tempo de serviço
            ])

    return clientes, capacidade, numero_veiculos


# =========================
# MATRIZ DE DISTÂNCIAS
# =========================

def gerar_matriz_distancia(clientes):
    # Quantidade de pontos, incluindo o depósito.
    n = len(clientes)

    # Matriz de distâncias euclidianas.
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
# REPRESENTAÇÃO RANDOM KEYS
# =========================

def decodificar_random_keys(individuo):
    # Cada indivíduo do ED é um vetor real.
    # A ordem crescente dos valores define a permutação dos clientes.
    indices = sorted(range(len(individuo)), key=lambda i: individuo[i])

    # Soma 1 porque o índice 0 da lista de clientes é o depósito.
    return [i + 1 for i in indices]


# =========================
# AVALIAÇÃO DE UMA ROTA
# =========================

def avaliar_rota(rota, clientes, dist, capacidade):
    # Carga acumulada do veículo.
    carga = 0

    # Tempo acumulado da rota.
    tempo = 0.0

    # Distância percorrida na rota.
    distancia = 0.0

    # Violação total das restrições da rota.
    violacao = 0.0

    # Toda rota começa no depósito, índice 0.
    anterior = 0

    for cliente in rota:
        # Soma a demanda do cliente atual.
        carga += clientes[cliente][3]

        # Se exceder a capacidade, acumula violação.
        if carga > capacidade:
            violacao += carga - capacidade

        # Calcula chegada ao cliente atual.
        chegada = tempo + dist[anterior][cliente]

        # Se chegar cedo, espera até o início da janela.
        inicio = max(chegada, clientes[cliente][4])

        # Se iniciar depois do prazo, acumula violação.
        if inicio > clientes[cliente][5]:
            violacao += inicio - clientes[cliente][5]

        # Soma distância do deslocamento.
        distancia += dist[anterior][cliente]

        # Atualiza tempo após o serviço.
        tempo = inicio + clientes[cliente][6]

        # Atualiza cliente anterior.
        anterior = cliente

    # Calcula retorno ao depósito.
    retorno = tempo + dist[anterior][0]

    # Verifica se o retorno viola a janela do depósito.
    if retorno > clientes[0][5]:
        violacao += retorno - clientes[0][5]

    # Soma distância de retorno ao depósito.
    distancia += dist[anterior][0]

    # A rota é viável se não houver nenhuma violação.
    return violacao == 0.0, distancia, violacao


# =========================
# DECODER: PERMUTAÇÃO -> ROTAS
# =========================

def construir_rotas(permutacao, clientes, dist, capacidade):
    # Lista final de rotas.
    rotas = []

    # Insere os clientes um por um.
    for cliente in permutacao:
        melhor_rota = -1
        melhor_posicao = -1
        melhor_aumento = float("inf")

        # Lista de rotas candidatas para receber o cliente.
        candidatas = []

        # Para acelerar, prioriza rotas cujo último cliente está perto do novo cliente.
        for r in range(len(rotas)):
            ultimo = rotas[r][-1]
            prioridade = dist[ultimo][cliente]
            candidatas.append((prioridade, r))

        # Ordena as rotas mais promissoras.
        candidatas.sort()

        # Testa apenas algumas rotas, para reduzir custo computacional.
        candidatas = candidatas[:MAX_ROTAS_TESTADAS]

        # Tenta inserir o cliente em cada rota candidata.
        for prioridade, r in candidatas:
            rota = rotas[r]

            # Avalia a rota antes da inserção.
            viavel_antiga, dist_antiga, viol_antiga = avaliar_rota(rota, clientes, dist, capacidade)

            # Não tenta inserir em rota já inviável.
            if not viavel_antiga:
                continue

            # Testa todas as posições possíveis dentro da rota.
            for pos in range(len(rota) + 1):
                nova_rota = rota[:pos] + [cliente] + rota[pos:]

                viavel_nova, dist_nova, viol_nova = avaliar_rota(nova_rota, clientes, dist, capacidade)

                # Só aceita inserções viáveis.
                if viavel_nova:
                    aumento = dist_nova - dist_antiga

                    # Guarda a inserção de menor aumento de distância.
                    if aumento < melhor_aumento:
                        melhor_aumento = aumento
                        melhor_rota = r
                        melhor_posicao = pos

        # Se não encontrou inserção viável, abre uma nova rota.
        if melhor_rota == -1:
            rotas.append([cliente])
        else:
            # Insere o cliente na melhor rota e melhor posição encontrada.
            rota = rotas[melhor_rota]
            rotas[melhor_rota] = rota[:melhor_posicao] + [cliente] + rota[melhor_posicao:]

    return rotas


# =========================
# AVALIAÇÃO DA SOLUÇÃO COMPLETA
# =========================

def avaliar_solucao(individuo, clientes, dist, capacidade, max_veiculos):
    # Converte vetor real em permutação.
    permutacao = decodificar_random_keys(individuo)

    # Converte permutação em rotas viáveis, quando possível.
    rotas = construir_rotas(permutacao, clientes, dist, capacidade)

    distancia_total = 0.0
    violacao_total = 0.0
    visitados = []

    # Avalia cada rota.
    for rota in rotas:
        viavel, distancia, violacao = avaliar_rota(rota, clientes, dist, capacidade)

        distancia_total += distancia
        violacao_total += violacao

        for cliente in rota:
            visitados.append(cliente)

    # Conjunto esperado de clientes.
    esperado = set(range(1, len(clientes)))

    # Conjunto realmente visitado.
    obtido = set(visitados)

    # Penaliza clientes faltantes.
    faltantes = len(esperado - obtido)

    # Penaliza clientes repetidos.
    repetidos = len(visitados) - len(obtido)

    # Penaliza excesso de veículos.
    excesso_veiculos = max(0, len(rotas) - max_veiculos)

    # Violações artificiais para garantir comparação pelo método de Deb.
    violacao_total += faltantes * 100000.0
    violacao_total += repetidos * 100000.0
    violacao_total += excesso_veiculos * 10000.0

    # Retorna todas as informações importantes da solução.
    return {
        "viavel": violacao_total == 0.0,
        "violacao": violacao_total,
        "veiculos": len(rotas),
        "distancia": distancia_total,
        "rotas": rotas
    }


# =========================
# MÉTODO DE DEB
# =========================

def deb_melhor(a, b):
    # Regra 1: solução viável vence solução inviável.
    if a["viavel"] and not b["viavel"]:
        return True

    # Regra 2: solução inviável perde para solução viável.
    if not a["viavel"] and b["viavel"]:
        return False

    # Regra 3: entre duas viáveis, compara objetivo.
    if a["viavel"] and b["viavel"]:
        # Primeiro objetivo: minimizar número de veículos.
        if a["veiculos"] < b["veiculos"]:
            return True

        if a["veiculos"] > b["veiculos"]:
            return False

        # Segundo objetivo: minimizar distância.
        return a["distancia"] <= b["distancia"]

    # Regra 4: entre duas inviáveis, vence a menor violação.
    return a["violacao"] <= b["violacao"]


# =========================
# POPULAÇÃO INICIAL
# =========================

def criar_populacao(tamanho, dimensao):
    populacao = []

    # Cada indivíduo é um vetor real de tamanho igual ao número de clientes.
    for i in range(tamanho):
        individuo = []

        for j in range(dimensao):
            individuo.append(random.random())

        populacao.append(individuo)

    return populacao


# =========================
# LIMITAÇÃO DO INTERVALO
# =========================

def limitar(valor):
    # Mantém cada componente do vetor no intervalo [0, 1].
    if valor < 0.0:
        return 0.0

    if valor > 1.0:
        return 1.0

    return valor


# =========================
# MUTAÇÃO DIFERENCIAL
# =========================

def mutacao_diferencial(populacao, indice_alvo):
    # Cria lista de índices, removendo o indivíduo alvo.
    indices = list(range(len(populacao)))
    indices.remove(indice_alvo)

    # Escolhe três indivíduos distintos.
    r1, r2, r3 = random.sample(indices, 3)

    x1 = populacao[r1]
    x2 = populacao[r2]
    x3 = populacao[r3]

    mutante = []

    # Fórmula do ED:
    # v = x1 + F * (x2 - x3)
    for j in range(len(x1)):
        valor = x1[j] + F * (x2[j] - x3[j])
        mutante.append(limitar(valor))

    return mutante


# =========================
# CROSSOVER BINOMIAL
# =========================

def crossover_binomial(alvo, mutante):
    dimensao = len(alvo)
    teste = []

    # Garante que pelo menos uma posição venha do mutante.
    j_obrigatorio = random.randrange(dimensao)

    # Para cada posição, escolhe se vem do alvo ou do mutante.
    for j in range(dimensao):
        if random.random() < CR or j == j_obrigatorio:
            teste.append(mutante[j])
        else:
            teste.append(alvo[j])

    return teste


# =========================
# EVOLUÇÃO DIFERENCIAL PARA VRPTW
# =========================

def evolucao_diferencial_vrptw(clientes, capacidade, max_veiculos):
    # Número de variáveis do indivíduo.
    dimensao = len(clientes) - 1

    # Calcula matriz de distâncias uma única vez.
    dist = gerar_matriz_distancia(clientes)

    # Cria população inicial.
    populacao = criar_populacao(NP, dimensao)

    # Armazena avaliação de cada indivíduo.
    avaliacoes = []

    # Melhor solução global encontrada.
    melhor_avaliacao = None

    # Históricos para gráficos.
    historico_distancia = []
    historico_veiculos = []
    historico_violacao = []

    # Avaliação inicial da população.
    for i in range(NP):
        avaliacao = avaliar_solucao(populacao[i], clientes, dist, capacidade, max_veiculos)
        avaliacoes.append(avaliacao)

        if melhor_avaliacao is None or deb_melhor(avaliacao, melhor_avaliacao):
            melhor_avaliacao = avaliacao

    sem_melhora = 0

    # Laço principal do ED.
    for geracao in range(MAX_GERACOES):
        melhorou = False

        # Para cada indivíduo alvo, cria mutante, teste e aplica seleção.
        for i in range(NP):
            mutante = mutacao_diferencial(populacao, i)
            teste = crossover_binomial(populacao[i], mutante)

            # Avalia o vetor teste.
            avaliacao_teste = avaliar_solucao(teste, clientes, dist, capacidade, max_veiculos)

            # Seleção direta pelo método de Deb.
            if deb_melhor(avaliacao_teste, avaliacoes[i]):
                populacao[i] = teste
                avaliacoes[i] = avaliacao_teste

                # Atualiza melhor global.
                if deb_melhor(avaliacao_teste, melhor_avaliacao):
                    melhor_avaliacao = avaliacao_teste
                    melhorou = True

        # Salva histórico da melhor solução da geração.
        historico_distancia.append(melhor_avaliacao["distancia"])
        historico_veiculos.append(melhor_avaliacao["veiculos"])
        historico_violacao.append(melhor_avaliacao["violacao"])

        # Mostra progresso a cada 25 gerações.
        if geracao % 25 == 0:
            print(
                "Geração:", geracao,
                "| Viável:", melhor_avaliacao["viavel"],
                "| Veículos:", melhor_avaliacao["veiculos"],
                "| Distância:", round(melhor_avaliacao["distancia"], 2),
                "| Violação:", round(melhor_avaliacao["violacao"], 2)
            )

        # Controle de parada por estagnação.
        if melhorou:
            sem_melhora = 0
        else:
            sem_melhora += 1

        if sem_melhora >= SEM_MELHORA_LIMITE:
            print("Parada antecipada na geração", geracao)
            break

    return melhor_avaliacao, historico_distancia, historico_veiculos, historico_violacao


# =========================
# VERIFICAÇÃO FINAL
# =========================

def verificar(rotas, clientes, capacidade):
    dist = gerar_matriz_distancia(clientes)
    visitados = []

    print()
    print("Verificação:")

    # Verifica rota por rota.
    for i in range(len(rotas)):
        viavel, distancia, violacao = avaliar_rota(rotas[i], clientes, dist, capacidade)
        carga = sum(clientes[c][3] for c in rotas[i])

        print(
            "Rota", i + 1,
            "| viável:", viavel,
            "| carga:", carga,
            "| distância:", round(distancia, 2),
            "| violação:", round(violacao, 2)
        )

        for cliente in rotas[i]:
            visitados.append(cliente)

    # Confere se todos os clientes foram atendidos uma única vez.
    esperado = set(range(1, len(clientes)))
    obtido = set(visitados)

    print("Clientes atendidos:", len(obtido), "de", len(esperado))
    print("Clientes repetidos:", len(visitados) - len(obtido))
    print("Clientes faltando:", sorted(list(esperado - obtido)))


# =========================
# IMPRESSÃO NO TERMINAL
# =========================

def imprimir(nome_instancia, avaliacao, max_veiculos):
    print()
    print("Nome da instância:", nome_instancia)
    print("Solução viável:", avaliacao["viavel"])
    print("Violação total:", round(avaliacao["violacao"], 2))
    print("Número de veículos usado:", avaliacao["veiculos"])
    print("Número de veículos máximo:", max_veiculos)
    print("Distância total:", round(avaliacao["distancia"], 2))
    print("Rotas:")

    for i in range(len(avaliacao["rotas"])):
        print("Rota " + str(i + 1) + ":", " ".join(str(c) for c in avaliacao["rotas"][i]))


# =========================
# SALVAR RESULTADOS EM TXT
# =========================

def salvar_resultado(nome_instancia, avaliacao, tempo_execucao):
    nome_saida = "resultado_ED_" + nome_instancia + ".txt"

    with open(nome_saida, "w", encoding="utf-8") as arquivo:
        arquivo.write("Numero de Veiculos: " + str(avaliacao["veiculos"]) + "\n")
        arquivo.write("Distancia Total: " + str(round(avaliacao["distancia"], 2)) + "\n")
        arquivo.write("Tempo de Execucao (segundos): " + str(round(tempo_execucao, 4)) + "\n")
        arquivo.write("Viavel: " + str(avaliacao["viavel"]) + "\n")
        arquivo.write("Violacao Total: " + str(round(avaliacao["violacao"], 2)) + "\n")
        arquivo.write("Semente: " + str(SEMENTE) + "\n")
        arquivo.write("Parametros ED: NP=" + str(NP) + ", F=" + str(F) + ", CR=" + str(CR) + ", MAX_GERACOES=" + str(MAX_GERACOES) + "\n")
        arquivo.write("Rotas:\n")

        for i in range(len(avaliacao["rotas"])):
            rota_texto = " ".join(str(c) for c in avaliacao["rotas"][i])
            arquivo.write("Rota " + str(i + 1) + ": " + rota_texto + "\n")

    print("Arquivo salvo:", nome_saida)


# =========================
# GRÁFICOS DE CONVERGÊNCIA
# =========================

def plotar_convergencia(nome_instancia, historico_distancia, historico_veiculos, historico_violacao):
    if len(historico_distancia) == 0:
        return

    # Gráfico da distância.
    plt.figure()
    plt.plot(range(len(historico_distancia)), historico_distancia)
    plt.xlabel("Geração")
    plt.ylabel("Melhor distância")
    plt.title("Convergência ED - Distância - " + nome_instancia)
    plt.grid(True)
    plt.savefig("convergencia_ED_distancia_" + nome_instancia + ".png", dpi=150, bbox_inches="tight")
    plt.close()

    # Gráfico do número de veículos.
    plt.figure()
    plt.plot(range(len(historico_veiculos)), historico_veiculos)
    plt.xlabel("Geração")
    plt.ylabel("Número de veículos")
    plt.title("Convergência ED - Veículos - " + nome_instancia)
    plt.grid(True)
    plt.savefig("convergencia_ED_veiculos_" + nome_instancia + ".png", dpi=150, bbox_inches="tight")
    plt.close()

    # Gráfico da violação total.
    plt.figure()
    plt.plot(range(len(historico_violacao)), historico_violacao)
    plt.xlabel("Geração")
    plt.ylabel("Violação total")
    plt.title("Convergência ED - Violação - " + nome_instancia)
    plt.grid(True)
    plt.savefig("convergencia_ED_violacao_" + nome_instancia + ".png", dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# PROGRAMA PRINCIPAL
# =========================

def main():
    # Define a semente do gerador aleatório.
    random.seed(SEMENTE)

    # Lista das instâncias que serão executadas.
    instancias = [
        "c1_2_1.txt",
        "c101.txt",
        "r209.txt",
        "rc2_4_9.txt",
        "rc208.txt"
    ]

    # Executa o ED para cada instância.
    for nome_arquivo in instancias:
        nome_instancia = nome_arquivo.replace(".txt", "")

        print()
        print("=" * 60)
        print("Executando instância:", nome_instancia)
        print("=" * 60)

        # Marca o tempo inicial.
        inicio = time.time()

        # Lê a instância.
        clientes, capacidade, max_veiculos = ler_instancia(nome_arquivo)

        # Executa o ED adaptado ao VRPTW.
        avaliacao, historico_distancia, historico_veiculos, historico_violacao = evolucao_diferencial_vrptw(
            clientes,
            capacidade,
            max_veiculos
        )

        # Marca o tempo final.
        fim = time.time()

        # Calcula tempo total.
        tempo_execucao = fim - inicio

        # Exibe, verifica, salva e plota os resultados.
        imprimir(nome_instancia, avaliacao, max_veiculos)
        verificar(avaliacao["rotas"], clientes, capacidade)
        salvar_resultado(nome_instancia, avaliacao, tempo_execucao)
        plotar_convergencia(nome_instancia, historico_distancia, historico_veiculos, historico_violacao)

        print("Tempo de execução:", round(tempo_execucao, 4), "segundos")


main()