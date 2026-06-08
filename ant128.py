import random
import matplotlib.pyplot as plt

alpha = 1
beta = 5
rho = 0.5
Q = 100
tau = 10**-6
numeroFormigas = 50

#Gera feromoneos iniciais para cada cidade e retorna a matriz de feromônios
def geraFeromoniosIniciais(num_cidades):
    feromonios = [[0 for _ in range(num_cidades)] for _ in range(num_cidades)]
    for i in range(num_cidades):
        for j in range(num_cidades):
            if i != j:
                feromonios[i][j] = tau
            else:
                feromonios[i][j] = 0
    return feromonios


#Lê a matriz de distância do arquivo e retorna uma matriz simetrica
def leMatrizDistancia(arquivo):
    matriz = []
    with open(arquivo, 'r') as f:
        for linha in f:
            linha = linha.strip()
            if linha == "" or linha.startswith("#"):
                continue
            numeros = linha.split()
            linha_matriz = [int(x) for x in numeros]
            matriz.append(linha_matriz)
    return matriz


#inicializa população de formigas
def geraPopulacaoInicial(num_formigas, num_cidades):
    populacao = [[0 for _ in range(num_cidades)] for _ in range(num_formigas)]
    for i in range(num_formigas):
        populacao[i][0] = random.randint(0, num_cidades - 1)
    return populacao


#probabilidade de transição de uma formiga escolher a próxima cidade
def probabilidadeTransicao(formiga, cidade_atual, cidades, feromonios):
    num_cidades = len(cidades)
    probabilidades = [0] * num_cidades
    soma = 0

    for j in range(num_cidades):
        if j not in formiga and j != cidade_atual:
            tau_ij = feromonios[cidade_atual][j] ** alpha
            eta_ij = (1 / cidades[cidade_atual][j]) ** beta
            probabilidades[j] = tau_ij * eta_ij
            soma += probabilidades[j]

    for j in range(num_cidades):
        if soma > 0:
            probabilidades[j] = probabilidades[j] / soma
        else:
            probabilidades[j] = 0

    return probabilidades


#escolhe a proxima cidade com base nas probabilidades de transição
def escolherProximaCidade(probabilidades):
    r = random.random()
    acumulado = 0

    for cidade, p in enumerate(probabilidades):
        acumulado += p
        if r <= acumulado:
            return cidade

    return probabilidades.index(max(probabilidades))


#constrói o caminho de uma formiga com base nas probabilidades de transição
def construirCaminho(formiga, cidades, feromonios):
    cidade_atual = formiga[0]

    for passo in range(1, len(cidades)):
        probabilidades = probabilidadeTransicao(
            formiga,
            cidade_atual,
            cidades,
            feromonios
        )

        proxima = escolherProximaCidade(probabilidades)

        formiga[passo] = proxima
        cidade_atual = proxima

    return formiga


#custo caminho percorrido por uma formiga
def custoCaminho(caminho, cidades):
    custo = 0

    for i in range(len(caminho)-1):
        custo += cidades[caminho[i]][caminho[i+1]]

    custo += cidades[caminho[-1]][caminho[0]]

    return custo


#atualiza feromonios com base no caminho percorrido por cada formiga
def atualizarFeromonios(feromonios, formigas, custos):
    n = len(feromonios)

    # evaporação
    for i in range(n):
        for j in range(n):
            feromonios[i][j] *= (1 - rho)

    # depósito
    for k in range(len(formigas)):
        caminho = formigas[k]
        custo = custos[k]
        deposito = Q / custo

        for i in range(len(caminho)-1):
            a = caminho[i]
            b = caminho[i+1]
            feromonios[a][b] += deposito
            feromonios[b][a] += deposito

        # volta para a primeira
        a = caminho[-1]
        b = caminho[0]
        feromonios[a][b] += deposito
        feromonios[b][a] += deposito


def inicializar():
    cidades = leMatrizDistancia("sgb128_dist.txt")
    numeroCidades = len(cidades)
    print("Número de cidades:", numeroCidades)
    feromonios = geraFeromoniosIniciais(numeroCidades)
    formigas = geraPopulacaoInicial(numeroFormigas, numeroCidades)
    vetorCusto = [0 for _ in range(numeroFormigas)]

    return cidades, feromonios, formigas, vetorCusto


def ACO(iteracoes):
    cidades, feromonios, formigas, vetorCusto = inicializar()
    melhor_custo = float("inf")
    melhor_caminho = None
    historico = []

    for it in range(iteracoes):
        for i in range(numeroFormigas):
            formigas[i] = construirCaminho(
                formigas[i],
                cidades,
                feromonios
            )
            vetorCusto[i] = custoCaminho(
                formigas[i],
                cidades
            )
        menor = min(vetorCusto)

        if menor < melhor_custo:
            melhor_custo = menor
            melhor_caminho = formigas[vetorCusto.index(menor)]
        historico.append(melhor_custo)
        print("Iteração", it, "melhor custo:", melhor_custo)
        atualizarFeromonios(
            feromonios,
            formigas,
            vetorCusto
        )
        formigas = geraPopulacaoInicial(numeroFormigas, len(cidades))
    print("Melhor solução encontrada:", melhor_custo)
    print("Caminho:", melhor_caminho)

    return melhor_caminho, melhor_custo, historico


# experimento fatorial variando parâmetros do ACO
def experimento_fatorial():

    configuracoes = [
        (1,5,0.3),
        (1,5,0.5),
        (1,5,0.7),
        (1,3,0.5),
        (2,5,0.5)
    ]

    for a,b,r in configuracoes:
        global alpha, beta, rho
        alpha = a
        beta = b
        rho = r

        print("\n-----------------------------------")
        print("Executando configuração:")
        print("alpha =",a," beta =",b," rho =",r)

        caminho, custo, historico = ACO(100)
        plt.plot(historico, label=f"a={a} b={b} r={r}")

    plt.xlabel("Iterações")
    plt.ylabel("Melhor custo")
    plt.title("Gráfico de convergência do ACO")
    plt.legend()
    plt.show()


experimento_fatorial()