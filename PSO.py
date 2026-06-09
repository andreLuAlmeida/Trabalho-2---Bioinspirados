import random
import matplotlib.pyplot as plt

# PARÂMETROS
N = 50
MAX_GERACOES = 500
c1 = 2.05
c2 = 2.05
w = 0.4
vmax = 2.0

# FUNÇÃO OBJETIVO
def funcao(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

# INICIALIZAÇÃO
def inicializar_populacao(tamanho):
    populacao = [[random.uniform(-10,10), random.uniform(-10,10)]for _ in range(tamanho)]
    velocidade = [[random.uniform(-1,1), random.uniform(-1,1)] for _ in range(tamanho)]
    return populacao, velocidade

# INICIALIZA PBEST E GBEST
def inicializar_melhores(populacao):
    pbest = [ind[:] for ind in populacao]
    fitness_pbest = [funcao(ind) for ind in populacao]
    indice_melhor = min(range(len(populacao)), key=lambda i: fitness_pbest[i])
    gbest = pbest[indice_melhor][:]
    fitness_gbest = fitness_pbest[indice_melhor]
    return pbest, fitness_pbest, gbest, fitness_gbest

# ATUALIZA PBEST E GBEST
def atualizar_melhores(populacao, pbest, fitness_pbest, gbest, fitness_gbest):

    melhorou = False

    for i in range(len(populacao)):
        fitness_atual = funcao(populacao[i])
        if fitness_atual < fitness_pbest[i]:
            fitness_pbest[i] = fitness_atual
            pbest[i] = populacao[i][:]
            if fitness_atual < fitness_gbest:
                fitness_gbest = fitness_atual
                gbest = populacao[i][:]
                melhorou = True
    return (pbest, fitness_pbest, gbest, fitness_gbest, melhorou)

# ATUALIZA VELOCIDADE
def atualizar_velocidade(populacao, velocidade, pbest, gbest):
    for i in range(len(populacao)):
        for j in range(len(populacao[i])):
            r1 = random.random()
            r2 = random.random()
            velocidade[i][j] = (w * velocidade[i][j] + c1 * r1 * (pbest[i][j] - populacao[i][j]) + c2 * r2 * (gbest[j] - populacao[i][j]))
            velocidade[i][j] = max( -vmax, min(vmax, velocidade[i][j]))

# ATUALIZA POSIÇÕES
def atualizar_posicoes(populacao, velocidade):
    for i in range(len(populacao)):
        for j in range(len(populacao[i])):
            populacao[i][j] += velocidade[i][j]
            populacao[i][j] = max(-10, min(10, populacao[i][j]))

# PSO
def pso():
    historico = []
    populacao, velocidade = inicializar_populacao(N)
    (pbest, fitness_pbest, gbest, fitness_gbest) = inicializar_melhores(populacao)
    sem_melhora = 0
    LIMITE_SEM_MELHORA = 50

    for geracao in range(MAX_GERACOES):
        (pbest, fitness_pbest, gbest, fitness_gbest, melhorou) = atualizar_melhores(populacao, pbest, fitness_pbest,gbest, fitness_gbest)
        if melhorou:
            sem_melhora = 0
        else:
            sem_melhora += 1
        if sem_melhora >= LIMITE_SEM_MELHORA:
            print( f"Parada antecipada na geração {geracao}")
            break
        atualizar_velocidade(populacao,velocidade, pbest, gbest)
        atualizar_posicoes( populacao, velocidade)
        historico.append(fitness_gbest)
    return gbest, fitness_gbest, historico

# MAIN
def main():
    melhor_solucao, melhor_fitness, historico = pso()
    print("\nMelhor solução:")
    print(melhor_solucao)
    print("\nFitness:")
    print(melhor_fitness)
    plt.plot(historico)
    plt.xlabel("Geração")
    plt.ylabel("Melhor fitness")
    plt.title("Convergência do PSO")
    plt.show()

if __name__ == "__main__":
    main()