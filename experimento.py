import os
import matplotlib.pyplot as plt
from AG import main_vrptw, decoder, decoder_max_fill

ARQUIVO      = "Instancias_teste/rc208.txt"
N_GERACOES   = 100
SEEDS        = [42, 7, 123, 999]
PASTA_BASE   = "experimento_fatorial"
AUTORES      = "Henrique Franco, Andre"

CONFIGURACOES = [
    {"nome": "OX_maxfill_mut005",  "cruzamento": "OX",   "taxa_mutacao": 0.05, "decoder_fn": decoder_max_fill, "usar_or_opt": False},
    {"nome": "OX_maxfill_mut010",  "cruzamento": "OX",   "taxa_mutacao": 0.10, "decoder_fn": decoder_max_fill, "usar_or_opt": False},
    {"nome": "CX_maxfill_mut005",  "cruzamento": "CX",   "taxa_mutacao": 0.05, "decoder_fn": decoder_max_fill, "usar_or_opt": False},
    {"nome": "CX_maxfill_mut010",  "cruzamento": "CX",   "taxa_mutacao": 0.10, "decoder_fn": decoder_max_fill, "usar_or_opt": False},
    {"nome": "BCRC_maxfill_mut005","cruzamento": "BCRC", "taxa_mutacao": 0.05, "decoder_fn": decoder_max_fill, "usar_or_opt": False},
    {"nome": "BCRC_maxfill_mut010","cruzamento": "BCRC", "taxa_mutacao": 0.10, "decoder_fn": decoder_max_fill, "usar_or_opt": False},
    # decoder padrão + or_opt: 1/4 das gerações e população menor para compensar custo O(N³)
    {"nome": "OX_oropt_mut005",    "cruzamento": "OX",   "taxa_mutacao": 0.05, "decoder_fn": decoder,         "usar_or_opt": True,  "n_geracoes_override": N_GERACOES // 4, "tam_pop_override": 30},
    {"nome": "OX_oropt_mut010",    "cruzamento": "OX",   "taxa_mutacao": 0.10, "decoder_fn": decoder,         "usar_or_opt": True,  "n_geracoes_override": N_GERACOES // 4, "tam_pop_override": 30},
    {"nome": "CX_oropt_mut005",    "cruzamento": "CX",   "taxa_mutacao": 0.05, "decoder_fn": decoder,         "usar_or_opt": True,  "n_geracoes_override": N_GERACOES // 4, "tam_pop_override": 30},
    {"nome": "CX_oropt_mut010",    "cruzamento": "CX",   "taxa_mutacao": 0.10, "decoder_fn": decoder,         "usar_or_opt": True,  "n_geracoes_override": N_GERACOES // 4, "tam_pop_override": 30},
]


def media(valores):
    return sum(valores) / len(valores)


def rodar_experimento():
    os.makedirs(PASTA_BASE, exist_ok=True)

    resumo_linhas = ["CONFIG".ljust(30) + "  VEIC_MED  DIST_MED\n" + "-" * 55]
    resultados_para_grafico = {}

    for cfg in CONFIGURACOES:
        nome     = cfg["nome"]
        pasta    = os.path.join(PASTA_BASE, nome)
        os.makedirs(pasta, exist_ok=True)

        n_ger   = cfg.get("n_geracoes_override", N_GERACOES)
        tam_pop = cfg.get("tam_pop_override", 100)

        dists  = []
        veics  = []
        linhas = [
            f"Configuração: {nome}",
            f"  cruzamento={cfg['cruzamento']} | taxa_mutacao={cfg['taxa_mutacao']}",
            f"  decoder={'max_fill' if cfg['decoder_fn'] is decoder_max_fill else 'padrão'}",
            f"  or_opt={cfg['usar_or_opt']} | n_geracoes={n_ger}",
            "",
        ]

        print(f"\n{'='*55}")
        print(f"Config: {nome}")

        for seed in SEEDS:
            print(f"  seed={seed} ...", end=" ", flush=True)

            grafico = os.path.join(pasta, f"convergencia_seed{seed}.png")
            resultado_txt = os.path.join(pasta, f"resultado_seed{seed}.txt")

            dist_f, veic_f, _ = main_vrptw(
                arquivo          = ARQUIVO,
                n_geracoes       = n_ger,
                tam_pop          = tam_pop,
                taxa_mutacao     = cfg["taxa_mutacao"],
                cruzamento       = cfg["cruzamento"],
                decoder_fn       = cfg["decoder_fn"],
                usar_or_opt      = cfg["usar_or_opt"],
                seed             = seed,
                caminho_grafico  = grafico,
                autores          = AUTORES,
                caminho_resultado= resultado_txt,
                verbose          = False,
            )
            dists.append(dist_f)
            veics.append(veic_f)
            print(f"veic={veic_f}  dist={dist_f:.2f}")
            linhas.append(f"  seed={seed:<6} → veículos={veic_f}  distância={dist_f:.4f}")

        med_dist = media(dists)
        med_veic = media(veics)
        linhas += [
            "",
            f"  Média veículos : {med_veic:.2f}",
            f"  Média distância: {med_dist:.4f}",
        ]

        with open(os.path.join(pasta, "resumo.txt"), "w") as f:
            f.write("\n".join(linhas))

        resumo_linhas.append(f"{nome:<30}  {med_veic:>8.2f}  {med_dist:>10.4f}")
        resultados_para_grafico[nome] = {"dist": med_dist, "veic": med_veic}

    # Salva resumo geral
    resumo_path = os.path.join(PASTA_BASE, "resumo_geral.txt")
    with open(resumo_path, "w") as f:
        f.write("\n".join(resumo_linhas))
    print(f"\n{'='*55}")
    print("\n".join(resumo_linhas))

    # Gráfico comparativo de médias
    nomes   = list(resultados_para_grafico.keys())
    m_dists = [resultados_para_grafico[n]["dist"] for n in nomes]
    m_veics = [resultados_para_grafico[n]["veic"] for n in nomes]
    xs      = range(len(nomes))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.bar(xs, m_dists, color='steelblue')
    ax1.set_xticks(xs)
    ax1.set_xticklabels(nomes, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel("Distância média")
    ax1.set_title("Comparativo de configurações — Distância média")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.bar(xs, m_veics, color='tomato')
    ax2.set_xticks(xs)
    ax2.set_xticklabels(nomes, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel("Veículos médios")
    ax2.set_title("Comparativo de configurações — Veículos médios")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_BASE, "comparativo_configs.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nResultados em: {PASTA_BASE}/")


if __name__ == '__main__':
    rodar_experimento()
