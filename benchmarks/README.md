# Benchmarks

Medições de tempo das implementações do zero contra as bibliotecas de referência.

## Protocolo

Toda comparação registra:

1. **Paridade:** com hiperparâmetros equivalentes, os resultados batem (`np.allclose`)?
   Se não, explicar o motivo.
2. **Métricas:** mesmo split e mesma seed para as duas versões.
3. **Escalabilidade:** tempo × n em gráfico log-log. Em log-log, um custo O(nᵏ) vira uma reta
   de inclinação k, e a distância vertical entre as retas mostra a diferença de constante.
4. **Diferenças de implementação:** o que a biblioteca faz que a minha versão não faz.

Os tempos usam a **mediana** de várias execuções, que é menos sensível a ruído que a média.

## Conteúdo

| Caminho | O que é |
| --- | --- |
| `harness.py` | Função `benchmark(model, X_tr, y_tr, X_te, y_te, metric)` para modelos com `fit`/`predict` (a partir da Fase 2) |
| `figures/` | Gráficos gerados pelos notebooks e usados nos READMEs |

## Resultados

| Benchmark | Notebook | Figura |
| --- | --- | --- |
| Distância euclidiana par-a-par | [06_benchmark_distancias](../notebooks/fase1/06_benchmark_distancias.ipynb) | [distancias_loglog.png](figures/distancias_loglog.png) |

Os tempos dependem da máquina. Para reproduzir, rode o notebook correspondente.
