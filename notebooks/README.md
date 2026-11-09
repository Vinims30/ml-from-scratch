# Notebooks

Exercícios de estudo e benchmarks, organizados por fase. O código que vale a pena reutilizar
sai daqui e vai para o pacote `mlscratch/`; os notebooks importam de lá.

Para abrir, com o ambiente ativado:

```bash
jupyter lab
```

## Fase 1: NumPy para ML

| Notebook | Conteúdo | Status |
| --- | --- | --- |
| [01_arrays_views_indexacao](fase1/01_arrays_views_indexacao.ipynb) | `shape`, `dtype`, views vs. cópias, indexação booleana e fancy indexing, one-hot | Concluído |
| [02_broadcasting](fase1/02_broadcasting.ipynb) | Regras de broadcasting, z-score por coluna, `keepdims`, diferenças par-a-par | Concluído |
| [03_agregacoes_distancias](fase1/03_agregacoes_distancias.ipynb) | Agregações com `axis`, `argsort`, `bincount`, `einsum`, distância euclidiana em três versões | Concluído |
| [04_linalg](fase1/04_linalg.ipynb) | `solve` vs. `inv` vs. `lstsq`, número de condição, colinearidade | Pendente |
| [05_estabilidade_numerica](fase1/05_estabilidade_numerica.ipynb) | `default_rng`, log-sum-exp, softmax e sigmoid estáveis | Pendente |
| [06_benchmark_distancias](fase1/06_benchmark_distancias.ipynb) | Paridade, `timeit` e gráfico log-log das três distâncias vs. scikit-learn | Faltam as conclusões |

## Convenções

- Aleatoriedade sempre com `rng = np.random.default_rng(42)`, nunca `np.random.seed`.
- Notebooks de benchmark terminam com uma seção **Conclusões**, escrita à mão depois de rodar.
- Figuras que vão para o README são salvas em `benchmarks/figures/`.
