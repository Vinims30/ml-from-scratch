# ml-from-scratch

Algoritmos de Machine Learning implementados do zero em NumPy, com API no estilo
scikit-learn (`fit`, `predict`, `predict_proba`), e comparados com as bibliotecas
de referência (scikit-learn e, mais tarde, PyTorch).

A ideia não é substituir as bibliotecas, e sim entender o que elas fazem por dentro.
Toda implementação é comparada em quatro pontos:

1. **Paridade:** com os mesmos hiperparâmetros, os resultados batem (`np.allclose`)?
2. **Métricas:** mesmo split e mesma seed para as duas versões.
3. **Escalabilidade:** tempo de execução com n crescendo, em gráfico log-log.
4. **Diferenças de implementação:** o que a biblioteca faz que a minha versão não faz.

## Progresso

| Fase | Tema | Status |
| --- | --- | --- |
| 1 | NumPy para ML + infraestrutura de avaliação | Em andamento |
| 2 | Modelos lineares + fluxo completo de projeto | Não iniciada |
| 3 | KNN, Naive Bayes, árvores | Não iniciada |
| 4 | Ensembles e SVM | Não iniciada |
| 5 | Não supervisionado | Não iniciada |
| 6 | Problemas do mundo real + projeto principal | Não iniciada |
| 7 | Fundamentos de DL: do zero → PyTorch | Não iniciada |
| 8 | CNNs, RNNs/LSTMs, Transformers | Não iniciada |

### Fase 1: pacote `mlscratch` v0.1

- [x] Distâncias euclidianas par-a-par (dois loops, um loop, vetorizada) + testes + benchmark
- [x] Funções numericamente estáveis (softmax, log-softmax, log-sum-exp, sigmoid) + testes
- [x] Pré-processamento (`StandardScaler`, `MinMaxScaler`, `OneHotEncoder`) + testes
- [x] Divisão de dados (`train_test_split`, `KFold`, `StratifiedKFold`) + testes
- [x] Métricas de regressão e classificação (incluindo log loss e ROC-AUC) + testes

## Instalação

Requer Python 3.11+.

```bash
git clone https://github.com/Vinims30/ml-from-scratch.git
cd ml-from-scratch
python -m venv .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
pip install -e ".[dev]"
```

O `pip install -e` instala o pacote em modo editável: alterações em `mlscratch/`
valem na hora, nos notebooks e nos testes, sem reinstalar.

## Uso

```python
import numpy as np
from mlscratch.utils.distances import euclidean_vectorized

A = np.array([[0.0, 0.0], [1.0, 1.0]])
B = np.array([[3.0, 4.0]])
euclidean_vectorized(A, B)
# array([[5.        ],
#        [3.60555128]])
```

## Testes

```bash
pytest -v
```

Os testes comparam cada implementação com a do scikit-learn. Veja [tests/README.md](tests/README.md).

## Resultados

### Distância euclidiana par-a-par (n = m = 2000, d = 50)

| Implementação | n = 100 | n = 500 | n = 2000 | Speedup vs. dois loops (n = 2000) |
| --- | --- | --- | --- | --- |
| Dois loops | 38,2 ms | 890,6 ms | 14,9 s | 1× |
| Um loop | 1,6 ms | 24,8 ms | 384,3 ms | 38,7× |
| Vetorizada | 0,08 ms | 1,2 ms | 50,0 ms | 297,8× |
| scikit-learn | 0,40 ms | 2,0 ms | 46,9 ms | 317,2× |

![Benchmark log-log das distâncias](benchmarks/figures/distancias_loglog.png)

**Conclusões:**

- **Paridade:** as três versões batem com o scikit-learn (`np.allclose`), com diferença máxima na casa de 10⁻¹⁵,
  causada pelo arredondamento de ponto flutuante em caminhos de cálculo diferentes.
- **Por que a vetorizada é tão mais rápida:** o loop sai do Python e passa a rodar em C compilado dentro do NumPy,
  sem o custo do interpretador a cada iteração; a multiplicação de matrizes usa BLAS, que aproveita paralelismo.
- **Minha versão vence com n pequeno:** o scikit-learn tem um custo fixo de ~0,3 ms por chamada para validar a entrada.
  Com n grande esse custo some e as duas versões empatam.
- **O que o scikit-learn faz diferente:** usa a mesma fórmula, mas zera a diagonal quando X e Y são o mesmo array
  e converte float32 para float64 em blocos, o que evita perda grande de precisão com pontos próximos.

Detalhes em [notebooks/fase1/06_benchmark_distancias.ipynb](notebooks/fase1/06_benchmark_distancias.ipynb).

## Estrutura

```text
ml-from-scratch/
├── mlscratch/          # o pacote: implementações do zero
│   └── utils/          # distances, numerics, preprocessing, model_selection, metrics
├── tests/              # paridade com o scikit-learn (pytest)
├── benchmarks/         # harness de tempo e figuras geradas
├── notebooks/          # exercícios e benchmarks, organizados por fase
└── docs/comparisons/   # um relatório por algoritmo (a partir da Fase 2)
```

Cada pasta tem seu próprio README explicando o conteúdo.
