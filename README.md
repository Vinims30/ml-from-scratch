# ml-from-scratch

Algoritmos de Machine Learning implementados do zero em NumPy, com API no estilo
scikit-learn (`fit`, `predict`, `predict_proba`) e comparados com as bibliotecas
de referência em **paridade**, **métricas**, **escalabilidade** e **diferenças de implementação**.

## Status

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

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
pip install -e ".[dev]"
```

## Testes

```bash
pytest
```

## Estrutura

```text
ml-from-scratch/
├── mlscratch/
│   └── utils/          # distances, preprocessing, model_selection, metrics, numerics
├── tests/              # paridade com sklearn via np.allclose
├── benchmarks/         # scripts que geram tabelas e gráficos de tempo
├── notebooks/          # exercícios e benchmarks por fase
└── docs/comparisons/   # um .md por algoritmo
```

## Comparações

| Algoritmo | Paridade | Métrica (minha / lib) | Speedup da lib | Detalhes |
| --- | --- | --- | --- | --- |
