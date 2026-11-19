# Testes

Cada módulo de `mlscratch/` tem um arquivo `test_<módulo>.py` que compara a implementação
do zero com a de referência: scikit-learn, ou `scipy.special` no caso de `numerics`.

## Como rodar

Com o ambiente ativado:

```bash
pytest -v                          # todos os testes
pytest tests/test_distances.py -v  # só um módulo
pytest -k paridade -v              # só testes cujo nome contém "paridade"
```

Sem ativar o ambiente: `.venv/bin/pytest -v`.

## Como ler o resultado

- `PASSED`: a implementação está correta naquele caso.
- `FAILED`: o pytest mostra qual teste falhou, com quais parâmetros (aparecem entre colchetes
  no nome, por exemplo `[euclidean_vectorized-5-3-2]`) e os valores que não bateram.

## Tipos de teste

| Tipo | O que verifica | Exemplo |
| --- | --- | --- |
| Paridade | Mesmo resultado do scikit-learn (`np.allclose`) | Distâncias vs. `pairwise_distances` |
| Valor conhecido | Um caso calculável à mão | Triângulo 3-4-5 |
| Propriedades | Algo que sempre tem que valer | Matriz de distâncias simétrica, folds disjuntos |
| Casos de borda | Situações que quebram implementações ingênuas | Arredondamento gerando `nan`, feature constante, empates na AUC |

Quando o resultado não pode ser idêntico ao do scikit-learn (por exemplo, splits com
embaralhamento diferente), testamos propriedades em vez de valores.

## Cobertura atual

| Arquivo | Testes | Status |
| --- | --- | --- |
| `test_distances.py` | 22 | Passando |
| `test_numerics.py` | 23 | Passando |
| `test_preprocessing.py` | 32 | Passando |
| `test_model_selection.py` | 69 | Passando (1 falha esperada, `xfail`: arredondamento no `train_test_split` estratificado com classes pequenas) |
| `test_metrics.py` | — | Aguardando implementação |
