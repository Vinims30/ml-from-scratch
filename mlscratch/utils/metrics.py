"""Métricas (Fase 1, Semana 3, Sessões 2 e 3).

Regressão:
- mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

Classificação:
- confusion_matrix  (via np.bincount)
- accuracy_score, precision_score, recall_score, f1_score  (binário e macro; zero_division)
- log_loss          (clip em [eps, 1 - eps])
- roc_auc_score     (pela curva/trapézio e por ranks/Mann-Whitney, com empates)

Paridade: sklearn.metrics
"""
