# Comparações

Um relatório por algoritmo, comparando a implementação do zero com a da biblioteca.
Começa na Fase 2, com os modelos lineares. As peças da Fase 1 (distâncias, métricas,
pré-processamento) são documentadas nos próprios notebooks.

## Como criar um relatório

Copie [`_template.md`](_template.md) para `<algoritmo>.md` (por exemplo `ridge.md`) e preencha:

1. **Ideia e matemática:** um parágrafo e a equação principal.
2. **Decisões da minha implementação:** solver, critério de parada, vetorização.
3. **Resultados:** métrica, tempo de fit e de predict para alguns tamanhos de n.
4. **Escalabilidade:** gráfico log-log de tempo × n com as duas implementações.
5. **O que a biblioteca faz diferente:** tabela *aspecto | minha versão | biblioteca | impacto observado*.
6. **Lições:** o que aprendi e não esperava.

## Relatórios

Nenhum ainda.
