# CLAUDE.md

Código dos experimentos do TCC "Detecção de Nódulos Pulmonares em Radiografias
de Tórax Utilizando Redes Neurais Convolucionais" (NODE21, Faster R-CNN).

**O contexto do trabalho não está aqui, está no repositório do texto:**
`~/.workspace/Tcc` (<https://github.com/SchafNunes/Tcc>). Antes de mudar
qualquer coisa, leia lá:

- `PENDENCIAS.md`, seção 0: estado atual e próximos passos.
- `DECISOES.md`: cada decisão (D1 em diante) com o motivo. Não reverta uma
  decisão sem ler o verbete.
- `EXPERIMENTOS.md`: cada execução e verificação (E1 em diante), com números.
- `DESENVOLVIMENTO.md`: as decisões na ordem do capítulo de desenvolvimento.

Vocabulário: [CONTEXT.md](CONTEXT.md). Termos do texto em português, código em
inglês.

## Regras

- Toda decisão nova: registrar em `Tcc/DECISOES.md` e `Tcc/DESENVOLVIMENTO.md`.
  Toda execução ou verificação: nova entrada em `Tcc/EXPERIMENTOS.md`.
- A métrica de seleção é `mean_sens` (D32), não o `rank` do desafio.
- Os YAML da mesma etapa diferem só na variável comparada. Não mude batch, taxa
  ou rotina de uma condição sem mudar todas e sem retreinar as anteriores.
- `splits/splits.csv` foi gerado uma vez (D16, E4). Não regenerar.
- Um `RUN_NAME` novo para cada execução: com um nome existente o treino retoma o
  `last.pt` daquele diretório.

## Uso

```
uv sync && uv run pytest          # local, CPU
```

Os dados ficam em `data/` (fora do git). Treinos no Colab pelo
`notebooks/colab_runner.ipynb`. Fluxo e modos do `scripts/train.py` no
[README](README.md).
