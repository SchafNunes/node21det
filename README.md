# node21det

Código dos experimentos do TCC: detecção de nódulos pulmonares no NODE21,
Faster R-CNN contra RetinaNet (ResNet-50 + FPN), com ablação de realce de
contraste. Adaptado da linha de base oficial
([node21_detection_baseline](https://github.com/node21challenge/node21_detection_baseline)).
A lógica da FROC tem como referência a solução MTEC do desafio
(Behrendt et al., 2023).

Vocabulário em [CONTEXT.md](CONTEXT.md).

## Ambiente local (CPU)

```
uv sync
uv run pytest
```

## Fluxo

```
# 1. duplicatas aproximadas (CPU, uma vez)
python scripts/find_duplicates.py --images DATA/images --metadata DATA/metadata.csv --out-dir splits/

# 2. partições (uma vez; o arquivo vai para o git)
python scripts/make_splits.py --metadata DATA/metadata.csv --groups splits/dup_groups.csv --out splits/splits.csv

# 3. treino da etapa 1
python scripts/train.py --config configs/frcnn_none.yaml --mode select \
    --images DATA/images --metadata DATA/metadata.csv --splits splits/splits.csv --out RUNS/frcnn_none
```

Interrompido, o mesmo comando retoma de `RUNS/<execução>/last.pt`.

Modos de `train.py`: `select` (treino/validação, etapas 1 e 2), `fold --fold k --epochs N`
e `final --epochs N` (etapa 3).
