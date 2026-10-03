# Detecção de nódulos no NODE21

Experimentos do TCC "Detecção de Nódulos Pulmonares em Radiografias de Tórax
Utilizando Redes Neurais Convolucionais". Os termos seguem o cap. 3 do texto. O
nome entre crases é o usado no código.

## Dados

**Radiografia** (`image`):
Uma radiografia de tórax PA na versão padronizada distribuída pelo NODE21, 1024 x 1024.
_Evite_: exame, slice, CXR (no código)

**Nódulo**:
Uma lesão anotada por radiologista. É a unidade de avaliação.
_Evite_: lesão, objeto, instância

**Caixa de referência** (`gt_boxes`):
A caixa delimitadora de um nódulo no metadata.csv, já no espaço da radiografia padronizada.
_Evite_: ground truth, anotação, bbox de referência

**Radiografia positiva / negativa** (`positive`):
Radiografia com pelo menos um nódulo / sem nenhum. Negativas entram no treino como imagens de fundo.
_Evite_: imagem normal, imagem com label 1

**Grupo de duplicatas** (`group`):
Conjunto de radiografias tão parecidas pela assinatura perceptual que podem vir do mesmo exame ou paciente. Fica inteiro numa partição.
_Evite_: cluster, paciente

## Partições e protocolo

**Partição** (`split`):
Treino (70%), validação (15%) ou teste (15%), estratificadas pela presença de nódulo e geradas uma única vez.
_Evite_: subset, conjunto (sozinho)

**Conjunto de seleção** (`split != test`):
A união de treino e validação, sobre a qual o k-fold é feito.
_Evite_: pool, treino+val

**Fold** (`fold`):
Um dos 3 terços do conjunto de seleção na etapa 3.
_Evite_: dobra

**Etapa**:
Uma das três fases do protocolo: 1, comparação de arquiteturas; 2, ablação de realce; 3, k-fold e modelo final.
_Evite_: fase, estágio (estágio é termo de arquitetura)

**Braço** (`arch`):
Uma das arquiteturas comparadas na etapa 1: Faster R-CNN (referência) ou RetinaNet (comparação).
_Evite_: modelo (sozinho), variante

**Condição de realce** (`enhancement`):
Sem realce, equalização global, CLAHE ou RMSHE, aplicada sobre a radiografia padronizada.
_Evite_: pré-processamento (que é o pipeline do NODE21)

**Execução** (`run`):
Um treino do protocolo, definido por uma configuração e um modo (seleção, fold ou final).
_Evite_: experimento, job

**Modelo de fold**:
Modelo treinado em dois folds e avaliado no terceiro, por número fixo de épocas. Mede a variabilidade.
_Evite_: modelo de validação cruzada

**Modelo final**:
Modelo treinado no conjunto de seleção inteiro, por número fixo de épocas. É o único avaliado no teste.
_Evite_: melhor modelo, modelo vencedor

## Predição e avaliação

**Predição** (`Prediction`):
Uma caixa com escore de confiança produzida pelo detector para uma radiografia.
_Evite_: detecção (como substantivo contável), candidato

**Verdadeiro positivo**:
Predição com IoU > 0,2 com uma caixa de referência ainda não atribuída, percorrendo as predições por escore decrescente. Duplicatas sobre o mesmo nódulo são falsos positivos.
_Evite_: acerto

**Escore da radiografia** (`image_scores`):
O maior escore entre as predições da radiografia, ou zero se não houver predição. Alimenta a AUC em nível de imagem.
_Evite_: probabilidade da imagem

**Sensibilidade a r FP/imagem** (`sens@r`):
Fração de nódulos encontrados no ponto da curva FROC com r falsos positivos por radiografia, lida em 1/8, 1/4 e 1/2.
_Evite_: recall (sozinho)

**Métrica de seleção** (`rank`):
0,75 x AUC + 0,25 x sensibilidade a 0,25 FP/imagem, na validação. Decide arquitetura, realce, interrupção antecipada e melhor época.
_Evite_: score, métrica composta (sem qualificar), "melhor desempenho"
