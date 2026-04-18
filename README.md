# Lab 08 — Alinhamento de LLM com DPO

Pipeline de alinhamento de modelo de linguagem usando Direct Preference Optimization (DPO) para comportamento HHH (Helpful, Honest, Harmless).

## Estrutura

```
hhh_dataset.jsonl   — 34 pares de preferência (prompt / chosen / rejected)
train_dpo.py        — pipeline de treinamento DPO
requirements.txt    — dependências do projeto
```

## Modelo

O modelo base utilizado é o `Qwen/Qwen2-0.5B`. O modelo ator recebe um adaptador LoRA (r=16) para atualização de pesos, enquanto o modelo de referência é carregado separadamente e mantido congelado durante todo o treinamento para o cálculo da divergência KL.

## Dataset

O arquivo `hhh_dataset.jsonl` contém 34 exemplos modelados como interações com um assistente de IA hospitalar. Os cenários cobrem ética médica, sigilo de prontuário, consentimento informado, conflito de interesse farmacêutico, integridade em pesquisa clínica e comunicação adequada com pacientes. Cada linha segue o formato obrigatório com as chaves `prompt`, `chosen` e `rejected`.

## Executar

```bash
pip install -r requirements.txt
python train_dpo.py
```

## Validação

Após o treinamento, a validação compara as log-probabilidades da resposta `chosen` e da resposta `rejected` dado um prompt malicioso. O modelo alinhado deve atribuir log-prob maior à resposta segura, comprovando que a distribuição foi ajustada pelo DPO.

## O parâmetro Beta (β = 0.1)

Na formulação matemática do DPO, o objetivo de treinamento é maximizar `log σ(β · (log πθ(yw|x)/πref(yw|x) − log πθ(yl|x)/πref(yl|x)))`, onde `yw` é a resposta preferida, `yl` a rejeitada, `πθ` o modelo ator e `πref` o modelo de referência congelado. O termo `β` escala a divergência KL implícita entre os dois modelos, funcionando como um imposto sobre o afastamento do comportamento original: valores altos de `β` penalizam fortemente qualquer desvio do modelo de referência, preservando fluência mas travando o aprendizado de preferências; valores baixos permitem que o modelo se afaste do base para absorver as preferências humanas, mas sem colapsar a distribuição e perder coerência linguística. O valor `β = 0.1` representa um equilíbrio que permite alinhamento efetivo sem degradar a qualidade das gerações.

---

Partes geradas/complementadas com IA, revisadas por Lara Vale. O recurso a ferramentas de IA se limitou à exploração inicial do tema, à organização de ideias e à criação de esboços de código, com revisão crítica aplicada a cada etapa.
