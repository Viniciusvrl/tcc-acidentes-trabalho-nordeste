# Acidentes de trabalho registrados no Nordeste (CAT/INSS, 2020 a 2025)

Código do Trabalho de Conclusão de Curso "Acidentes de trabalho registrados no Nordeste (CAT/INSS, 2020 a 2025):
análise descritiva e classificação do tipo de acidente com aprendizado de máquina" (Sistemas de Informação, UNIT).

- **Autor:** Vinícius (preencha o nome completo)
- **Orientador:** Sergio Roberto Santos Santana

O projeto integra os arquivos abertos de Comunicação de Acidente de Trabalho (CAT) do INSS, monta uma base
da região Nordeste e compara quatro algoritmos (Regressão Logística, Árvore de Decisão, Random Forest e
XGBoost) na classificação do tipo de acidente (típico, trajeto ou doença do trabalho) e na identificação
dos acidentes com óbito, além de uma análise descritiva por grupo.

## Estrutura do repositório

| Arquivo | Função |
|---|---|
| `gerar_base.py` | Executa em sequência os quatro passos abaixo e gera a base |
| `limpeza_cat.py` | Lê, integra e limpa os arquivos da CAT (Brasil) |
| `montar_base_ne.py` | Seleciona o Nordeste, marca os meses e remove registros com dado faltante |
| `exportar_csv.py` | Exporta `base_nordeste.csv` e `amostra_1000.csv` |
| `analise_dezembro.py` | Evidências sobre dezembro e meses de cobertura incompleta (opcional) |
| `modelo_random_forest.py` | Random Forest; também contém funções compartilhadas pelos demais scripts |
| `modelo_regressao_logistica.py`, `modelo_arvore_decisao.py`, `modelo_xgboost.py` | Um script por algoritmo |
| `modelos_comparativos.py` | Motor compartilhado pelos scripts de Regressão Logística, Árvore e XGBoost |
| `comparar_modelos.py` | Tabela e gráficos comparativos dos quatro algoritmos |
| `analise_descritiva.py` | Proporções por grupo, qui-quadrado e V de Cramér |
| `analises_complementares.py` | Análises de sensibilidade e tabelas pedidas na revisão: ausência por layout, antes e depois da exclusão, meses regulares, blocos de variáveis, intervalos de confiança e checagem do óbito |

Mantenha todos os scripts na mesma pasta: eles importam uns aos outros.

## Requisitos

- Python 3 (resultados do trabalho gerados com Python 3.14.3)
- Bibliotecas: pandas, numpy, scipy, scikit-learn, matplotlib, pyarrow e xgboost

```
pip install -r requirements.txt
```

Versões usadas nos resultados do trabalho: pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1,
matplotlib 3.11.2, xgboost 3.4.1 e pyarrow 26.0.0. Com outras versões, os resultados podem variar na
terceira casa decimal.

## Dados de entrada

Os dados são públicos e não estão neste repositório. Baixe os arquivos CSV de Comunicação de Acidente de
Trabalho de 2020 a 2025 no Portal Brasileiro de Dados Abertos (dados.gov.br), para onde o INSS migrou a publicação:
https://dados.gov.br/dados/conjuntos-dados/inss-comunicacao-de-acidente-de-trabalho-cat1
https://dados.gov.br/dados/conjuntos-dados/comunicacoes-de-acidente-de-trabalho-cat-plano-de-dados-abertos-jun-2023-a-jun-2025

Coloque os arquivos em uma pasta chamada `entrada`, dentro da pasta de trabalho. Os arquivos não têm
estrutura única: foram identificados 5 layouts (A, A2, B, C, D), e cada linha tem o layout reconhecido
automaticamente (coluna `layout_origem`). Nos arquivos do layout D (jun. a out./2023) a data de emissão
da CAT não pôde ser identificada; ela fica ausente (11,0% da base) e não é usada nos modelos.

## Como reproduzir

Pasta de trabalho esperada:

```
<pasta de trabalho>/
├── *.py
├── entrada/    <- CSV da CAT baixados do INSS
└── saida/      <- criada pelos scripts
```

Execute os comandos na pasta dos scripts. Os caminhos padrão são relativos a essa pasta: os CSV da CAT em
`entrada/` e os resultados em `saida/`. Para usar outros locais, indique `--pasta <caminho>` e `--saida
<caminho>` no `gerar_base.py`, ou `--entrada` e `--saida` nos demais scripts. No Windows, troque `python`
por `py` se o comando `python` não for reconhecido.

```
python gerar_base.py                    # 1. limpeza, base do Nordeste e CSV
python modelo_random_forest.py          # 2. Random Forest
python modelo_regressao_logistica.py    # 3. Regressão Logística
python modelo_arvore_decisao.py         # 4. Árvore de Decisão
python modelo_xgboost.py                # 5. XGBoost
python comparar_modelos.py              # 6. Comparação dos quatro algoritmos
python analise_descritiva.py            # 7. Análise descritiva
python analises_complementares.py       # 8. Análises complementares (depois dos passos 2 a 6)
```

Cada script de modelo aceita `--rapido` para um teste curto e `--sem-tuning` para pular a busca de
hiperparâmetros. Os resultados do `--rapido` **não servem para o trabalho** e ficam gravados na mesma pasta
dos resultados completos: depois de um teste rápido, rode de novo os passos 2 a 5 sem a opção. As sementes
aleatórias são fixas (42).

O `analises_complementares.py` lê os hiperparâmetros e as matrizes de confusão gravados pelos scripts de
modelo, por isso deve rodar depois deles. Aceita `--parte descritiva` (só tabelas e figuras, sem treino)
ou `--parte modelos` (sensibilidade com o Random Forest: meses regulares, divisão alternativa, blocos de
variáveis, registros com dado ausente, permutação por blocos, intervalos de confiança e checagem do óbito).

Saídas, dentro de `saida/`:

- `base_nordeste.*`: base final (parquet e CSV) e `amostra_1000.csv`;
- `modelos/<algoritmo>/`: métricas, matrizes de confusão, importância das variáveis e coeficientes;
- `modelos/comparativo_algoritmos.csv` e gráficos comparativos;
- `descritiva/`: tabelas e gráficos da análise descritiva;
- `complementar/`: tabelas, figuras e logs das análises complementares;
- `dicionario_rotulos.csv` e `removidos_por_ausencia.csv`: rótulos completos da fonte e registros excluídos.

## Filtros aplicados (e quantos registros cada um afeta)

1. Leitura e integração dos layouts: 8 linhas com estrutura inconsistente removidas.
2. Remoção de linhas duplicadas na base nacional: 522.466.
3. Localização: a UF do acidente **não é usada** (rótulos trocados na fonte: entre 94,5% e 99,2% dos
   registros de cada rótulo apontam para uma única UF de empregador diferente, e a coincidência entre as
   duas UF não passa de 0,05%). A correspondência entre rótulo e UF é a mesma nos cinco layouts, mas não
   recupera todas as UFs, por isso a coluna foi descartada. O recorte do Nordeste usa a **UF do empregador** (dois primeiros dígitos
   do código IBGE do município): MA, PI, CE, RN, PB, PE, AL, SE e BA.
4. Colunas descartadas: UF do acidente, CNPJ, tipo do empregador, datas de despacho do benefício e de
   afastamento, espécie do benefício, origem do cadastro e filiação do segurado.
5. Duplicados nas colunas finais: 106 removidos.
6. Dados faltantes: removidos os registros com ausência em qualquer variável de análise (agente causador,
   CBO, CID, CNAE, sexo, idade, natureza da lesão, parte do corpo, emitente, tipo de acidente, óbito,
   município e UF do empregador): 24.645 de 255.699 (9,6%). Lista em `removidos_por_ausencia.csv`.
7. Nenhum mês é excluído; os de cobertura incompleta ficam marcados (`mes_estavel`).

Base final: **231.054 linhas x 31 colunas**, 73 meses de ocorrência (71 entre jan./2020 e nov./2025 e
2 de 2019, com 4 registros).

## Dicionário das colunas da base final (`base_nordeste.csv`, separador `;`, UTF-8)

### Originais da CAT (padronizadas)

| Coluna | Descrição |
|---|---|
| `agente_causador` | Agente causador do acidente (texto de até 20 caracteres, como na fonte). "Nic" = não classificado em outra parte |
| `cbo` | Código da ocupação (CBO) do trabalhador |
| `cid` | Código do diagnóstico (CID-10) |
| `cnae` | Código da atividade econômica do empregador (CNAE) |
| `emitente_cat` | Quem emitiu a CAT: Empregador, Segurado/Dependente, Sindicato, Médico, Autoridade Pública |
| `indica_obito` | 1 se a CAT indica óbito; 0 caso contrário |
| `natureza_lesao` | Natureza da lesão (texto de até 20 caracteres) |
| `parte_corpo` | Parte do corpo atingida (texto de até 20 caracteres) |
| `sexo` | Feminino ou Masculino |
| `tipo_acidente` | Típico, Trajeto ou Doença (alvo principal) |
| `data_acidente` | Data do acidente (AAAA-MM-DD) |
| `data_nascimento` | Data de nascimento do trabalhador |
| `data_emissao_cat` | Data de emissão da CAT; ausente nos arquivos do layout D |
| `munic_empregador_cod` | Código IBGE do município do empregador (6 dígitos) |
| `uf_empregador` | UF do empregador (deriva do código IBGE); é a localização usada em todo o trabalho |

### Derivadas

| Coluna | Descrição |
|---|---|
| `ano`, `mes` | Ano e mês do acidente |
| `dia_semana` | Dia da semana do acidente: 0 = segunda-feira … 6 = domingo |
| `idade` | Idade do trabalhador na data do acidente, em anos (com decimais) |
| `faixa_etaria` | 14-24, 25-34, 35-44, 45-54, 55-64, 65+ |
| `dias_ate_emissao` | Dias entre o acidente e a emissão da CAT (ausente no layout D); **não usada nos modelos** |
| `cnae_divisao` | Divisão da CNAE (2 primeiros dígitos de `cnae`) |
| `cbo_grande_grupo` | Grande grupo da CBO (1º dígito de `cbo`): 0 forças armadas; 1 dirigentes e gerentes; 2 profissionais das ciências e das artes; 3 técnicos de nível médio; 4 serviços administrativos; 5 serviços e comércio; 6 agropecuária; 7 e 8 produção de bens e serviços industriais; 9 reparação e manutenção |
| `cid_capitulo` | Letra inicial do CID-10 (S e T lesões; M osteomuscular; F transtornos mentais; etc.) |
| `ym` | Mês do acidente, AAAA-MM |

### Controle de cobertura temporal

| Coluna | Descrição |
|---|---|
| `mes_estavel` | 1 se o mês tem cobertura regular nos arquivos de origem (47 meses); 0 se incompleta (26 meses) |
| `janela` | 1 a 4 = janelas contínuas de cobertura regular (jan./2020–ago./2021; jun.–set./2022; fev./2023–jul./2024; jun.–out./2025); 0 = meses de cobertura incompleta |
| `mes_dezembro` | 1 se o acidente ocorreu em dezembro (menos CATs: as emitidas em janeiro ficam fora dos arquivos) |
| `ultimo_mes_dados` | 1 no último mês de cobertura regular (out./2025) |
| `layout_origem` | Layout do arquivo de origem: A, A2, B, C ou D |
| `arquivo` | Nome do arquivo CSV de origem |

## Variáveis usadas nos modelos

- **Completo (12):** agente causador, natureza da lesão, parte do corpo, sexo, emitente da CAT, divisão da
  CNAE, grande grupo da CBO, capítulo do CID, UF do empregador, idade, mês, dia da semana.
- **Perfil (7):** sexo, divisão da CNAE, grande grupo da CBO, UF do empregador, idade, mês, dia da semana.
- **Alvos:** `tipo_acidente` (principal) e `indica_obito` (complementar, 0,59% dos registros).
- **Divisão temporal:** treino = acidentes até dez./2023 (142.657); teste = jan./2024 em diante (88.397),
  reportado com todos os meses e só com `mes_estavel = 1`. O corte `2024-01` é o padrão dos scripts de
  modelo e de `--corte`. O modelo de exemplo de `montar_base_ne.py` e a sensibilidade de
  `analise_dezembro.py` usam `2023-02`; eles servem só para a análise de dezembro e não entram nos
  resultados dos algoritmos.
- **Hiperparâmetros:** busca aleatória de 8 combinações com validação cruzada temporal (3 divisões), numa
  amostra de 40.000 registros do treino; classes com pesos inversamente proporcionais à frequência.

## Limitações da base (resumo)

- A CAT cobre só trabalhadores com vínculo formal, e há subnotificação.
- A localização é a do empregador, não a do acidente.
- Não há denominador (trabalhadores expostos): os resultados descrevem registros, não risco.
- Os arquivos trazem praticamente só CATs emitidas no ano do acidente; dezembro e o último mês ficam
  incompletos, e a cobertura mensal é irregular.
- A remoção por dado faltante foi desigual (20,1% em 2021 e 19,8% em 2022, 1,8% e 1,3% em 2024 e 2025).
- Rótulos de agente causador, natureza da lesão e parte do corpo vêm truncados em 20 caracteres, e a fonte
  não traz os códigos desses campos. Em 14 dos 270 rótulos de agente causador o corte junta descrições
  diferentes (`dicionario_rotulos.csv` traz o rótulo completo mais longo visto na fonte).
- O município do empregador pode ser o da sede, e não o do estabelecimento onde ocorreu o acidente.

## Licença

Código sob licença MIT (arquivo `LICENSE`). Os dados pertencem ao INSS e seguem os termos do Portal
Brasileiro de Dados Abertos.
