- genomes: folder que contém todos os arquivos com genomas e dos dados tratados de cada genoma. É gigante e gerenciada pelos scripts de tratamento e coleta de dados.

- tools: folder que contém ferramentas 3rd-party utilizadas pelo Phemale.

- phemale: folder que contém arquivos do Phemale

- phemale/tests.ipynb: notebook para testes pontuais de código

- phemale/main.py: script de uso do Phemale

- phemale/data_module: folder que contém bibliotecas de coleta e tratamento de dados

- phemale/data_module/collect_data.py: script da biblioteca que realiza coleta de genomas do NCBI e roda o EggNOG para gerar arquivos dos grupos ortólogos encontrados em cada genoma

- phemale/data_module/transform_data: script da biblioteca que parseia os arquivos gerados pelo EggNOG, transforma os e-value em escala legível para treino e gera arquivos de vetores dos grupos ortólogos prontos para treino

- phemale/data_module/mount_data: script da biblioteca que monta os dataet de treino/validação

- phemale/data_module/data_io: script da biblioteca que lê arquivos de treino e escreve o log dos treinos

- phemale/training_module: contém biblotecas de treino para cada modelo: sklearn, tensorflow e lgbm

- phemale/results: onde se encontram os resultados dos treinos (logs, modelos treinados, gráficos)
