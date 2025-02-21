**README**

# Pipeline e Modelagem para Progressão da Doença de Parkinson

Este repositório contém o desenvolvimento de um pipeline de pré-processamento e modelagem para auxiliar na análise e predição de progressão da Doença de Parkinson. O trabalho foi desenvolvido no contexto da disciplina **Mineração de Dados e Aplicações na Engenharia (2024.2)**, do curso de **Engenharia de Computação** da **Universidade Federal do Maranhão (UFMA)**, sob orientação do professor **Dr. Thales Levi Azevedo Valente**.

---

## Sumário
1. [Objetivo](#objetivo)  
2. [Estrutura do Projeto](#estrutura-do-projeto)  
3. [Como Usar](#como-usar)  
4. [Requisitos e Dependências](#requisitos-e-dependencias)  
5. [Contribuindo](#contribuindo)  
6. [Agradecimentos](#agradecimentos)  
7. [Autores](#autores)  
8. [Licença](#licenca)

---

## Objetivo
O objetivo principal deste projeto é desenvolver um fluxo de trabalho (pipeline) que:

- Integre dados clínicos, proteicos e de peptídeos de pacientes com Doença de Parkinson.
- Realize pré-processamento, limpeza e tratamento de valores ausentes e outliers.
- Crie recursos temporais (lag features, intervalos entre visitas).
- Aplique técnicas de análise exploratória, detecção de correlações e visualização de dados.  
- Forneça subsídios para a modelagem e previsão da progressão da doença.

---

## Estrutura do Projeto

```
.
├── pre_data_progressao_parkinson/  # Pacote principal
│   ├── __init__.py
│   ├── cli.py                     # Ponto de entrada para linha de comando
│   └── gera_grafico.py           # Funções auxiliares de plot
├── base_de_dados_tratada.csv      # Exemplo de base final após pré-processamento
├── pre_processamento_dados.py     # Lógica de pré-processamento e engenharia de atributos
├── setup.py                       # Arquivo de instalação do pacote
├── README.md                      # Este arquivo
└── ...
```

- **pre_processamento_dados.py**: Contém as etapas de carga, limpeza, integração e transformação dos dados, além de funções para geração de gráficos e análises de correlação.
- **base_de_dados_tratada.csv**: Arquivo CSV resultante do pipeline de pré-processamento, incluindo dados integrados e tratados.
- **setup.py**: Permite instalar o pacote localmente, gerando scripts de linha de comando.
- **pre_data_progressao_parkinson**: Diretório que abriga o pacote principal, com scripts auxiliares (por exemplo, geração de gráficos) e o script principal de execução em linha de comando.

---

## Como Usar

1. **Clonar o repositório**  
   ```bash
   git clone https://github.com/FaustinoNeto/pre_data_progressao_parkinson.git
   cd pre_data_progressao_parkinson
   ```

2. **Instalar o pacote**  
   Você pode instalar localmente através do `setup.py`:
   ```bash
   pip install .
   ```
   Isso criará o comando `run_parkinson` no ambiente Python instalado.

3. **Executar o pipeline**  
   Após a instalação, basta rodar:
   ```bash
   run_parkinson
   ```
   Esse comando executa o pipeline de pré-processamento, gerando o arquivo `base_de_dados_tratada.csv` pronto para as próximas etapas de análise ou modelagem.

---

## Requisitos e Dependências
- **Python 3.7+**
- **pandas**
- **numpy**
- **matplotlib**
- **seaborn**
- **scikit-learn**
- **xgboost**

Todas as dependências são instaladas automaticamente via `pip` a partir do arquivo [`setup.py`](setup.py).

---

## Contribuindo
Contribuições são bem-vindas! Para isso:
1. Faça um fork do projeto.
2. Crie uma nova branch para sua feature/correção (`git checkout -b feature-xyz`).
3. Envie um Pull Request descrevendo detalhadamente suas modificações.

---

## Reconhecimentos e Direitos Autorais
@autor: FAUSTINO DOS SANTOS GARCEZ NETO
@contato: faustinonetox@gmail.com
@data última versão: 21/02/2025
@versão: 1.0
@outros repositórios: https://github.com/FaustinoNeto
@Agradecimentos: Universidade Federal do Maranhão (UFMA), Professor Doutor Thales Levi Azevedo Valente, e colegas de curso.
Copyright/License
Este material é resultado de um trabalho acadêmico para a disciplina MINERAÇÃO DE DADOS E APLICAÇÕES NA ENGENHARIA, sob a orientação do professor Dr. THALES LEVI AZEVEDO VALENTE, semestre letivo 2024.2, curso Engenharia da Computação, na Universidade Federal do Maranhão (UFMA).
Todo o material sob esta licença é software livre: pode ser usado para fins acadêmicos e comerciais sem nenhum custo. Não há papelada, nem royalties, nem restrições de "copyleft" do tipo GNU. Ele é licenciado sob os termos da Licença MIT, conforme descrito abaixo, e, portanto, é compatível com a GPL e também se qualifica como software de código aberto. É de domínio público. Os detalhes legais estão abaixo. O espírito desta licença é que você é livre para usar este material para qualquer finalidade, sem nenhum custo. O único requisito é que, se você usá-los, nos dê crédito.
Licenciado sob a Licença MIT. Permissão é concedida, gratuitamente, a qualquer pessoa que obtenha uma cópia deste software e dos arquivos de documentação associados (o "Software"), para lidar no Software sem restrição, incluindo sem limitação os direitos de usar, copiar, modificar, mesclar, publicar, distribuir, sublicenciar e/ou vender cópias do Software, e permitir pessoas a quem o Software é fornecido a fazê-lo, sujeito às seguintes condições:
Este aviso de direitos autorais e este aviso de permissão devem ser incluídos em todas as cópias ou partes substanciais do Software.
O SOFTWARE É FORNECIDO "COMO ESTÁ", SEM GARANTIA DE QUALQUER TIPO, EXPRESSA OU IMPLÍCITA, INCLUINDO MAS NÃO SE LIMITANDO ÀS GARANTIAS DE COMERCIALIZAÇÃO, ADEQUAÇÃO A UM DETERMINADO FIM E NÃO INFRINGÊNCIA. EM NENHUM CASO OS AUTORES OU DETENTORES DE DIREITOS AUTORAIS SERÃO RESPONSÁVEIS POR QUALQUER RECLAMAÇÃO, DANOS OU OUTRA RESPONSABILIDADE, SEJA EM AÇÃO DE CONTRATO, TORT OU OUTRA FORMA, DECORRENTE DE, FORA DE OU EM CONEXÃO COM O SOFTWARE OU O USO OU OUTRAS NEGOCIAÇÕES NO SOFTWARE.
Para mais informações sobre a Licença MIT: https://opensource.org/licenses/MIT
