# 🌐 Agente de Tradução de Documentos Multi-Provedor (Ollama, OpenAI, Anthropic) com LangGraph

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-v0.1+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Este repositório contém uma implementação profissional de um **Agente de Tradução de Documentos** que utiliza a metodologia de **Tradução Agentica (Agentic Translation)**. Construído com **LangGraph**, o agente oferece suporte nativo a múltiplos provedores de LLM: **Ollama** (para modelos locais como `gemma4:12b` via WSL ou Windows), **OpenAI** (via API, ex: `gpt-4o`) e **Anthropic** (via API, ex: `claude-3-5-sonnet-latest`).

O agente processa arquivos nos formatos `.md`, `.txt`, `.pdf` e `.docx` de forma inteligente, dividindo-os em blocos lógicos, gerando glossários dinâmicos e aplicando um ciclo de **Tradução -> Revisão (Reflexão) -> Refinamento** para garantir a máxima fluidez, fidelidade e consistência técnica.

---

## 🛠️ Arquitetura do Agente (LangGraph)

O fluxo de controle do agente é modelado como um grafo cíclico direcionado usando **LangGraph**, o que garante a consistência contextual bloco a bloco:

```text
               +-------------------------------------------+
               |                  INÍCIO                   |
               +-------------------------------------------+
                                     |
                                     v
               +-------------------------------------------+
               |      Inicializar CLI ou Web App           |
               +-------------------------------------------+
                                     |
                                     v
                    /---------------------------------\
                   /       Provedor & Conexão          \
                   \            Válidos?               /
                    \---------------------------------/
                               /           \
                             Sim           Não
                             /               \
                            v                 v
            +---------------------------+   +---------------------------+
            | Ler arquivo e calcular    |   | Erro de conexão ou chave  |
            | hash SHA-256 do documento |   | e abortar a execução      |
            +---------------------------+   +---------------------------+
                          |
                          v
                    /---------------------------------\
                   /     Cache de tradução válido      \
                   \       encontrado em disco?        /
                    \---------------------------------/
                               /           \
                             Sim           Não
                             /               \
                            v                 v
                    /---------------\       +---------------------------+
                   /  Deseja retomar \      | Dividir documento em      |
                   \   do cache?     /      | blocos (chunks) lógicos   |
                    \---------------/       +---------------------------+
                       /          \                       |
                     Sim          Não                     |
                     /              \                     |
                    v                v                    v
      +------------------------+   +------------------------------------+
      | Carregar progresso e   |   | Apagar cache antigo do disco       |
      | estado do cache JSON   |   +------------------------------------+
      +------------------------+                     |
                  |                                  |
                  |                                  v
                  |                    +--------------------------------+
                  |                    | Inicializar estado do grafo    |
                  |                    +--------------------------------+
                  |                                  |
                  +---------------->  +--------------+
                                    |
                                    v
       ===============================================================
                       CICLO DO GRAFO (LANGGRAPH)
       ===============================================================
                                    |
                                    v
                     +-------------------------------+
                     | Nó 1: Extrair Glossário (LLM) | (Se não houver cache)
                     +-------------------------------+
                                    |
                                    v
                                    +<----------------------------------+
                                    |                                   |
                                    v                                   |
                     +-------------------------------+                  |
                     | Nó 2: Traduzir Bloco (LLM)    |                  |
                     | (Glossário + Bloco Anterior)  |                  |
                     +-------------------------------+                  |
                                    |                                   |
                                    +<------------------+               |
                                    |                   |               |
                                    v                   |               |
                     +-------------------------------+  |               |
                     | Nó 3: Revisar Bloco (LLM)     |  |               |
                     | (Feedback ou "APROVADO")      |  |               |
                     +-------------------------------+  |               |
                                    |                   |               |
                                    v                   |               |
                              /-----------\             |               |
                             /  Aprovado   \            |               |
                            <  ou Limite    >           |               |
                             \  Atingido?  /            |               |
                              \-----------/             |               |
                               /         \              |               |
                             Não         Sim            |               |
                             /             \            |               |
                            v               v           |               |
             +---------------------------+  +------------------------+  |
             | Nó 4: Refinar Tradução    |  | Nó 5: Finalizar Bloco  |  |
             | (LLM com base no feedback)|  | (Salva Cache/Parcial)  |  |
             +---------------------------+  +------------------------+  |
                            |                           |               |
                            |                           v               |
                            +--------------------------/---------\      |
                                                      /   Mais    \     |
                                                     <   blocos?   >    |
                                                      \  (Loop)   /     |
                                                       \---------/      |
                                                        /       \       |
                                                      Não       Sim     |
                                                      /           \     |
                                                     v             v    |
                                                    |       +---------------+
                                                    |       | Nó 6: Increm. |-|
                                                    |       | Índice        |
                                                    |       +---------------+
                                                    v
       ===============================================================
                                FINALIZAÇÃO
       ===============================================================
                            |
                            v
             +-----------------------------------------------+
             | Mesclar blocos e reconstruir documento final  |
             +-----------------------------------------------+
                            |
                            v
             +-----------------------------------------------+
             | Limpar cache JSON e apagar arquivo parcial.txt|
             +-----------------------------------------------+
                            |
                            v
               +-------------------------------------------+
               |               FIM / SUCESSO               |
               +-------------------------------------------+
```

### Principais Fases do Fluxo:
1. **Carregamento de Vocabulário & Extração de Glossário**: O usuário pode passar uma lista de vocabulário personalizado (`.txt` ou `.json`) para garantir a consistência de termos chaves. O agente também realiza uma leitura inicial dinâmica das primeiras partes do texto para identificar jargões técnicos adicionais, mesclando-os (dando prioridade aos termos fornecidos pelo usuário).
2. **Tradução Iterativa (Context-Aware)**: Cada bloco do documento é traduzido levando em consideração o glossário e o *contexto do bloco anterior* (original e traduzido), mantendo a coesão geral.
3. **Reflexão Crítica**: Um revisor virtual analisa a tradução buscando erros ortográficos, inconsistências de terminologia de glossário, literalismo exagerado ou omissões.
4. **Refinamento em Loop (Multi-pass)**: Se houver críticas, a tradução entra em um ciclo de refinamento interativo (até no máximo 3 vezes) para reaplicar melhorias até obter o selo "APROVADO" ou esgotar as tentativas.
5. **Finalização do Bloco**: O bloco aprovado/refinado é consolidado, o checkpoint do cache no disco é atualizado e o arquivo de visualização parcial é atualizado.

---

## ✨ Recursos

* **Multi-formato**: Suporte nativo para carregar e traduzir documentos `.txt`, `.md`, `.pdf` (extração de texto) e `.docx` (Word).
* **Consistência Terminológica**: Extração e aplicação automática de glossário de termos para evitar que um termo técnico seja traduzido de diferentes formas.
* **CLI Interativa**: Uma interface de terminal rica desenvolvida com a biblioteca `rich`, que exibe em tempo real cada etapa do processo (tabela do glossário, críticas do revisor e refinamentos).
* **Autodetecção**: Capaz de autodetectar o idioma de origem ou aceitar definições manuais.
* **Resiliência e Retomada (Checkpoints)**: Salva o progresso bloco a bloco. Caso o processo seja interrompido, você pode continuar exatamente de onde parou.
* **Suporte Multi-Provedor**: Escolha usar **Ollama** (para rodar localmente com 100% de privacidade), **OpenAI** ou **Anthropic** (para modelos de nuvem de última geração).

---

## 💾 Resiliência de Tradução (Checkpoints) & Tradução Parcial

Para documentos longos (como livros com centenas ou milhares de blocos), o agente oferece duas camadas importantes de segurança contra perda de progresso:

### 1. Checkpoint de Progresso (Cache JSON)
O agente cria automaticamente um arquivo oculto de cache `.<nome_do_arquivo>.translation_cache.json` no mesmo diretório do documento.
*   **Identificação de Alterações**: O cache salva um hash SHA256 do arquivo original e os parâmetros de idiomas. Se o arquivo de origem for editado ou os idiomas forem alterados, o cache é invalidado para garantir a integridade da fragmentação.
*   **Retomada na CLI**: Ao rodar a CLI novamente para o mesmo arquivo, ela detecta o cache e pergunta interativamente se deseja continuar.
*   **Retomada no Streamlit**: A aplicação web detecta o progresso anterior automaticamente, oferecendo a opção de retomar a execução com um único clique.
*   **Limpeza Automática**: Assim que o documento é 100% traduzido e salvo com sucesso, o cache correspondente é apagado automaticamente.

### 2. Tradução Parcial em Texto (.txt)
Além do cache de estado técnico do LangGraph, você pode habilitar a gravação do texto traduzido acumulado em tempo real em um arquivo `.txt` físico de progresso:
*   **Nome do Arquivo**: Salvo como `<nome_de_saida>_partial.txt` (ex: `livro_translated_partial.txt`).
*   **Funcionamento**: A cada bloco finalizado e aprovado, o texto traduzido parcial acumulado é regravado no arquivo, permitindo que você acompanhe o progresso real abrindo o arquivo a qualquer momento em seu leitor favorito.
*   **Opções de Controle**:
    *   **CLI**: Use o parâmetro `--partial` ou `--no-partial` (habilitado por padrão).
    *   **Streamlit**: Ative ou desative o salvamento em tempo real através do checkbox `"Salvar tradução parcial (.txt)"` na barra lateral.
*   **Limpeza**: Assim que a tradução final com formatação original for gerada e salva com sucesso, o arquivo parcial temporário é excluído automaticamente para manter seu diretório limpo.

---

## 🚀 Como Iniciar

### Pré-requisitos

1. **Python 3.10+** instalado.
2. Dependendo do provedor que desejar utilizar:
   *   **Ollama**: Ollama instalado e rodando com o modelo desejado (p.ex., `gemma4:12b`) baixado:
       ```bash
       ollama pull gemma4:12b
       ```
   *   **OpenAI**: Uma chave de API válida configurada na variável de ambiente `OPENAI_API_KEY` ou informada nas configurações.
   *   **Anthropic**: Uma chave de API válida configurada na variável de ambiente `ANTHROPIC_API_KEY` ou informada nas configurações.

### Instalação

1. Clone este repositório no seu ambiente local:
   ```bash
   git clone https://github.com/seu-usuario/document-translation-agent.git
   cd document-translation-agent
   ```

2. Instale as dependências do projeto:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Como Utilizar

O projeto pode ser executado tanto pela linha de comando quanto via interface web (Streamlit).

### Executando via Web App (Streamlit)
Você pode rodar a interface interativa no seu navegador para traduzir arquivos de forma visual, além de escolher e configurar os provedores de LLM e chaves de API pela barra lateral:
```bash
streamlit run app.py
```

### Executando via CLI (Terminal)

#### Exemplo com Ollama (Padrão local)
```bash
python cli.py --input sample.md --target "Português"
```

#### Exemplo com OpenAI (Cloud)
```bash
python cli.py --input sample.md --provider openai --model gpt-4o-mini --api-key "SUA-CHAVE-DE-API"
```

#### Exemplo com Anthropic (Cloud)
```bash
python cli.py --input sample.md --provider anthropic --model claude-3-5-sonnet-latest --api-key "SUA-CHAVE-DE-API"
```

### Parâmetros Suportados pela CLI
```bash
Opções:
  -i, --input PATH      Caminho do arquivo de entrada (.txt, .md, .pdf, .docx). [obrigatório]
  -t, --target TEXT     Idioma de destino da tradução (padrão: Português).
  -s, --source TEXT     Idioma de origem da tradução (padrão: Autodetectar).
  -o, --output TEXT     Caminho para salvar o arquivo traduzido (padrão: nome_do_arquivo_translated.ext).
  -p, --provider TEXT   Provedor de LLM a ser utilizado ('ollama', 'openai', 'anthropic').
  -m, --model TEXT      Nome do modelo de LLM (p.ex., gpt-4o, claude-3-5-sonnet-latest, gemma4:12b).
  -k, --api-key TEXT    Chave de API do provedor (opcional se já estiver no ambiente).
  --help                Exibe o menu de ajuda.
```

---

## 📂 Estrutura do Repositório

```text
├── config.py         # Configuração centralizada (prompts, modelos, temperaturas, base URL)
├── state.py          # Definição do estado de tradução (TranslationState) do LangGraph
├── models.py         # Inicialização dos LLMs do Ollama com LangChain
├── helpers.py        # Helpers para retentativas de LLM, parser de JSON e tokens
├── nodes.py          # Nós do grafo (extrair glossário, traduzir, revisar, refinar, etc.)
├── agent.py          # Compilação e estruturação do fluxo do LangGraph
├── cli.py            # Interface de linha de comando rica usando rich e click
├── utils.py          # Utilitários de leitura de arquivo, cache e chunking inteligente
├── sample.md         # Documento markdown técnico de amostra para testes
├── requirements.txt  # Lista de dependências Python (incluindo streamlit)
├── .gitignore        # Configuração do git para ignorar arquivos temporários e caches
└── README.md         # Documentação do projeto
```


---

## ⚖️ Licença

Este projeto é distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para obter mais informações.
