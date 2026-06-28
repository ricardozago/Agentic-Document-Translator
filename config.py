import os

# --- Provedor de LLM Ativo ---
# Opções: 'ollama', 'openai', 'anthropic'
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# --- Configurações Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:12b")

# --- Configurações OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# --- Configurações Anthropic ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

# --- Temperaturas do Fluxo ---
TEMP_GLOSSARY = float(os.getenv("TEMP_GLOSSARY", "0.0"))
TEMP_TRANSLATION = float(os.getenv("TEMP_TRANSLATION", "0.2"))
TEMP_REFLECTION = float(os.getenv("TEMP_REFLECTION", "0.1"))

# --- Helper para obter o modelo ativo ---
def get_active_model_name():
    if LLM_PROVIDER == "ollama":
        return OLLAMA_MODEL
    elif LLM_PROVIDER == "openai":
        return OPENAI_MODEL
    elif LLM_PROVIDER == "anthropic":
        return ANTHROPIC_MODEL
    return "unknown"


# --- Prompts do Agente ---

PROMPT_GLOSSARY = """Você é um especialista em terminologia e tradução.
Sua tarefa é analisar o trecho de documento fornecido abaixo e extrair um glossário contendo termos técnicos, jargões, termos específicos do domínio ou nomes próprios que necessitam de tradução consistente.

Idioma de Origem: {source_lang}
Idioma de Destino: {target_lang}

Gere o glossário mapeando o termo no idioma de origem para a melhor tradução no idioma de destino.
Você deve retornar APENAS um objeto JSON válido contendo os pares chave-valor de tradução, sem qualquer outra introdução, explicação ou código markdown.

Exemplo de saída esperada:
{{
  "machine learning": "aprendizado de máquina",
  "gradient descent": "gradiente descente"
}}

Texto para análise:
{sample_text}
"""

PROMPT_TRANSLATION = """Você é um tradutor literário e técnico profissional.
Traduza o trecho de texto fornecido abaixo {lang_instruction}. Mantenha a mesma formatação Markdown (como #, *, ```) e estilo original do autor.
Você pode ajustar quebras de linha que podem ter ocorrido a problemas na extração de texto do documento, mas não remova ou altere qualquer formatação original do autor (ex: #, *, ```).

{prev_context}

Glossário de referência:
{glossary_str}

Texto a ser traduzido:
{chunk}

Tradução (não explique nada, envie apenas a tradução, pois o que vier a seguir será utilizado diretamente):"""

PROMPT_REFLECTION = """Você é um revisor de tradução experiente e criterioso.
Sua tarefa é analisar criticamente a tradução proposta para um trecho de documento original de {source_lang} para {target_lang}.

Texto Original:
{chunk}

Tradução Proposta:
{proposed_translation}

Diretrizes de Glossário:
{glossary_str}

Avalie a tradução proposta sob os seguintes aspectos:
1. Erros gramaticais, de concordância ou ortografia no idioma de destino.
2. Fidelidade de conteúdo (frases omitidas ou significados alterados).
3. Consistência terminológica de acordo com o glossário de referência.
4. Fluidez da leitura e tom natural no idioma de destino.

Se a tradução estiver correta, fluida e precisa, responda APENAS: APROVADO
Caso precise de melhorias, aponte objetivamente e em poucas linhas as correções recomendadas.

Crítica:"""

PROMPT_REFINEMENT = """Você é um tradutor especialista. Sua missão é refinar a tradução proposta para o texto original com base nas sugestões e críticas do revisor.
Você pode ajustar quebras de linha que podem ter ocorrido a problemas na extração de texto do documento, mas não remova ou altere qualquer formatação original do autor (ex: #, *, ```).

Texto Original:
{chunk}

Tradução Proposta Inicial:
{proposed_translation}

Feedback do Revisor:
{feedback}

Aplique as correções necessárias para melhorar o texto, mantendo a formatação Markdown e o estilo original.

Tradução Refinada Final (não explique nada, envie apenas a tradução, pois o que vier a seguir será utilizado diretamente):"""
