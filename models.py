from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import config

# Variáveis globais para os LLMs usados nos nós do grafo
llm_glossary = None
llm_translation = None
llm_reflection = None

def init_models(
    provider: str = None,
    model_name: str = None,
    base_url: str = None,
    api_key: str = None
):
    """
    Inicializa dinamicamente os modelos de linguagem com base no provedor escolhido.
    """
    global llm_glossary, llm_translation, llm_reflection
    
    prov = (provider or config.LLM_PROVIDER).lower()
    
    if prov == "ollama":
        model = model_name or config.OLLAMA_MODEL
        url = base_url or config.OLLAMA_BASE_URL
        llm_glossary = ChatOllama(model=model, base_url=url, temperature=config.TEMP_GLOSSARY)
        llm_translation = ChatOllama(model=model, base_url=url, temperature=config.TEMP_TRANSLATION)
        llm_reflection = ChatOllama(model=model, base_url=url, temperature=config.TEMP_REFLECTION)
        
    elif prov == "openai":
        model = model_name or config.OPENAI_MODEL
        key = api_key or config.OPENAI_API_KEY
        llm_glossary = ChatOpenAI(model=model, api_key=key if key else None, temperature=config.TEMP_GLOSSARY)
        llm_translation = ChatOpenAI(model=model, api_key=key if key else None, temperature=config.TEMP_TRANSLATION)
        llm_reflection = ChatOpenAI(model=model, api_key=key if key else None, temperature=config.TEMP_REFLECTION)
        
    elif prov == "anthropic":
        model = model_name or config.ANTHROPIC_MODEL
        key = api_key or config.ANTHROPIC_API_KEY
        llm_glossary = ChatAnthropic(model=model, api_key=key if key else None, temperature=config.TEMP_GLOSSARY)
        llm_translation = ChatAnthropic(model=model, api_key=key if key else None, temperature=config.TEMP_TRANSLATION)
        llm_reflection = ChatAnthropic(model=model, api_key=key if key else None, temperature=config.TEMP_REFLECTION)
        
    else:
        raise ValueError(f"Provedor de LLM inválido ou não suportado: {prov}")

# Inicializa os modelos padrão no carregamento do módulo
init_models()
