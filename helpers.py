import json
import re
import time
from typing import Dict

# Helper para invocar o LLM com retentativas automáticas em caso de falha de conexão
def invoke_llm_with_retry(llm_instance, prompt: str, max_retries: int = 3, delay: float = 3.0):
    for i in range(max_retries):
        try:
            return llm_instance.invoke(prompt)
        except Exception as e:
            if i == max_retries - 1:
                # Se for a última tentativa, propaga o erro
                raise e
            # Espera um tempo antes de tentar novamente (útil se o WSL/Ollama estiver temporariamente sobrecarregado)
            time.sleep(delay)

# Extrai a contagem de tokens de forma resiliente a partir do objeto de resposta do LangChain
def extract_token_usage(response) -> Dict[str, int]:
    prompt = 0
    completion = 0
    
    # 1. Tentar via usage_metadata (comum em versões recentes do LangChain)
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        prompt = response.usage_metadata.get('input_tokens', 0)
        completion = response.usage_metadata.get('output_tokens', 0)
        
    # 2. Tentar via response_metadata (específico de Ollama/outros provedores)
    if not prompt or not completion:
        meta = getattr(response, 'response_metadata', {}) or {}
        prompt = meta.get('prompt_eval_count') or meta.get('token_usage', {}).get('prompt_tokens', 0)
        completion = meta.get('eval_count') or meta.get('token_usage', {}).get('completion_tokens', 0)
        
    return {
        "prompt": prompt or 0,
        "completion": completion or 0,
        "total": (prompt or 0) + (completion or 0)
    }

# Função auxiliar para extrair JSON de texto de forma resiliente
def parse_json_glossary(text: str) -> Dict[str, str]:
    text_clean = text.strip()
    
    # 1. Tentar encontrar blocos de código markdown ```json ... ```
    match = re.search(r'```json\s*(.*?)\s*```', text_clean, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # 2. Tentar encontrar a partir do primeiro '{' até o último '}'
        match_braces = re.search(r'(\{.*\})', text_clean, re.DOTALL)
        if match_braces:
            json_str = match_braces.group(1).strip()
        else:
            json_str = text_clean
            
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            return {str(k).strip(): str(v).strip() for k, v in data.items()}
    except Exception:
        pass
        
    return {}
