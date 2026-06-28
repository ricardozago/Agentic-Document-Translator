from typing import List, Dict, Any, TypedDict

class TranslationState(TypedDict):
    source_text: str
    source_lang: str
    target_lang: str
    glossary: Dict[str, str]
    chunks: List[str]
    current_chunk_idx: int
    translated_chunks: List[str]  # Traduções iniciais/brutas de cada bloco
    reflection_feedback: str      # Críticas do revisor para o bloco atual
    final_chunks: List[str]       # Traduções finais pós-refinamento
    refinement_count: int         # Número de vezes que o bloco atual foi refinado
    current_translation: str      # Tradução ativa sob revisão/refinamento
    total_prompt_tokens: int      # Total de tokens de prompt processados
    total_completion_tokens: int  # Total de tokens de resposta gerados
    last_node_tokens: Dict[str, int]  # Tokens do último passo {"prompt": X, "completion": Y}
