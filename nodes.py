from typing import Dict, Any, List
from langgraph.graph import END

import config
import models
import helpers
from state import TranslationState

# Nó 1: Extração de Glossário
def extract_glossary(state: TranslationState) -> Dict[str, Any]:
    # Mantém o glossário existente (que pode incluir o vocabulário customizado do usuário)
    glossary = dict(state.get("glossary") or {})
    
    # Se já passou do primeiro bloco, não re-extrai
    if state.get("current_chunk_idx", 0) > 0:
        return {"glossary": glossary}
        
    source_lang = state.get("source_lang") or "Autodetectar"
    target_lang = state.get("target_lang") or "Português"
    chunks = state.get("chunks", [])
    
    # Analisa até as primeiras 3 partes do texto para economizar contexto
    sample_text = "\n\n".join(chunks[:3])
    
    prompt = config.PROMPT_GLOSSARY.format(
        source_lang=source_lang,
        target_lang=target_lang,
        sample_text=sample_text
    )
    
    prompt_tokens = 0
    completion_tokens = 0
    try:
        response = helpers.invoke_llm_with_retry(models.llm_glossary, prompt)
        extracted_glossary = helpers.parse_json_glossary(response.content)
        # Mesclar o glossário extraído. Os termos de vocabulário do usuário têm prioridade absoluta.
        for k, v in extracted_glossary.items():
            k_lower = k.lower().strip()
            # Verifica se o termo já foi fornecido pelo usuário (ignorando maiúsculas/minúsculas)
            if not any(ek.lower().strip() == k_lower for ek in glossary):
                glossary[k.strip()] = v.strip()
        tokens = helpers.extract_token_usage(response)
        prompt_tokens = tokens["prompt"]
        completion_tokens = tokens["completion"]
    except Exception:
        pass
        
    return {
        "glossary": glossary,
        "total_prompt_tokens": state.get("total_prompt_tokens", 0) + prompt_tokens,
        "total_completion_tokens": state.get("total_completion_tokens", 0) + completion_tokens,
        "last_node_tokens": {"prompt": prompt_tokens, "completion": completion_tokens}
    }

# Nó 2: Tradução Inicial do Bloco Atual
def translate_chunk(state: TranslationState) -> Dict[str, Any]:
    idx = state.get("current_chunk_idx", 0)
    chunks = state.get("chunks", [])
    chunk = chunks[idx]
    
    source_lang = state.get("source_lang") or ""
    target_lang = state.get("target_lang") or "Português"
    glossary = state.get("glossary", {})
    
    # Formatação do glossário para o prompt
    if glossary:
        glossary_str = "\n".join([f"- {k} -> {v}" for k, v in glossary.items()])
    else:
        glossary_str = "(Nenhum termo extraído no glossário)"
        
    # Instrução de idioma de forma natural
    if source_lang and source_lang != "Autodetectar":
        lang_instruction = f"de {source_lang} para {target_lang}"
    else:
        lang_instruction = f"para {target_lang} (identifique o idioma de origem automaticamente)"
        
    # Contexto do bloco anterior para coesão
    prev_context = ""
    if idx > 0:
        prev_source = chunks[idx - 1]
        final_chunks = state.get("final_chunks", [])
        prev_target = final_chunks[idx - 1] if idx - 1 < len(final_chunks) else ""
        prev_context = f"""--- Coerência com Bloco Anterior ---
Texto Original Anterior: {prev_source}
Tradução Anterior: {prev_target}
------------------------------------"""

    prompt = config.PROMPT_TRANSLATION.format(
        lang_instruction=lang_instruction,
        prev_context=prev_context,
        glossary_str=glossary_str,
        chunk=chunk
    )
    
    response = helpers.invoke_llm_with_retry(models.llm_translation, prompt)
    tokens = helpers.extract_token_usage(response)
    
    translated_chunks = list(state.get("translated_chunks", []))
    draft = response.content.strip()
    translated_chunks.append(draft)
    
    return {
        "translated_chunks": translated_chunks,
        "current_translation": draft,
        "refinement_count": 0,
        "total_prompt_tokens": state.get("total_prompt_tokens", 0) + tokens["prompt"],
        "total_completion_tokens": state.get("total_completion_tokens", 0) + tokens["completion"],
        "last_node_tokens": tokens
    }

# Nó 3: Reflexão (Revisão da Tradução)
def reflect_chunk(state: TranslationState) -> Dict[str, Any]:
    idx = state.get("current_chunk_idx", 0)
    chunks = state.get("chunks", [])
    chunk = chunks[idx]
    
    source_lang = state.get("source_lang") or "Autodetectar"
    target_lang = state.get("target_lang") or "Português"
    glossary = state.get("glossary", {})
    
    proposed_translation = state.get("current_translation") or ""
    refinement_count = state.get("refinement_count", 0)
    
    if glossary:
        glossary_str = "\n".join([f"- {k} -> {v}" for k, v in glossary.items()])
    else:
        glossary_str = "(Nenhum)"
        
    prompt = config.PROMPT_REFLECTION.format(
        source_lang=source_lang,
        target_lang=target_lang,
        chunk=chunk,
        proposed_translation=proposed_translation,
        glossary_str=glossary_str
    )
    
    response = helpers.invoke_llm_with_retry(models.llm_reflection, prompt)
    tokens = helpers.extract_token_usage(response)
    
    return {
        "reflection_feedback": response.content.strip(),
        "refinement_count": refinement_count,
        "total_prompt_tokens": state.get("total_prompt_tokens", 0) + tokens["prompt"],
        "total_completion_tokens": state.get("total_completion_tokens", 0) + tokens["completion"],
        "last_node_tokens": tokens
    }

# Nó 4: Refinamento da Tradução com base no Feedback
def refine_chunk(state: TranslationState) -> Dict[str, Any]:
    idx = state.get("current_chunk_idx", 0)
    chunks = state.get("chunks", [])
    chunk = chunks[idx]
    
    proposed_translation = state.get("current_translation") or ""
    feedback = state.get("reflection_feedback", "")
    refinement_count = state.get("refinement_count", 0)
    
    prompt = config.PROMPT_REFINEMENT.format(
        chunk=chunk,
        proposed_translation=proposed_translation,
        feedback=feedback
    )
    
    response = helpers.invoke_llm_with_retry(models.llm_translation, prompt)
    refined_translation = response.content.strip()
    tokens = helpers.extract_token_usage(response)
    
    return {
        "current_translation": refined_translation,
        "refinement_count": refinement_count + 1,
        "total_prompt_tokens": state.get("total_prompt_tokens", 0) + tokens["prompt"],
        "total_completion_tokens": state.get("total_completion_tokens", 0) + tokens["completion"],
        "last_node_tokens": tokens
    }

# Nó 5: Finalizar o Bloco Atual
def finalize_chunk(state: TranslationState) -> Dict[str, Any]:
    final_chunks = list(state.get("final_chunks", []))
    current_translation = state.get("current_translation") or ""
    final_chunks.append(current_translation)
    return {"final_chunks": final_chunks}

# Nó 6: Incrementar o índice do bloco
def increment_index(state: TranslationState) -> Dict[str, Any]:
    return {"current_chunk_idx": state.get("current_chunk_idx", 0) + 1}

# Roteador condicional após a reflexão: decide se vai refinar ou finalizar
def route_after_reflect(state: TranslationState) -> str:
    feedback = state.get("reflection_feedback", "")
    refinement_count = state.get("refinement_count", 0)
    
    is_approved = "APROVADO" in feedback.upper() and len(feedback.strip()) < 15
    if is_approved or refinement_count >= 3:
        return "finalize"
    return "refine"

# Roteador condicional após a finalização: decide se vai para o próximo ou fim
def route_after_finalize(state: TranslationState) -> str:
    idx = state.get("current_chunk_idx", 0)
    chunks = state.get("chunks", [])
    if idx + 1 < len(chunks):
        return "next"
    return "end"
