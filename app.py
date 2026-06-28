import streamlit as st
import os
import time
import urllib.request
from agent import compile_translation_graph
import utils
import config

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Agente de Tradução de Documentos",
    page_icon="🌐",
    layout="wide"
)

# Estilização CSS customizada
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #6C63FF;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-connected {
        color: #2ec4b6;
        font-weight: 600;
    }
    .status-disconnected {
        color: #e63946;
        font-weight: 600;
    }
    .chunk-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #6C63FF;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌐 Agente de Tradução Inteligente</div>', unsafe_allow_html=True)

# 1. Verificar conexão com o Ollama
ollama_connected = False
try:
    urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2)
    ollama_connected = True
except Exception:
    pass

# Barra Lateral (Sidebar) de Configurações
st.sidebar.title("Configurações do Agente")

provider = st.sidebar.selectbox(
    "Provedor de LLM",
    ["Ollama", "OpenAI", "Anthropic"],
    index=["ollama", "openai", "anthropic"].index(config.LLM_PROVIDER.lower())
)

provider_lower = provider.lower()

# Parâmetros dinâmicos do provedor
api_key = ""
model_name = ""
base_url = ""

if provider_lower == "ollama":
    st.sidebar.markdown("### Configurações do Ollama")
    if ollama_connected:
        st.sidebar.markdown(f'Status: <span class="status-connected">● Conectado ({config.OLLAMA_BASE_URL})</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f'Status: <span class="status-disconnected">● Desconectado ({config.OLLAMA_BASE_URL})</span>', unsafe_allow_html=True)
        st.sidebar.warning("Ollama local não pôde ser alcançado. Certifique-se de que o serviço está rodando.")
    base_url = st.sidebar.text_input("Ollama Base URL", value=config.OLLAMA_BASE_URL)
    model_name = st.sidebar.text_input("Nome do Modelo", value=config.OLLAMA_MODEL)
    
elif provider_lower == "openai":
    st.sidebar.markdown("### Configurações da OpenAI")
    default_key = config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    api_key = st.sidebar.text_input("OpenAI API Key", value=default_key, type="password")
    model_name = st.sidebar.text_input("Nome do Modelo", value=config.OPENAI_MODEL)
    
elif provider_lower == "anthropic":
    st.sidebar.markdown("### Configurações da Anthropic")
    default_key = config.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
    api_key = st.sidebar.text_input("Anthropic API Key", value=default_key, type="password")
    model_name = st.sidebar.text_input("Nome do Modelo", value=config.ANTHROPIC_MODEL)

st.markdown(f'<div class="subtitle">Tradução Agentica em tempo real usando LangGraph e {provider} ({model_name})</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")

source_lang = st.sidebar.selectbox(
    "Idioma de Origem (Source)",
    ["Autodetectar", "Inglês", "Português", "Espanhol", "Francês", "Alemão", "Italiano", "Japonês", "Chinês"]
)

target_lang = st.sidebar.selectbox(
    "Idioma de Destino (Target)",
    ["Português", "Inglês", "Espanhol", "Francês", "Alemão", "Italiano", "Japonês", "Chinês"],
    index=0
)

save_partial = st.sidebar.checkbox("Salvar tradução parcial (.txt)", value=True)
vocab_file = st.sidebar.file_uploader("Vocabulário Personalizado (.txt, .json)", type=["txt", "json"])

custom_vocab = {}
if vocab_file is not None:
    try:
        custom_vocab = utils.parse_vocabulary_content(vocab_file.getvalue(), vocab_file.name)
        st.sidebar.success(f"✔ Vocabulário carregado: {len(custom_vocab)} termos.")
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar vocabulário: {e}")

# Layout Principal da App
col_orig, col_dest = st.columns(2)

uploaded_file = st.file_uploader("Carregue seu documento (.txt, .md, .pdf, .docx)", type=["txt", "md", "pdf", "docx"])

if uploaded_file is not None:
    # Grava o arquivo carregado temporariamente para leitura
    temp_filename = f"temp_{uploaded_file.name}"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        document_text = utils.read_document(temp_filename)
        preview_text = document_text[:1200] + ("\n\n... [Texto truncado para prévia] ..." if len(document_text) > 1200 else "")
        
        # Calcular nomes de arquivos de saída
        base, ext = os.path.splitext(uploaded_file.name)
        out_filename = f"{base}_translated{ext}"
        partial_filename = f"{base}_translated_partial.txt"
        
        with col_orig:
            st.subheader("📄 Prévia do Documento Original")
            st.text_area("Texto extraído:", value=preview_text, height=350, disabled=True)
            
            # Verificar cache de tradução
            cache = utils.load_translation_cache(temp_filename, target_lang, source_lang if source_lang != "Autodetectar" else "")
            resume = False
            
            if cache:
                cached_idx = cache.get("current_chunk_idx", 0)
                cached_total = len(cache.get("chunks", []))
                if cached_idx < cached_total:
                    st.warning(f"📝 Cache de tradução detectado! Tradução anterior interrompida no bloco {cached_idx + 1}/{cached_total}.")
                    resume_option = st.radio(
                        "Deseja retomar a tradução?",
                        [f"Sim, retomar do bloco {cached_idx + 1}", "Não, iniciar do zero (descartar cache)"]
                    )
                    resume = (resume_option.startswith("Sim"))
                    if not resume:
                        if st.button("Descartar Cache Permanentemente"):
                            utils.clear_translation_cache(temp_filename)
                            st.rerun()
                else:
                    utils.clear_translation_cache(temp_filename)
            
            # Heurística para chunks
            chunks = utils.split_text(document_text)
            
            # Desativar o botão se o Ollama estiver offline ou se a chave de API estiver vazia para OpenAI/Anthropic
            can_translate = True
            if provider_lower == "ollama" and not ollama_connected:
                can_translate = False
            elif provider_lower in ["openai", "anthropic"] and not api_key:
                can_translate = False
                st.sidebar.warning("Insira uma chave de API para prosseguir.")
                
            translate_button = st.button("Iniciar Tradução do Agente", disabled=not can_translate, use_container_width=True)
            
        if translate_button:
            # Inicializar os LLMs com os parâmetros dinâmicos da UI
            import models
            try:
                models.init_models(
                    provider=provider_lower,
                    model_name=model_name,
                    base_url=base_url if provider_lower == "ollama" else None,
                    api_key=api_key if provider_lower in ["openai", "anthropic"] else None
                )
            except Exception as e:
                st.error(f"Erro ao inicializar modelos do provedor {provider}: {e}")
                st.stop()
            progress_bar = st.progress(0)
            status_text = st.empty()
            glossary_area = st.empty()
            log_container = st.container()
            
            # Estado Inicial do LangGraph (com ou sem cache)
            if resume and cache:
                glossary_to_use = cache.get("glossary", {})
                if custom_vocab:
                    for k, v in custom_vocab.items():
                        glossary_to_use[k] = v
                initial_state = {
                    "source_text": document_text,
                    "source_lang": source_lang if source_lang != "Autodetectar" else "",
                    "target_lang": target_lang,
                    "glossary": glossary_to_use,
                    "chunks": chunks,
                    "current_chunk_idx": cache.get("current_chunk_idx", 0),
                    "translated_chunks": cache.get("translated_chunks", []),
                    "reflection_feedback": "",
                    "final_chunks": cache.get("final_chunks", []),
                    "refinement_count": 0,
                    "current_translation": "",
                    "total_prompt_tokens": cache.get("total_prompt_tokens", 0),
                    "total_completion_tokens": cache.get("total_completion_tokens", 0),
                    "last_node_tokens": {"prompt": 0, "completion": 0}
                }
                current_chunk_idx = cache.get("current_chunk_idx", 0)
                glossary = glossary_to_use
                final_translation_chunks = cache.get("final_chunks", [])
            else:
                initial_state = {
                    "source_text": document_text,
                    "source_lang": source_lang if source_lang != "Autodetectar" else "",
                    "target_lang": target_lang,
                    "glossary": custom_vocab,
                    "chunks": chunks,
                    "current_chunk_idx": 0,
                    "translated_chunks": [],
                    "reflection_feedback": "",
                    "final_chunks": [],
                    "refinement_count": 0,
                    "current_translation": "",
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "last_node_tokens": {"prompt": 0, "completion": 0}
                }
                current_chunk_idx = 0
                glossary = custom_vocab
                final_translation_chunks = []
            graph = compile_translation_graph()
            
            status_text.info(f"Executando tradução... (Iniciando no Bloco {current_chunk_idx + 1})")
            
            with log_container:
                # Inicializar contadores de tokens
                total_prompt_tokens = 0
                total_completion_tokens = 0
                
                # Loop de Streaming do Grafo
                for event in graph.stream(initial_state):
                    for node_name, state_update in event.items():
                        # Acumular tokens
                        if "total_prompt_tokens" in state_update:
                            total_prompt_tokens = state_update["total_prompt_tokens"]
                        if "total_completion_tokens" in state_update:
                            total_completion_tokens = state_update["total_completion_tokens"]
                            
                        if node_name == "extract_glossary":
                            glossary = state_update.get("glossary", {})
                            tokens = state_update.get("last_node_tokens", {})
                            token_info = f" (Tokens: Entrada: {tokens.get('prompt', 0)} | Saída: {tokens.get('completion', 0)})" if tokens and (tokens.get('prompt', 0) > 0 or tokens.get('completion', 0) > 0) else ""
                            
                            if glossary:
                                with glossary_area.expander(f"🔑 Glossário de Termos Identificados{token_info}", expanded=True):
                                    st.table(glossary)
                            status_text.info(f"Traduzindo Bloco {current_chunk_idx + 1}...")
                            
                        elif node_name == "translate_chunk":
                            idx = state_update.get("current_chunk_idx", 0)
                            latest_raw = state_update.get("current_translation", "")
                            tokens = state_update.get("last_node_tokens", {})
                            token_info = f" (Tokens: Entrada: {tokens.get('prompt', 0)} | Saída: {tokens.get('completion', 0)})" if tokens else ""
                            
                            st.markdown(f"✍️ **[Bloco {idx + 1}/{len(chunks)}] Tradução Proposta (Inicial){token_info}:**")
                            st.code(latest_raw, language="markdown")
                            
                        elif node_name == "reflect_chunk":
                            idx = state_update.get("current_chunk_idx", 0)
                            feedback = state_update.get("reflection_feedback", "")
                            ref_count = state_update.get("refinement_count", 0)
                            tokens = state_update.get("last_node_tokens", {})
                            token_info = f" (Tokens: Entrada: {tokens.get('prompt', 0)} | Saída: {tokens.get('completion', 0)})" if tokens else ""
                            
                            if "APROVADO" in feedback.upper() and len(feedback.strip()) < 15:
                                if ref_count == 0:
                                    st.success(f"✔ [Bloco {idx + 1}/{len(chunks)}] Aprovado diretamente pelo revisor!{token_info}")
                                else:
                                    st.success(f"✔ [Bloco {idx + 1}/{len(chunks)}] Aprovado pelo revisor após {ref_count} refinamento(s)!{token_info}")
                            else:
                                title_suffix = f" (Revisão #{ref_count + 1}){token_info}"
                                st.warning(f"⚠ [Bloco {idx + 1}/{len(chunks)}] Crítica do Revisor{title_suffix}:\n{feedback}")
                                
                        elif node_name == "refine_chunk":
                            idx = state_update.get("current_chunk_idx", 0)
                            refined = state_update.get("current_translation", "")
                            ref_count = state_update.get("refinement_count", 0)
                            tokens = state_update.get("last_node_tokens", {})
                            token_info = f" (Tokens: Entrada: {tokens.get('prompt', 0)} | Saída: {tokens.get('completion', 0)})" if tokens else ""
                            
                            st.markdown(f"🔄 **[Bloco {idx + 1}/{len(chunks)}] Refinamento (Tentativa #{ref_count}){token_info}:**")
                            st.code(refined, language="markdown")
                            
                        elif node_name == "finalize_chunk":
                            idx = state_update.get("current_chunk_idx", 0)
                            final_chunks = state_update.get("final_chunks", [])
                            translated_chunks = state_update.get("translated_chunks", [])
                            
                            if final_chunks:
                                final_translation_chunks = final_chunks
                                
                            # Salvar cache após finalizar o bloco
                            utils.save_translation_cache(
                                file_path=temp_filename,
                                target_lang=target_lang,
                                source_lang=source_lang if source_lang != "Autodetectar" else "",
                                current_chunk_idx=idx + 1,
                                translated_chunks=translated_chunks,
                                final_chunks=final_translation_chunks,
                                glossary=glossary,
                                chunks=chunks
                            )
                            
                            # Salvar tradução parcial se habilitado
                            if save_partial:
                                try:
                                    partial_content = "\n\n".join(final_translation_chunks)
                                    utils.write_document(partial_filename, partial_content)
                                except Exception:
                                    pass
                                    
                            # Atualiza barra de progresso
                            prog_val = int(((idx + 1) / len(chunks)) * 100)
                            progress_bar.progress(prog_val)
                            
                        elif node_name == "increment_index":
                            idx = state_update.get("current_chunk_idx", 0)
                            status_text.info(f"Traduzindo Bloco {idx + 1}...")
                            
            status_text.success(f"🎉 Tradução concluída com sucesso! Contagem total de tokens: Entrada: {total_prompt_tokens} | Saída: {total_completion_tokens} (Total: {total_prompt_tokens + total_completion_tokens})")
            st.balloons()
            
            # Exibir resultados finais e download
            if final_translation_chunks:
                final_content = "\n\n".join(final_translation_chunks)
                
                # Limpar o cache de tradução após o sucesso total
                utils.clear_translation_cache(temp_filename)
                
                # Limpar arquivo de tradução parcial se existir
                if save_partial and os.path.exists(partial_filename):
                    try:
                        os.remove(partial_filename)
                    except Exception:
                        pass
                
                utils.write_document(out_filename, final_content)
                
                st.subheader("📝 Resultado da Tradução")
                tab_preview, tab_download = st.tabs(["Prévia do Texto Traduzido", "Download do Arquivo"])
                
                with tab_preview:
                    st.text_area("Texto Traduzido Completo", value=final_content, height=400)
                    
                with tab_download:
                    with open(out_filename, "rb") as f:
                        st.download_button(
                            label="⬇ Baixar Documento Traduzido",
                            data=f,
                            file_name=out_filename,
                            mime="application/octet-stream"
                        )
                    st.success(f"O arquivo foi salvo localmente no repositório como: **{out_filename}**")
                    
    except Exception as e:
        st.error(f"Erro no processamento do agente: {e}")
    finally:
        # Remover arquivo temporário
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
else:
    st.info("Envie um arquivo para começar.")
