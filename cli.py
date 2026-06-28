import sys
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import click
import time
import os
import urllib.request
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent import compile_translation_graph
import utils
import config

console = Console()

@click.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Caminho do arquivo de entrada (.txt, .md, .pdf, .docx).')
@click.option('--target', '-t', default='Português', help='Idioma de destino da tradução (padrão: Português).')
@click.option('--source', '-s', default='Autodetectar', help='Idioma de origem da tradução (padrão: Autodetectar).')
@click.option('--output', '-o', default=None, help='Caminho do arquivo de saída.')
@click.option('--partial/--no-partial', default=True, help='Salvar tradução parcial em um arquivo .txt durante o processamento (padrão: True).')
@click.option('--vocabulary', '-v', default=None, type=click.Path(exists=True), help='Caminho do arquivo de vocabulário (.txt ou .json) para manter consistência.')
@click.option('--provider', '-p', default=None, type=click.Choice(['ollama', 'openai', 'anthropic'], case_sensitive=False), help='Provedor de LLM a ser utilizado.')
@click.option('--model', '-m', default=None, help='Nome do modelo de LLM (p.ex., gpt-4o, claude-3-5-sonnet-latest, gemma4:12b).')
@click.option('--api-key', '-k', default=None, help='Chave de API do provedor selecionado (opcional se já definida no ambiente).')
def translate_cli(input, target, source, output, partial, vocabulary, provider, model, api_key):
    """
    Agente de Tradução de Documentos usando LangGraph e LLMs.
    """
    import models

    # Determinar provedor e modelo resolvidos
    provider_resolved = (provider or config.LLM_PROVIDER).lower()
    
    if provider_resolved == "ollama":
        model_resolved = model or config.OLLAMA_MODEL
        # 0. Validar conexão com o Ollama
        try:
            urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        except Exception:
            console.print(f"\n[bold red]✖ ERRO DE CONEXÃO:[/bold red] Não foi possível acessar o Ollama na rota [cyan]{config.OLLAMA_BASE_URL}[/cyan].")
            console.print("[yellow]Verifique se o Ollama está rodando e exposto corretamente.[/yellow]\n")
            return
            
    elif provider_resolved == "openai":
        model_resolved = model or config.OPENAI_MODEL
        resolved_key = api_key or config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            console.print("\n[bold red]✖ ERRO DE AUTENTICAÇÃO:[/bold red] Chave de API da OpenAI não encontrada.")
            console.print("[yellow]Informe a chave através do parâmetro --api-key / -k ou configure a variável de ambiente OPENAI_API_KEY.[/yellow]\n")
            return
            
    elif provider_resolved == "anthropic":
        model_resolved = model or config.ANTHROPIC_MODEL
        resolved_key = api_key or config.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            console.print("\n[bold red]✖ ERRO DE AUTENTICAÇÃO:[/bold red] Chave de API da Anthropic não encontrada.")
            console.print("[yellow]Informe a chave através do parâmetro --api-key / -k ou configure a variável de ambiente ANTHROPIC_API_KEY.[/yellow]\n")
            return
            
    else:
        console.print(f"\n[bold red]✖ ERRO:[/bold red] Provedor [cyan]{provider_resolved}[/cyan] não suportado.")
        return

    # Inicializar os LLMs com os parâmetros resolvidos
    try:
        models.init_models(
            provider=provider_resolved,
            model_name=model_resolved,
            api_key=api_key
        )
    except Exception as e:
        console.print(f"\n[bold red]✖ Erro ao inicializar modelos:[/bold red] {e}")
        return

    # Determinar caminho do arquivo de saída
    if not output:
        base, ext = os.path.splitext(input)
        output = f"{base}_translated{ext}"
        
    partial_output = f"{os.path.splitext(output)[0]}_partial.txt"
        
    console.print(Panel(
        f"[bold magenta]AGENTE DE TRADUÇÃO INTELIGENTE[/bold magenta]\n"
        f"[dim]LangGraph + {provider_resolved.capitalize()} ({model_resolved})[/dim]\n\n"
        f"[bold cyan]Arquivo de Entrada :[/bold cyan] [white]{input}[/white]\n"
        f"[bold cyan]Idioma de Origem   :[/bold cyan] [white]{source}[/white]\n"
        f"[bold cyan]Idioma de Destino  :[/bold cyan] [white]{target}[/white]\n"
        f"[bold cyan]Arquivo de Saída   :[/bold cyan] [white]{output}[/white]" + 
        (f"\n[bold cyan]Tradução Parcial   :[/bold cyan] [white]{partial_output}[/white]" if partial else "") +
        (f"\n[bold cyan]Vocabulário        :[/bold cyan] [white]{vocabulary}[/white]" if vocabulary else ""),
        title="[bold green]Parâmetros de Tradução[/bold green]",
        border_style="magenta",
        padding=(1, 2)
    ))
    
    # Carregar vocabulário customizado se fornecido
    custom_vocab = {}
    if vocabulary:
        try:
            custom_vocab = utils.load_vocabulary(vocabulary)
            console.print(f"[bold green]✔[/bold green] Vocabulário personalizado carregado: [bold cyan]{len(custom_vocab)}[/bold cyan] termos.\n")
        except Exception as e:
            console.print(f"[bold red]✖ Erro ao carregar vocabulário:[/bold red] {e}")
            return
    
    # 1. Carregar documento
    with console.status("[bold yellow]Carregando documento...[/bold yellow]", spinner="dots") as status:
        try:
            document_text = utils.read_document(input)
        except Exception as e:
            console.print(f"[bold red]✖ Erro ao ler o arquivo:[/bold red] {e}")
            return
            
    # 2. Dividir em blocos (chunking)
    with console.status("[bold yellow]Dividindo texto em blocos inteligentes...[/bold yellow]", spinner="dots") as status:
        chunks = utils.split_text(document_text)
        num_chunks = len(chunks)
        
    console.print(f"[bold green]✔[/bold green] Documento carregado. Total de [bold cyan]{num_chunks}[/bold cyan] blocos para traduzir.\n")
    
    if num_chunks == 0:
        console.print("[bold red]✖ Documento vazio ou sem conteúdo de texto extraível.[/bold red]")
        return
        
    # Verificar se existe cache para retomar
    cache = utils.load_translation_cache(input, target, source)
    resume = False
    
    if cache:
        cached_idx = cache.get("current_chunk_idx", 0)
        cached_total = len(cache.get("chunks", []))
        if cached_idx < cached_total:
            console.print(f"[yellow]⚠ Cache de tradução detectado![/yellow] Tradução anterior parou no bloco [bold cyan]{cached_idx + 1}/{cached_total}[/bold cyan].")
            if click.confirm("Deseja retomar a tradução a partir deste ponto?", default=True):
                resume = True
                console.print("[bold green]✔ Retomando tradução a partir do cache.[/bold green]\n")
            else:
                utils.clear_translation_cache(input)
                console.print("[bold yellow]⚠ Iniciando do zero (cache descartado).[/bold yellow]\n")
        else:
            utils.clear_translation_cache(input)
            
    # Inicializar estado inicial do LangGraph
    if resume:
        glossary_to_use = cache.get("glossary", {})
        if custom_vocab:
            # Mescla novos termos fornecidos no vocabulário atual
            for k, v in custom_vocab.items():
                glossary_to_use[k] = v
                
        initial_state = {
            "source_text": document_text,
            "source_lang": source if source != "Autodetectar" else "",
            "target_lang": target,
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
        glossary_printed = True
        final_translation_chunks = cache.get("final_chunks", [])
    else:
        initial_state = {
            "source_text": document_text,
            "source_lang": source if source != "Autodetectar" else "",
            "target_lang": target,
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
        glossary_printed = False
        final_translation_chunks = []        
    # Compilar o grafo
    graph = compile_translation_graph()
    
    # Executar grafo com streaming de passos e atualizar interface
    console.print("[bold yellow]Executando fluxo do agente...[/bold yellow]\n")
    
    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    try:
        # Percorre o gerador de streaming do LangGraph
        for event in graph.stream(initial_state):
            for node_name, state_update in event.items():
                # Acumular tokens totais para o sumário final
                if "total_prompt_tokens" in state_update:
                    total_prompt_tokens = state_update["total_prompt_tokens"]
                if "total_completion_tokens" in state_update:
                    total_completion_tokens = state_update["total_completion_tokens"]
                
                # Passo A: Extração de Glossário
                if node_name == "extract_glossary":
                    glossary = state_update.get("glossary", {})
                    tokens = state_update.get("last_node_tokens", {})
                    
                    if glossary and not glossary_printed:
                        table = Table(title="[bold green]Glossário de Termos Extraídos/Carregados[/bold green]", show_header=True, header_style="bold cyan")
                        table.add_column("Termo Original", style="yellow")
                        table.add_column("Tradução Recomendada", style="green")
                        for k, v in glossary.items():
                            table.add_row(k, v)
                        console.print(table)
                        console.print()
                        glossary_printed = True
                        
                    if tokens and (tokens.get("prompt", 0) > 0 or tokens.get("completion", 0) > 0):
                        console.print(f"[dim]⚡ Tokens do Glossário: Entrada: {tokens.get('prompt', 0)} | Saída: {tokens.get('completion', 0)}[/dim]\n")
                    elif not glossary:
                        console.print("[dim]Nenhum jargão específico extraído para o glossário. Continuando fluxo normal...[/dim]\n")
                
                # Passo B: Tradução do Bloco
                elif node_name == "translate_chunk":
                    idx = state_update.get("current_chunk_idx", current_chunk_idx)
                    latest_raw = state_update.get("current_translation", "")
                    tokens = state_update.get("last_node_tokens", {})
                    
                    token_str = f"Tokens: In: {tokens.get('prompt', 0)} | Out: {tokens.get('completion', 0)}" if tokens else ""
                    console.print(Panel(
                        latest_raw,
                        title=f"[bold yellow]Bloco {idx + 1}/{num_chunks} - Tradução Proposta (Inicial)[/bold yellow]",
                        subtitle=f"[dim]{token_str}[/dim]" if token_str else None,
                        border_style="yellow",
                        padding=(1, 2)
                    ))
                
                # Passo C: Revisão/Reflexão
                elif node_name == "reflect_chunk":
                    idx = state_update.get("current_chunk_idx", current_chunk_idx)
                    feedback = state_update.get("reflection_feedback", "")
                    ref_count = state_update.get("refinement_count", 0)
                    tokens = state_update.get("last_node_tokens", {})
                    
                    token_str = f" (Tokens: In: {tokens.get('prompt', 0)} | Out: {tokens.get('completion', 0)})" if tokens else ""
                    if "APROVADO" in feedback.upper() and len(feedback.strip()) < 15:
                        if ref_count == 0:
                            console.print(f"[bold green]✔ Bloco {idx + 1} revisado e aprovado diretamente pelo revisor!{token_str}[/bold green]\n")
                        else:
                            console.print(f"[bold green]✔ Bloco {idx + 1} revisado e aprovado pelo revisor após {ref_count} refinamento(s)!{token_str}[/bold green]\n")
                    else:
                        title_suffix = f" (Revisão #{ref_count + 1})"
                        console.print(Panel(
                            f"[yellow]{feedback}[/yellow]",
                            title=f"[bold red]⚠ Bloco {idx + 1}/{num_chunks} - Crítica do Revisor{title_suffix}[/bold red]",
                            subtitle=f"[dim]Tokens: In: {tokens.get('prompt', 0)} | Out: {tokens.get('completion', 0)}[/dim]" if tokens else None,
                            border_style="red",
                            padding=(1, 2)
                        ))
                
                # Passo D: Refinamento
                elif node_name == "refine_chunk":
                    idx = state_update.get("current_chunk_idx", current_chunk_idx)
                    refined = state_update.get("current_translation", "")
                    ref_count = state_update.get("refinement_count", 0)
                    tokens = state_update.get("last_node_tokens", {})
                    
                    token_str = f"Tokens: In: {tokens.get('prompt', 0)} | Out: {tokens.get('completion', 0)}" if tokens else ""
                    console.print(Panel(
                        refined,
                        title=f"[bold green]Bloco {idx + 1}/{num_chunks} - Refinamento (Tentativa #{ref_count})[/bold green]",
                        subtitle=f"[dim]{token_str}[/dim]" if token_str else None,
                        border_style="blue",
                        padding=(1, 2)
                    ))

                # Passo E: Finalização do Bloco
                elif node_name == "finalize_chunk":
                    idx = state_update.get("current_chunk_idx", current_chunk_idx)
                    final_chunks = state_update.get("final_chunks", [])
                    translated_chunks = state_update.get("translated_chunks", [])
                    
                    if final_chunks:
                        final_translation_chunks = final_chunks
                        
                    # Salvar cache após finalizar o bloco (inclui tokens acumulados)
                    utils.save_translation_cache(
                        file_path=input,
                        target_lang=target,
                        source_lang=source,
                        current_chunk_idx=idx + 1,
                        translated_chunks=translated_chunks,
                        final_chunks=final_translation_chunks,
                        glossary=glossary,
                        chunks=chunks
                    )
                    
                    # Salvar arquivo de tradução parcial se a opção estiver ativa
                    if partial:
                        try:
                            partial_content = "\n\n".join(final_translation_chunks)
                            utils.write_document(partial_output, partial_content)
                        except Exception as e:
                            console.print(f"[bold red]⚠ Alerta: Falha ao gravar tradução parcial:[/bold red] {e}\n")
                            
                    console.print(f"[bold green]✔ Bloco {idx + 1} finalizado.[/bold green]\n")
                
                # Passo F: Atualização do índice para o próximo bloco
                elif node_name == "increment_index":
                    current_chunk_idx = state_update.get("current_chunk_idx", current_chunk_idx)
                    
    except Exception as e:
        console.print(f"\n[bold red]✖ Ocorreu um erro durante a execução do agente:[/bold red] {e}")
        return
        
    # 3. Consolidar e salvar arquivo traduzido
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if final_translation_chunks:
        # Usa o delimitador com base em parágrafo (\n\n) para a montagem final
        final_content = "\n\n".join(final_translation_chunks)
        try:
            utils.write_document(output, final_content)
            # Limpar o cache de tradução após o sucesso total
            utils.clear_translation_cache(input)
            # Limpar arquivo de tradução parcial, pois a tradução final completa foi salva
            if partial and os.path.exists(partial_output):
                try:
                    os.remove(partial_output)
                except Exception:
                    pass
            console.print(Panel(
                f"[bold green]✔ Tradução concluída com sucesso![/bold green]\n\n"
                f"[bold cyan]Arquivo salvo em    :[/bold cyan] {output}\n"
                f"[bold cyan]Blocos traduzidos   :[/bold cyan] {len(final_translation_chunks)}/{num_chunks}\n"
                f"[bold cyan]Contagem de Tokens  :[/bold cyan] Entrada: {total_prompt_tokens} | Saída: {total_completion_tokens} (Total: {total_prompt_tokens + total_completion_tokens})\n"
                f"[bold cyan]Tempo total de exec :[/bold cyan] {elapsed_time:.2f} segundos",
                border_style="green",
                padding=(1, 2)
            ))
        except Exception as e:
            console.print(f"[bold red]✖ Erro ao salvar o documento traduzido:[/bold red] {e}")
    else:
        console.print("[bold red]✖ Não foi possível consolidar a tradução final (nenhum bloco processado).[/bold red]")

if __name__ == '__main__':
    translate_cli()
