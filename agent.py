from langgraph.graph import StateGraph, END
from state import TranslationState
import nodes

# Compilação do Grafo do LangGraph
def compile_translation_graph():
    workflow = StateGraph(TranslationState)
    
    # Registrar os nós no fluxo
    workflow.add_node("extract_glossary", nodes.extract_glossary)
    workflow.add_node("translate_chunk", nodes.translate_chunk)
    workflow.add_node("reflect_chunk", nodes.reflect_chunk)
    workflow.add_node("refine_chunk", nodes.refine_chunk)
    workflow.add_node("finalize_chunk", nodes.finalize_chunk)
    workflow.add_node("increment_index", nodes.increment_index)
    
    # Configurar pontos de entrada e transições
    workflow.set_entry_point("extract_glossary")
    
    workflow.add_edge("extract_glossary", "translate_chunk")
    workflow.add_edge("translate_chunk", "reflect_chunk")
    
    # Transição após reflexão
    workflow.add_conditional_edges(
        "reflect_chunk",
        nodes.route_after_reflect,
        {
            "finalize": "finalize_chunk",
            "refine": "refine_chunk"
        }
    )
    
    # Refinamento volta para reflexão
    workflow.add_edge("refine_chunk", "reflect_chunk")
    
    # Transição após finalização
    workflow.add_conditional_edges(
        "finalize_chunk",
        nodes.route_after_finalize,
        {
            "next": "increment_index",
            "end": END
        }
    )
    
    workflow.add_edge("increment_index", "translate_chunk")
    
    return workflow.compile()
