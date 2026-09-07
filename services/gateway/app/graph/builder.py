"""
Graph assembly.

    START
      |
      +--------------+--------------+---------------+
      v              v              v                (all three run before
   [router]   [sentiment_monitor] [input_guardrail]   any downstream node)
      |              |              |
      +------+-------+------+-------+
             v
      (conditional edge on router.intent -- see router.py::route_edge)
      |        |           |           |
      v        v           v           v
 order_lookup reschedule address_change fallback
      |        |           |           |
      +--------+-----+-----+-----------+
                     v
                 [composer]
                     |
                     v
             [output_guardrail]
                     |
                    END

router, sentiment_monitor, and input_guardrail are independent LangGraph
nodes all fed from START, so the graph runtime executes them concurrently --
sentiment and the safety check genuinely cannot block the reply path because
neither has an edge to the domain agents; both only feed composer.

Note what input_guardrail's verdict does NOT do: override route_edge.
route_edge is evaluated off intent_router's own completion and, empirically,
cannot reliably see a sibling branch's same-superstep write -- checking it
there would be silently racy, not a real gate. The actual enforcement is
nodes/input_guardrail.py::guardrail_block_patch(), called at the top of
order_lookup/reschedule/address_change/fallback: by the time any of those
runs it's a later superstep, so the merged state (including the verdict) is
guaranteed present as ordinary node input, the same guarantee composer relies
on to see both its concurrent predecessors.

output_guardrail is a plain linear node after composer, not a concurrent
branch -- there's nothing to fan in there, just a last screening pass on the
reply before END. See CLAUDE.md rule 4 before adding a fourth concurrent
branch off START: it needs its own state key, same as this one did.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.checkpointer import get_checkpointer
from app.graph.nodes.address_change import address_change
from app.graph.nodes.composer import compose
from app.graph.nodes.fallback import fallback
from app.graph.nodes.input_guardrail import input_guardrail
from app.graph.nodes.order_lookup import order_lookup
from app.graph.nodes.output_guardrail import output_guardrail
from app.graph.nodes.reschedule import reschedule
from app.graph.nodes.router import route, route_edge
from app.graph.nodes.sentiment import monitor_sentiment
from app.graph.state import CallState


def build_graph() -> StateGraph:
    graph = StateGraph(CallState)

    graph.add_node("intent_router", route)
    graph.add_node("sentiment_monitor", monitor_sentiment)
    graph.add_node("input_guardrail", input_guardrail)
    graph.add_node("order_lookup", order_lookup)
    graph.add_node("reschedule", reschedule)
    graph.add_node("address_change", address_change)
    graph.add_node("fallback", fallback)
    graph.add_node("composer", compose)
    graph.add_node("output_guardrail", output_guardrail)

    # Fan out from START -- router, sentiment_monitor, and input_guardrail all
    # run before the graph proceeds, with no dependency between them.
    graph.add_edge(START, "intent_router")
    graph.add_edge(START, "sentiment_monitor")
    graph.add_edge(START, "input_guardrail")

    graph.add_conditional_edges(
        "intent_router",
        route_edge,
        {
            "order_lookup": "order_lookup",
            "reschedule": "reschedule",
            "address_change": "address_change",
            "fallback": "fallback",
        },
    )

    # Domain agents and sentiment_monitor all fan in to composer. LangGraph
    # waits for every incoming edge before running a node, so composer only
    # fires once both the chosen agent AND sentiment_monitor have finished.
    for node in ("order_lookup", "reschedule", "address_change", "fallback"):
        graph.add_edge(node, "composer")
    graph.add_edge("sentiment_monitor", "composer")
    graph.add_edge("input_guardrail", "composer")

    graph.add_edge("composer", "output_guardrail")
    graph.add_edge("output_guardrail", END)
    return graph


def compile_graph():
    """
    Compiled graph, checkpointer-backed.

    Callers should use this via a context manager (the checkpointer holds a
    sqlite3 connection) -- see app/graph/runner.py for the turn-execution
    wrapper that handles this correctly.
    """
    with get_checkpointer() as checkpointer:
        return build_graph().compile(checkpointer=checkpointer)
