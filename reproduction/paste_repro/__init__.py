"""PASTE reproduction components for trace analysis and live tool–LLM runs."""

from .analysis import evaluate_held_out
from .invocation import Invocation, canonicalize_arguments
from .live_broker import LiveAuthoritativeResult, LiveBrokerStats, LiveToolBroker
from .live_executor import (
    PublicWebToolExecutor,
    SyncToolMapExecutor,
    WikipediaLiveExecutor,
)
from .contextual_mapper import (
    CONTEXTUAL_POLICY_VERSION,
    ContextualURLReranker,
    load_contextual_artifact,
    save_contextual_artifact,
)
from .mapper import URLRankMapper, load_artifact, save_artifact, write_json_atomic
from .scheduler import AuthoritativeResult, SchedulerStats, SpeculativeScheduler
from .tool_prediction import (
    ContextualTraceVisitPredictor,
    TRACE_LEARNED_VISIT_POLICY_VERSION,
    TraceLearnedVisitPredictor,
    VisitPredictor,
    load_visit_predictor,
    structured_search_results,
)
from .traces import (
    SearchResult,
    SearchVisitTransition,
    SessionTrace,
    TraceFormatError,
    extract_search_visit_transitions,
    load_sessions,
    load_trace,
    split_sessions,
    transitions_from_sessions,
)

__all__ = [
    "AuthoritativeResult",
    "CONTEXTUAL_POLICY_VERSION",
    "ContextualTraceVisitPredictor",
    "ContextualURLReranker",
    "Invocation",
    "LiveAuthoritativeResult",
    "LiveBrokerStats",
    "LiveToolBroker",
    "PublicWebToolExecutor",
    "SchedulerStats",
    "SearchResult",
    "SearchVisitTransition",
    "SessionTrace",
    "SpeculativeScheduler",
    "SyncToolMapExecutor",
    "TraceFormatError",
    "TraceLearnedVisitPredictor",
    "VisitPredictor",
    "TRACE_LEARNED_VISIT_POLICY_VERSION",
    "URLRankMapper",
    "WikipediaLiveExecutor",
    "canonicalize_arguments",
    "evaluate_held_out",
    "extract_search_visit_transitions",
    "load_artifact",
    "load_contextual_artifact",
    "load_sessions",
    "load_trace",
    "load_visit_predictor",
    "save_artifact",
    "save_contextual_artifact",
    "split_sessions",
    "structured_search_results",
    "transitions_from_sessions",
    "write_json_atomic",
]

__version__ = "0.1.0"
