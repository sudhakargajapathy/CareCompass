"""LangGraph Orchestrator for coordinating the multi-agent healthcare provider matching workflow."""

import logging
from typing import Dict, List, Any, Optional, TypedDict, Callable
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import asyncio
from datetime import datetime

from .data_gatherer import DataGathererAgent
from .preference_scorer import PreferenceScorerAgent
from .critic_validator import CriticValidatorAgent
from utils.vector_store import get_vector_store
from utils.config import get_config

logger = logging.getLogger(__name__)


class ProgressUpdate(TypedDict):
    """Real-time progress update structure."""
    step_name: str           # "gather_data", "score_providers", etc.
    agent_name: str          # "DataGathererAgent", "PreferenceScorerAgent", etc.
    status: str              # "started", "in_progress", "completed", "failed"
    action: str              # "Found 7 providers in Phoenix, AZ"
    metrics: Dict[str, Any]  # {"providers_found": 7, "location": "Phoenix, AZ"}
    progress_percentage: int # 0-100
    timestamp: str


ProgressCallback = Callable[[ProgressUpdate], None]


class WorkflowState(TypedDict):
    """State structure for the provider matching workflow."""
    # Input parameters
    specialty: str
    location: str
    insurance: Optional[str]
    preferences: Dict[str, Any]

    # Agent outputs
    gathered_data: Dict[str, Any]
    scored_providers: Dict[str, Any]
    validation_results: Dict[str, Any]

    # Workflow metadata
    current_step: str
    workflow_id: str
    error_messages: List[str]
    execution_log: List[Dict[str, Any]]

    # Final output
    final_recommendations: List[Dict[str, Any]]
    workflow_summary: Dict[str, Any]


class ProviderMatchingOrchestrator:
    """LangGraph orchestrator for coordinating all agents in the provider matching workflow."""

    def __init__(self, progress_callback: Optional[ProgressCallback] = None):
        """Initialize the orchestrator with all agents and workflow components."""
        self.config = get_config()
        self.progress_callback = progress_callback

        # Initialize agents
        self.data_gatherer = DataGathererAgent()
        self.preference_scorer = PreferenceScorerAgent()
        self.critic_validator = CriticValidatorAgent()
        self.vector_store = get_vector_store()

        # Build workflow graph
        self.workflow = None
        self._build_workflow()

        logger.info("Provider matching orchestrator initialized")

    def _build_workflow(self) -> None:
        """Build the LangGraph workflow for provider matching."""
        try:
            # Create state graph
            workflow = StateGraph(WorkflowState)

            # Add workflow nodes
            workflow.add_node("initialize", self._initialize_workflow)
            workflow.add_node("gather_data", self._gather_provider_data)
            workflow.add_node("score_providers", self._score_providers)
            workflow.add_node("validate_rankings", self._validate_rankings)
            workflow.add_node("finalize_results", self._finalize_results)
            workflow.add_node("handle_error", self._handle_error)

            # Set entry point
            workflow.set_entry_point("initialize")

            # Add edges (workflow flow)
            workflow.add_edge("initialize", "gather_data")
            workflow.add_conditional_edges(
                "gather_data",
                self._check_data_gathering_success,
                {
                    "success": "score_providers",
                    "error": "handle_error"
                }
            )
            workflow.add_conditional_edges(
                "score_providers",
                self._check_scoring_success,
                {
                    "success": "validate_rankings",
                    "error": "handle_error"
                }
            )
            workflow.add_conditional_edges(
                "validate_rankings",
                self._check_validation_success,
                {
                    "success": "finalize_results",
                    "error": "handle_error"
                }
            )
            workflow.add_edge("finalize_results", END)
            workflow.add_edge("handle_error", END)

            # Compile workflow
            self.workflow = workflow.compile()
            logger.info("LangGraph workflow compiled successfully")

        except Exception as e:
            logger.error(f"Failed to build workflow: {e}")
            raise

    def _log_step(self, state: WorkflowState, step: str, status: str, details: Dict[str, Any]) -> None:
        """Log workflow step execution."""
        log_entry = {
            "step": step,
            "status": status,
            "timestamp": "now",  # In production, use actual timestamp
            "details": details
        }
        state["execution_log"].append(log_entry)
        logger.info(f"Workflow step completed: {step} - {status}")

    def _emit_progress(
        self,
        step_name: str,
        agent_name: str,
        status: str,
        action: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Emit progress update to registered callback."""
        if not self.progress_callback:
            return

        # Progress mapping: 5 steps total (initialize, gather, score, validate, finalize)
        step_progress_map = {
            "initialize": {"started": 5, "completed": 10},
            "gather_data": {"started": 10, "in_progress": 25, "completed": 35},
            "score_providers": {"started": 35, "in_progress": 50, "completed": 60},
            "validate_rankings": {"started": 60, "in_progress": 75, "completed": 85},
            "finalize_results": {"started": 85, "completed": 100}
        }

        progress_pct = step_progress_map.get(step_name, {}).get(status, 0)

        update = ProgressUpdate(
            step_name=step_name,
            agent_name=agent_name,
            status=status,
            action=action,
            metrics=metrics,
            progress_percentage=progress_pct,
            timestamp=datetime.now().isoformat()
        )

        try:
            self.progress_callback(update)
        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")
            # Don't let callback failures break workflow

    def _initialize_workflow(self, state: WorkflowState) -> WorkflowState:
        """Initialize the workflow with input validation and setup."""
        try:
            self._log_step(state, "initialize", "started", {"input_validation": "in_progress"})

            # Validate required inputs
            if not state.get("specialty") or not state.get("location"):
                raise ValueError("Specialty and location are required")

            # Set default preferences if not provided
            if not state.get("preferences"):
                state["preferences"] = {
                    "location_weight": 0.4,
                    "rating_weight": 0.3,
                    "insurance_priority": 0.3
                }

            # Initialize workflow metadata
            state["current_step"] = "initialize"
            state["error_messages"] = []
            state["execution_log"] = []

            # Generate workflow ID
            import uuid
            state["workflow_id"] = str(uuid.uuid4())[:8]

            self._log_step(state, "initialize", "completed", {
                "workflow_id": state["workflow_id"],
                "specialty": state["specialty"],
                "location": state["location"]
            })

            return state

        except Exception as e:
            state["error_messages"].append(f"Initialization failed: {str(e)}")
            self._log_step(state, "initialize", "failed", {"error": str(e)})
            return state

    def _gather_provider_data(self, state: WorkflowState) -> WorkflowState:
        """Execute data gathering using the DataGathererAgent."""
        try:
            state["current_step"] = "gather_data"
            self._log_step(state, "gather_data", "started", {
                "agent": "DataGathererAgent",
                "search_params": {
                    "specialty": state["specialty"],
                    "location": state["location"],
                    "insurance": state.get("insurance")
                }
            })

            # Emit progress: started
            self._emit_progress(
                step_name="gather_data",
                agent_name="DataGathererAgent",
                status="started",
                action=f"Searching for {state['specialty']} providers in {state['location']}",
                metrics={"specialty": state["specialty"], "location": state["location"]}
            )

            # Gather provider data
            gathered_data = self.data_gatherer.gather_providers(
                specialty=state["specialty"],
                location=state["location"],
                insurance=state.get("insurance")
            )

            state["gathered_data"] = gathered_data

            # Emit progress: in progress (with FHIR details)
            provider_count = len(gathered_data.get("providers", []))
            search_meta = gathered_data.get("search_metadata", {})
            fhir_count = search_meta.get("fhir_count", 0)
            tavily_count = search_meta.get("tavily_count", 0)

            fhir_note = ""
            if fhir_count > 0:
                fhir_note = f" (verified {fhir_count} via FHIR)"

            self._emit_progress(
                step_name="gather_data",
                agent_name="DataGathererAgent",
                status="in_progress",
                action=f"Found {provider_count} providers in {state['location']}{fhir_note}",
                metrics={
                    "providers_found": provider_count,
                    "fhir_count": fhir_count,
                    "tavily_count": tavily_count,
                }
            )

            # Store providers in vector store for future searches
            if gathered_data.get("providers"):
                success = self.vector_store.add_providers(gathered_data["providers"])
                logger.info(f"Vector store update: {'success' if success else 'failed'}")

            if gathered_data.get("status") != "success":
                state["error_messages"].append(f"Data gathering failed with status: {gathered_data.get('status')}")

            self._log_step(state, "gather_data", "completed", {
                "providers_found": provider_count,
                "fhir_verified": fhir_count,
                "tavily_found": tavily_count,
                "data_status": gathered_data.get("status"),
                "vector_store_updated": True
            })

            # Emit progress: completed
            self._emit_progress(
                step_name="gather_data",
                agent_name="DataGathererAgent",
                status="completed",
                action=f"Extracted {provider_count} providers successfully{fhir_note}",
                metrics={
                    "providers_found": provider_count,
                    "fhir_count": fhir_count,
                    "tavily_count": tavily_count,
                    "status": gathered_data.get("status")
                }
            )

            return state

        except Exception as e:
            state["error_messages"].append(f"Data gathering failed: {str(e)}")
            self._log_step(state, "gather_data", "failed", {"error": str(e)})
            return state

    def _score_providers(self, state: WorkflowState) -> WorkflowState:
        """Execute provider scoring using the PreferenceScorerAgent."""
        try:
            state["current_step"] = "score_providers"
            self._log_step(state, "score_providers", "started", {
                "agent": "PreferenceScorerAgent",
                "providers_to_score": len(state["gathered_data"].get("providers", []))
            })

            # Emit progress: started
            self._emit_progress(
                step_name="score_providers",
                agent_name="PreferenceScorerAgent",
                status="started",
                action=f"Analyzing {len(state['gathered_data'].get('providers', []))} providers",
                metrics={"providers_to_score": len(state["gathered_data"].get("providers", []))}
            )

            # Score providers
            scored_results = self.preference_scorer.score_providers(
                providers=state["gathered_data"]["providers"],
                preferences=state["preferences"]
            )

            state["scored_providers"] = scored_results

            self._log_step(state, "score_providers", "completed", {
                "ranking_status": scored_results.get("status"),
                "providers_ranked": len(scored_results.get("ranked_providers", [])),
                "top_provider": scored_results.get("scoring_metadata", {}).get("top_provider")
            })

            # Emit progress: completed
            self._emit_progress(
                step_name="score_providers",
                agent_name="PreferenceScorerAgent",
                status="completed",
                action=f"Ranked {len(scored_results.get('ranked_providers', []))} providers",
                metrics={
                    "providers_ranked": len(scored_results.get("ranked_providers", [])),
                    "top_provider": scored_results.get("scoring_metadata", {}).get("top_provider", "Unknown")
                }
            )

            return state

        except Exception as e:
            state["error_messages"].append(f"Provider scoring failed: {str(e)}")
            self._log_step(state, "score_providers", "failed", {"error": str(e)})
            return state

    def _validate_rankings(self, state: WorkflowState) -> WorkflowState:
        """Execute ranking validation using the CriticValidatorAgent."""
        try:
            state["current_step"] = "validate_rankings"
            self._log_step(state, "validate_rankings", "started", {
                "agent": "CriticValidatorAgent",
                "providers_to_validate": len(state["scored_providers"].get("ranked_providers", []))
            })

            # Emit progress: started
            self._emit_progress(
                step_name="validate_rankings",
                agent_name="CriticValidatorAgent",
                status="started",
                action="Validating provider rankings for bias",
                metrics={"providers_to_validate": len(state["scored_providers"].get("ranked_providers", []))}
            )

            # Validate rankings
            validation_results = self.critic_validator.validate_rankings(
                ranked_providers=state["scored_providers"]["ranked_providers"],
                preferences=state["preferences"]
            )

            state["validation_results"] = validation_results

            self._log_step(state, "validate_rankings", "completed", {
                "validation_status": validation_results.get("status"),
                "bias_severity": validation_results.get("validation_metadata", {}).get("bias_severity"),
                "ranking_confidence": validation_results.get("validation_metadata", {}).get("ranking_confidence")
            })

            # Emit progress: completed
            confidence = validation_results.get("validation_metadata", {}).get("ranking_confidence", "medium")
            self._emit_progress(
                step_name="validate_rankings",
                agent_name="CriticValidatorAgent",
                status="completed",
                action=f"Validation complete with {confidence} confidence",
                metrics={
                    "confidence": confidence,
                    "bias_severity": validation_results.get("validation_metadata", {}).get("bias_severity", "unknown")
                }
            )

            return state

        except Exception as e:
            state["error_messages"].append(f"Ranking validation failed: {str(e)}")
            self._log_step(state, "validate_rankings", "failed", {"error": str(e)})
            return state

    def _finalize_results(self, state: WorkflowState) -> WorkflowState:
        """Finalize the workflow results and prepare final recommendations."""
        try:
            state["current_step"] = "finalize_results"
            self._log_step(state, "finalize_results", "started", {
                "consolidating_results": True
            })

            # Extract final recommendations
            ranked_providers = state["scored_providers"].get("ranked_providers", [])
            validation_results = state["validation_results"].get("validation_results", {})

            # Prepare final recommendations
            final_recommendations = []
            for i, provider in enumerate(ranked_providers[:5]):  # Top 5 recommendations
                recommendation = {
                    "rank": i + 1,
                    "provider": provider,
                    "recommendation_confidence": "high",  # Could be adjusted based on validation
                    "ai_reasoning": provider.get("ai_reasoning", ""),
                    "validation_notes": ""
                }

                # Add validation insights
                top_validations = validation_results.get("top_provider_validation", {}).get("top_provider_validations", [])
                for val in top_validations:
                    if val.get("rank") == i + 1:
                        recommendation["validation_notes"] = val.get("validation_notes", "")
                        recommendation["recommendation_confidence"] = val.get("confidence_in_recommendation", "medium")
                        break

                final_recommendations.append(recommendation)

            state["final_recommendations"] = final_recommendations

            # Create workflow summary
            workflow_summary = {
                "workflow_id": state["workflow_id"],
                "total_providers_found": len(state["gathered_data"].get("providers", [])),
                "final_recommendations_count": len(final_recommendations),
                "data_gathering_status": state["gathered_data"].get("status"),
                "scoring_status": state["scored_providers"].get("status"),
                "validation_status": state["validation_results"].get("status"),
                "overall_confidence": self._calculate_overall_confidence(state),
                "execution_steps": len(state["execution_log"]),
                "errors_encountered": len(state["error_messages"])
            }

            state["workflow_summary"] = workflow_summary

            self._log_step(state, "finalize_results", "completed", workflow_summary)

            return state

        except Exception as e:
            state["error_messages"].append(f"Result finalization failed: {str(e)}")
            self._log_step(state, "finalize_results", "failed", {"error": str(e)})
            return state

    def _handle_error(self, state: WorkflowState) -> WorkflowState:
        """Handle workflow errors and provide meaningful error responses."""
        try:
            self._log_step(state, "handle_error", "started", {
                "error_count": len(state["error_messages"])
            })

            # Create error summary
            error_summary = {
                "workflow_failed": True,
                "error_messages": state["error_messages"],
                "failed_step": state.get("current_step", "unknown"),
                "partial_results": {
                    "data_gathered": bool(state.get("gathered_data")),
                    "providers_scored": bool(state.get("scored_providers")),
                    "validation_completed": bool(state.get("validation_results"))
                }
            }

            state["workflow_summary"] = error_summary

            self._log_step(state, "handle_error", "completed", error_summary)

            return state

        except Exception as e:
            logger.error(f"Error handling failed: {e}")
            return state

    def _calculate_overall_confidence(self, state: WorkflowState) -> str:
        """Calculate overall confidence in the workflow results."""
        try:
            confidence_factors = []

            # Data gathering confidence
            if state["gathered_data"].get("status") == "success":
                provider_count = len(state["gathered_data"].get("providers", []))
                if provider_count >= 5:
                    confidence_factors.append("high")
                elif provider_count >= 2:
                    confidence_factors.append("medium")
                else:
                    confidence_factors.append("low")
            else:
                confidence_factors.append("low")

            # Scoring confidence
            if state["scored_providers"].get("status") == "success":
                confidence_factors.append("high")
            else:
                confidence_factors.append("low")

            # Validation confidence
            validation_confidence = state["validation_results"].get("validation_metadata", {}).get("ranking_confidence", "medium")
            confidence_factors.append(validation_confidence)

            # Calculate overall
            high_count = confidence_factors.count("high")
            medium_count = confidence_factors.count("medium")
            low_count = confidence_factors.count("low")

            if high_count >= 2 and low_count == 0:
                return "high"
            elif low_count >= 2:
                return "low"
            else:
                return "medium"

        except Exception:
            return "low"

    # Conditional edge functions
    def _check_data_gathering_success(self, state: WorkflowState) -> str:
        """Check if data gathering was successful."""
        if state.get("gathered_data", {}).get("status") == "success":
            return "success"
        return "error"

    def _check_scoring_success(self, state: WorkflowState) -> str:
        """Check if provider scoring was successful."""
        if state.get("scored_providers", {}).get("status") == "success":
            return "success"
        return "error"

    def _check_validation_success(self, state: WorkflowState) -> str:
        """Check if ranking validation was successful."""
        if state.get("validation_results", {}).get("status") == "success":
            return "success"
        return "error"

    def execute_workflow(self, specialty: str, location: str, insurance: Optional[str] = None, preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the complete provider matching workflow.

        Args:
            specialty: Medical specialty to search for
            location: Location to search in
            insurance: Insurance type filter (optional)
            preferences: User preference weights (optional)

        Returns:
            Dictionary containing workflow results and recommendations
        """
        try:
            logger.info(f"Starting provider matching workflow: {specialty} in {location}")

            # Create initial state
            initial_state = WorkflowState(
                specialty=specialty,
                location=location,
                insurance=insurance,
                preferences=preferences or {},
                gathered_data={},
                scored_providers={},
                validation_results={},
                current_step="",
                workflow_id="",
                error_messages=[],
                execution_log=[],
                final_recommendations=[],
                workflow_summary={}
            )

            # Execute workflow
            result = self.workflow.invoke(initial_state)

            logger.info(f"Workflow completed with status: {result.get('workflow_summary', {}).get('overall_confidence', 'unknown')}")

            # Extract alternative perspectives from critic validator results
            validation_results = result.get("validation_results", {})
            alternative_perspectives = validation_results.get("validation_results", {}).get("alternative_rankings", [])

            return {
                "success": len(result.get("error_messages", [])) == 0,
                "final_recommendations": result.get("final_recommendations", []),
                "workflow_summary": result.get("workflow_summary", {}),
                "execution_log": result.get("execution_log", []),
                "agent_outputs": {
                    "data_gatherer": result.get("gathered_data", {}),
                    "preference_scorer": result.get("scored_providers", {}),
                    "critic_validator": validation_results
                },
                "alternative_perspectives": alternative_perspectives,
                "error_messages": result.get("error_messages", [])
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "final_recommendations": [],
                "workflow_summary": {"workflow_failed": True, "error": str(e)},
                "execution_log": [],
                "agent_outputs": {},
                "error_messages": [str(e)]
            }

    def execute_workflow_streaming(
        self,
        specialty: str,
        location: str,
        insurance: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None
    ) -> Dict[str, Any]:
        """Execute workflow with streaming progress updates.

        Uses workflow.stream() instead of invoke() to enable real-time progress tracking.

        Args:
            specialty: Medical specialty to search for
            location: Location to search in
            insurance: Insurance type filter (optional)
            preferences: User preference weights (optional)
            progress_callback: Callback function for receiving progress updates

        Returns:
            Dictionary containing workflow results and recommendations (same format as execute_workflow)
        """
        self.progress_callback = progress_callback

        try:
            logger.info(f"Starting streaming workflow: {specialty} in {location}")

            # Create initial state (same as execute_workflow)
            initial_state = WorkflowState(
                specialty=specialty,
                location=location,
                insurance=insurance,
                preferences=preferences or {},
                gathered_data={},
                scored_providers={},
                validation_results={},
                current_step="",
                workflow_id="",
                error_messages=[],
                execution_log=[],
                final_recommendations=[],
                workflow_summary={}
            )

            # Execute workflow with streaming
            final_state = None
            for state_chunk in self.workflow.stream(initial_state):
                for step_name, step_state in state_chunk.items():
                    if step_name not in ["__start__", "__end__"]:
                        final_state = step_state

            if final_state is None:
                raise RuntimeError("Workflow stream did not produce final state")

            # Extract results (same format as execute_workflow)
            result = final_state
            validation_results = result.get("validation_results", {})
            alternative_perspectives = validation_results.get("validation_results", {}).get("alternative_rankings", [])

            logger.info(f"Streaming workflow completed with status: {result.get('workflow_summary', {}).get('overall_confidence', 'unknown')}")

            return {
                "success": len(result.get("error_messages", [])) == 0,
                "final_recommendations": result.get("final_recommendations", []),
                "workflow_summary": result.get("workflow_summary", {}),
                "execution_log": result.get("execution_log", []),
                "agent_outputs": {
                    "data_gatherer": result.get("gathered_data", {}),
                    "preference_scorer": result.get("scored_providers", {}),
                    "critic_validator": validation_results
                },
                "alternative_perspectives": alternative_perspectives,
                "error_messages": result.get("error_messages", [])
            }

        except Exception as e:
            logger.error(f"Streaming workflow execution failed: {e}")
            return {
                "success": False,
                "final_recommendations": [],
                "workflow_summary": {"workflow_failed": True, "error": str(e)},
                "execution_log": [],
                "agent_outputs": {},
                "error_messages": [str(e)]
            }
        finally:
            self.progress_callback = None


def create_orchestrator() -> ProviderMatchingOrchestrator:
    """Factory function to create a ProviderMatchingOrchestrator instance.

    Returns:
        ProviderMatchingOrchestrator instance
    """
    return ProviderMatchingOrchestrator()