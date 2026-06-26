"""CareCompass - AI-Powered Healthcare Provider Matching System

A Streamlit application that uses multi-agent AI to find and rank healthcare providers
based on user preferences and requirements.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our agent system
from agents.orchestrator import create_orchestrator
from utils.config import get_config, check_environment
from utils.auth import get_authenticator
from utils.audit_log import log_audit_event
from utils.rate_limit import get_rate_limiter
from utils.security_headers import inject_security_headers
from utils.security import InputValidator


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "workflow_results" not in st.session_state:
        st.session_state.workflow_results = None
    if "search_executed" not in st.session_state:
        st.session_state.search_executed = False
    if "show_agent_logs" not in st.session_state:
        st.session_state.show_agent_logs = False
    if "last_search_params" not in st.session_state:
        st.session_state.last_search_params = None
    if "fhir_enabled" not in st.session_state:
        st.session_state.fhir_enabled = False


def check_api_keys() -> bool:
    """Check if all required API keys are configured."""
    is_valid, missing_keys = check_environment()
    if not is_valid:
        st.error("🚨 Missing API Keys")
        st.markdown("Please set up the following API keys in your `.env` file:")
        for key in missing_keys:
            st.markdown(f"- `{key.upper()}_API_KEY`")
        st.markdown("Copy `.env.example` to `.env` and add your API keys.")
        return False
    return True


def render_header():
    """Render the application header."""
    st.set_page_config(
        page_title="CareCompass",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_security_headers()

    st.title("🏥 CareCompass")
    st.markdown("*AI-Powered Healthcare Provider Matching*")

    # Add description
    with st.expander("About CareCompass"):
        st.markdown("""
        CareCompass uses a multi-agent AI system to intelligently match you with healthcare providers:

        **🔍 Data Gatherer Agent** - Searches for providers using Tavily API and extracts structured data with Claude Haiku
        **📊 Preference Scorer Agent** - Ranks providers using your preferences with GPT-4o-mini
        **🛡️ Critic Validator Agent** - Validates rankings and identifies potential issues with Claude Sonnet
        **🎯 LangGraph Orchestrator** - Coordinates all agents in an intelligent workflow

        The system provides transparent reasoning for every recommendation to help you make informed healthcare decisions.
        """)


def render_search_form() -> Optional[Dict[str, Any]]:
    """Render the provider search form."""
    st.header("🔍 Find Healthcare Providers")

    with st.form("provider_search"):
        col1, col2 = st.columns(2)

        with col1:
            specialty = st.selectbox(
                "Medical Specialty",
                [
                    "Neurology",
                    "Cardiology",
                    "Dermatology",
                    "Orthopedics",
                    "Gastroenterology",
                    "Endocrinology",
                    "Psychiatry",
                    "Oncology",
                    "Rheumatology",
                    "Pulmonology",
                    "Family Medicine",
                    "Internal Medicine"
                ],
                index=0
            )

            location = st.text_input(
                "Location",
                placeholder="e.g., Phoenix, AZ",
                value="Phoenix, AZ"
            )

        with col2:
            insurance = st.selectbox(
                "Insurance (Optional)",
                ["", "Aetna", "Blue Cross Blue Shield", "Cigna", "UnitedHealth", "Medicare", "Medicaid", "Other"],
                index=0
            )

        st.subheader("🎯 Preference Weights")

        col3, col4, col5 = st.columns(3)
        with col3:
            location_weight = st.slider("Location Importance", 0.0, 1.0, 0.4, 0.1)
        with col4:
            rating_weight = st.slider("Rating Importance", 0.0, 1.0, 0.3, 0.1)
        with col5:
            insurance_priority = st.slider("Insurance Importance", 0.0, 1.0, 0.3, 0.1)

        notes = st.text_area(
            "Additional Requirements (Optional)",
            placeholder="e.g., Need evening appointments, prefer female doctor, etc.",
            height=100,
            max_chars=500
        )

        submitted = st.form_submit_button("🚀 Find Providers", type="primary", use_container_width=True)

        if submitted:
            if not specialty or not location:
                st.error("Please provide both specialty and location.")
                return None

            # Validate inputs using security module
            validator = InputValidator()

            # Validate specialty
            safe_specialty = validator.sanitize_specialty(specialty)
            if not safe_specialty:
                st.error("Invalid specialty selected. Please choose from the dropdown list.")
                return None

            # Validate location
            safe_location = validator.sanitize_location(location)
            if not safe_location:
                st.error("Invalid location format. Please use: City, State (e.g., Phoenix, AZ)")
                return None

            # Validate insurance if provided
            safe_insurance = None
            if insurance:
                safe_insurance = validator.sanitize_insurance(insurance)
                # If insurance validation fails, just proceed without it
                if not safe_insurance:
                    st.warning("Insurance provider not recognized. Searching without insurance filter.")

            # Validate notes
            safe_notes = validator.sanitize_notes(notes)

            # Normalize weights to sum to 1.0
            total_weight = location_weight + rating_weight + insurance_priority
            if total_weight > 0:
                location_weight /= total_weight
                rating_weight /= total_weight
                insurance_priority /= total_weight

            return {
                "specialty": safe_specialty,
                "location": safe_location,
                "insurance": safe_insurance,
                "preferences": {
                    "location_weight": location_weight,
                    "rating_weight": rating_weight,
                    "insurance_priority": insurance_priority,
                    "notes": safe_notes
                }
            }

    return None


def search_params_changed(current_params: Dict[str, Any], last_params: Optional[Dict[str, Any]]) -> bool:
    """Check if search parameters have changed since last search.

    Args:
        current_params: Current form parameters
        last_params: Last executed search parameters

    Returns:
        True if parameters changed or this is first search
    """
    if last_params is None:
        return True  # First search

    # Compare core search fields
    if current_params["specialty"] != last_params.get("specialty"):
        return True
    if current_params["location"] != last_params.get("location"):
        return True
    if current_params["insurance"] != last_params.get("insurance"):
        return True

    # Compare preference weights
    current_prefs = current_params["preferences"]
    last_prefs = last_params.get("preferences", {})

    if current_prefs.get("location_weight") != last_prefs.get("location_weight"):
        return True
    if current_prefs.get("rating_weight") != last_prefs.get("rating_weight"):
        return True
    if current_prefs.get("insurance_priority") != last_prefs.get("insurance_priority"):
        return True
    if current_prefs.get("notes") != last_prefs.get("notes"):
        return True

    return False


def render_execution_timeline(execution_log: List[Dict[str, Any]]) -> None:
    """Render agent execution timeline from execution log.

    Args:
        execution_log: List of execution log entries from workflow
    """
    if not execution_log:
        st.info("No execution log available")
        return

    st.subheader("🤖 Agent Execution Timeline")

    # Group log entries by step
    step_groups = {}
    for entry in execution_log:
        step = entry.get("step", "unknown")
        if step not in step_groups:
            step_groups[step] = []
        step_groups[step].append(entry)

    # Step metadata
    step_info = {
        "initialize": {"icon": "🎯", "name": "Initialization", "agent": "Orchestrator"},
        "gather_data": {"icon": "🔍", "name": "Data Gathering", "agent": "DataGathererAgent"},
        "score_providers": {"icon": "📊", "name": "Preference Scoring", "agent": "PreferenceScorerAgent"},
        "validate_rankings": {"icon": "🛡️", "name": "Validation", "agent": "CriticValidatorAgent"},
        "finalize_results": {"icon": "✨", "name": "Finalizing", "agent": "Orchestrator"},
        "handle_error": {"icon": "❌", "name": "Error Handling", "agent": "Orchestrator"}
    }

    # Display each step
    for step, entries in step_groups.items():
        info = step_info.get(step, {"icon": "•", "name": step.title(), "agent": "Unknown"})

        # Find started and completed entries
        started_entry = next((e for e in entries if e.get("status") == "started"), None)
        completed_entry = next((e for e in entries if e.get("status") == "completed"), None)
        failed_entry = next((e for e in entries if e.get("status") == "failed"), None)

        with st.expander(f"{info['icon']} **{info['name']}** ({info['agent']})", expanded=False):
            # Status
            if failed_entry:
                st.error(f"❌ Failed: {failed_entry.get('details', {}).get('error', 'Unknown error')}")
            elif completed_entry:
                st.success("✅ Completed")
            elif started_entry:
                st.info("⏳ Started")

            # Display details from completed or started entry
            details_entry = completed_entry or started_entry
            if details_entry:
                details = details_entry.get("details", {})

                # Show key metrics
                if details:
                    metric_cols = st.columns(min(len(details), 4))
                    for i, (key, value) in enumerate(list(details.items())[:4]):
                        with metric_cols[i]:
                            metric_name = key.replace("_", " ").title()
                            st.metric(metric_name, value)

                    # Show remaining details in a simple list
                    if len(details) > 4:
                        st.markdown("**Additional Details:**")
                        for key, value in list(details.items())[4:]:
                            st.text(f"  • {key.replace('_', ' ').title()}: {value}")

            # Show all log entries in a simple list
            if len(entries) > 1:
                st.markdown("**All Events:**")
                for entry in entries:
                    st.caption(f"• {entry.get('status', 'unknown').title()}: {entry.get('timestamp', 'N/A')}")
                    if entry.get('details'):
                        st.json(entry['details'], expanded=False)


def render_provider_card(provider: Dict[str, Any], rank: int) -> None:
    """Render a provider information card."""
    with st.container():
        # Provider header
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(f"### {rank}. {provider.get('name', 'Unknown Provider')}")
            st.markdown(f"**{provider.get('specialty', 'N/A')}**")

            # FHIR badges and data source
            fhir_metadata = provider.get('fhir_metadata', {})
            data_source = provider.get('data_source', '')

            if fhir_metadata.get('network_verified'):
                st.markdown(":green[**Verified In-Network**]")
            if fhir_metadata.get('npi'):
                st.caption(f"NPI: {fhir_metadata['npi']}")
            if data_source:
                source_labels = {
                    'fhir': 'FHIR Directory',
                    'fhir+tavily': 'FHIR + Web',
                    'tavily': 'Web Search',
                }
                st.caption(f"Source: {source_labels.get(data_source, data_source)}")

        with col2:
            rating = provider.get('rating', 0)
            review_count = provider.get('review_count', None)
            rating_confidence = provider.get('rating_confidence', {})

            if rating > 0:
                rating_display = f"{rating}/5.0"
                # Add review count if available
                if review_count:
                    st.metric("Rating", rating_display, delta=f"{review_count} reviews")
                else:
                    st.metric("Rating", rating_display)

                # Show reliability badge
                reliability = rating_confidence.get('reliability', 'unknown')
                if reliability == 'high':
                    st.caption("🟢 High reliability")
                elif reliability == 'moderate':
                    st.caption("🟡 Moderate reliability")
                elif reliability == 'low':
                    st.caption("🟠 Low reliability")
            else:
                st.metric("Rating", "Unrated")
                st.caption("⚪ No reviews")

        with col3:
            final_score = provider.get('final_score', 0)
            st.metric("Match Score", f"{final_score:.1f}")

        # Provider details
        col4, col5 = st.columns(2)

        with col4:
            st.markdown("📍 **Location**")
            st.markdown(provider.get('location', 'N/A'))

            if provider.get('phone'):
                st.markdown("📞 **Phone**")
                st.markdown(provider.get('phone'))

            if provider.get('distance'):
                st.markdown("🚗 **Distance**")
                st.markdown(f"{provider.get('distance')} away")

        with col5:
            if provider.get('insurance_accepted'):
                st.markdown("💳 **Insurance Accepted**")
                insurance_list = provider.get('insurance_accepted', [])
                if isinstance(insurance_list, list):
                    st.markdown(", ".join(insurance_list[:3]))
                else:
                    st.markdown(str(insurance_list))

            if provider.get('years_experience'):
                st.markdown("👨‍⚕️ **Experience**")
                st.markdown(f"{provider.get('years_experience')} years")

        # Review Summary
        review_summary = provider.get('review_summary', '')
        review_sentiment = provider.get('review_sentiment', 'unknown')

        # Only show review summary if we have actual patient feedback (not generic statements)
        if (review_summary and
            review_summary != "No reviews available" and
            review_sentiment != "unknown" and
            len(review_summary.strip()) > 0):
            with st.expander("💬 Patient Reviews Summary", expanded=False):
                # Sentiment indicator
                sentiment_emoji = {
                    'positive': '😊',
                    'mixed': '😐',
                    'negative': '😞',
                    'unknown': '❓'
                }
                sentiment_color = {
                    'positive': 'green',
                    'mixed': 'orange',
                    'negative': 'red',
                    'unknown': 'gray'
                }

                emoji = sentiment_emoji.get(review_sentiment.lower(), '❓')
                color = sentiment_color.get(review_sentiment.lower(), 'gray')

                st.markdown(f"**Overall Sentiment:** :{color}[{emoji} {review_sentiment.title()}]")
                st.markdown(f"**Common Themes:**")
                st.markdown(f"> {review_summary}")

        # Data quality warnings
        data_warnings = provider.get('data_warnings', [])
        if data_warnings:
            st.warning("⚠️ **Data Quality Notes:**")
            for warning in data_warnings:
                st.caption(f"• {warning}")

        # AI Reasoning
        if provider.get('ai_reasoning'):
            with st.expander("🤖 AI Analysis"):
                st.markdown("**Why this provider was ranked here:**")
                st.markdown(provider.get('ai_reasoning'))

                if provider.get('ai_strengths'):
                    st.markdown("**Strengths:**")
                    for strength in provider.get('ai_strengths', []):
                        st.markdown(f"• {strength}")

                if provider.get('ai_concerns'):
                    st.markdown("**Considerations:**")
                    for concern in provider.get('ai_concerns', []):
                        st.markdown(f"• {concern}")

                confidence = provider.get('ai_confidence', 50)
                st.progress(confidence / 100)
                st.caption(f"AI Confidence: {confidence}%")

        st.divider()


def map_alternative_to_full_providers(alternative_ranking: Dict[str, Any], original_providers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map alternative ranking data back to full provider objects.

    Args:
        alternative_ranking: Alternative ranking scenario with reranked_providers
        original_providers: Original list of full provider objects

    Returns:
        List of providers with both original and alternative ranking data
    """
    reranked = alternative_ranking.get('reranked_providers', [])
    mapped_providers = []

    for alt_provider in reranked:
        # Find matching provider from original list
        provider_name = alt_provider.get('name')
        original = next(
            (p for p in original_providers if p.get('name') == provider_name),
            None
        )

        if original:
            # Combine original data with new ranking info
            mapped = original.copy()
            mapped['alt_rank'] = alt_provider.get('new_rank')
            mapped['alt_reasoning'] = alt_provider.get('reasoning')
            mapped['alt_score'] = alt_provider.get('new_score')
            mapped['original_rank'] = alt_provider.get('original_rank')
            mapped_providers.append(mapped)

    return mapped_providers


def render_alternative_provider_card(provider: Dict[str, Any]) -> None:
    """Render a compact provider card for alternative perspectives.

    Args:
        provider: Provider dictionary with alternative ranking data
    """
    with st.container():
        # Header with rank change indicator
        original_rank = provider.get('original_rank', provider.get('final_rank', 0))
        alt_rank = provider.get('alt_rank', provider.get('final_rank', 0))

        rank_change = original_rank - alt_rank
        if rank_change > 0:
            rank_indicator = f"⬆️ +{rank_change}"
            rank_color = "green"
        elif rank_change < 0:
            rank_indicator = f"⬇️ {rank_change}"
            rank_color = "red"
        else:
            rank_indicator = "➡️ No change"
            rank_color = "gray"

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {alt_rank}. {provider.get('name', 'Unknown Provider')}")
            st.markdown(f"**{provider.get('specialty', 'General')}**")
        with col2:
            st.markdown(f":{rank_color}[{rank_indicator}]")
            st.metric("Score", provider.get('alt_score', provider.get('final_score', 0)))

        # Key info
        col3, col4, col5 = st.columns(3)
        with col3:
            rating = provider.get('rating', 0)
            st.caption(f"⭐ {rating}/5.0")
        with col4:
            distance = provider.get('distance', 'N/A')
            st.caption(f"📍 {distance}")
        with col5:
            insurance = provider.get('insurance_accepted', [])
            ins_text = insurance[0] if insurance else 'N/A'
            st.caption(f"🏥 {ins_text}")

        # Alternative reasoning
        if provider.get('alt_reasoning'):
            with st.expander("Why this ranking?"):
                st.markdown(provider['alt_reasoning'])

        st.divider()


def render_alternative_perspectives(original_providers: List[Dict[str, Any]], alternative_rankings: List[Dict[str, Any]]) -> None:
    """Render alternative ranking perspectives with comparison.

    Args:
        original_providers: Original ranked provider list
        alternative_rankings: List of alternative ranking scenarios
    """
    st.header("🔄 Alternative Ranking Perspectives")

    if not alternative_rankings:
        st.info("No alternative perspectives available.")
        return

    # Create tabs: Original + 3 alternatives
    tab_names = ["Your Ranking"] + [
        alt.get("scenario_name", f"Perspective {i+1}")
        for i, alt in enumerate(alternative_rankings)
    ]
    tabs = st.tabs(tab_names)

    # Tab 0: Original ranking (top 8)
    with tabs[0]:
        st.markdown("**Your personalized ranking based on your preferences**")
        st.caption(f"Showing top {min(8, len(original_providers))} providers")

        for provider in original_providers[:8]:
            rank = provider.get('final_rank', 0)
            with st.container():
                st.markdown(f"### {rank}. {provider.get('name', 'Unknown Provider')}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rating", f"{provider.get('rating', 0)}/5.0")
                with col2:
                    st.metric("Distance", provider.get('distance', 'N/A'))
                with col3:
                    st.metric("Score", provider.get('final_score', 0))
                st.divider()

    # Tabs 1-3: Alternative perspectives
    for i, alt in enumerate(alternative_rankings):
        with tabs[i+1]:
            st.markdown(f"**{alt.get('description', 'Alternative perspective')}**")

            # Show adjusted weights
            weights = alt.get('adjusted_weights', {})
            if weights:
                st.caption("Adjusted preference weights:")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Location", f"{weights.get('location_weight', 0):.0%}")
                with col2:
                    st.metric("Rating", f"{weights.get('rating_weight', 0):.0%}")
                with col3:
                    st.metric("Insurance", f"{weights.get('insurance_priority', 0):.0%}")
                with col4:
                    st.metric("Experience", f"{weights.get('experience_weight', 0):.0%}")

            st.markdown("---")

            # Show key insights
            insights = alt.get('key_insights', [])
            if insights:
                st.markdown("**Key Insights:**")
                for insight in insights:
                    st.markdown(f"💡 {insight}")
                st.markdown("---")

            # Show reranked providers
            mapped_providers = map_alternative_to_full_providers(alt, original_providers)
            st.caption(f"Showing top {min(8, len(mapped_providers))} providers in this perspective")

            for provider in mapped_providers[:8]:
                render_alternative_provider_card(provider)


def render_agent_workflow(workflow_results: Dict[str, Any]) -> None:
    """Render agent workflow visualization."""
    st.header("🤖 Agent Decision Process")

    # Workflow overview
    execution_log = workflow_results.get("execution_log", [])
    agent_outputs = workflow_results.get("agent_outputs", {})

    # Agent status indicators
    col1, col2, col3 = st.columns(3)

    with col1:
        data_status = agent_outputs.get("data_gatherer", {}).get("status", "unknown")
        status_color = "🟢" if data_status == "success" else "🔴"
        st.markdown(f"{status_color} **Data Gatherer**")
        st.markdown(f"Status: {data_status.title()}")
        if data_status == "success":
            data_meta = agent_outputs.get("data_gatherer", {}).get("search_metadata", {})
            provider_count = len(agent_outputs.get("data_gatherer", {}).get("providers", []))
            fhir_count = data_meta.get("fhir_count", 0)
            st.markdown(f"Found: {provider_count} providers")
            if fhir_count > 0:
                st.markdown(f":green[FHIR verified: {fhir_count}]")

    with col2:
        score_status = agent_outputs.get("preference_scorer", {}).get("status", "unknown")
        status_color = "🟢" if score_status == "success" else "🔴"
        st.markdown(f"{status_color} **Preference Scorer**")
        st.markdown(f"Status: {score_status.title()}")
        if score_status == "success":
            top_provider = agent_outputs.get("preference_scorer", {}).get("scoring_metadata", {}).get("top_provider")
            if top_provider:
                st.markdown(f"Top: {top_provider}")

    with col3:
        validation_status = agent_outputs.get("critic_validator", {}).get("status", "unknown")
        status_color = "🟢" if validation_status == "success" else "🔴"
        st.markdown(f"{status_color} **Critic Validator**")
        st.markdown(f"Status: {validation_status.title()}")
        if validation_status == "success":
            confidence = agent_outputs.get("critic_validator", {}).get("validation_metadata", {}).get("ranking_confidence", "unknown")
            st.markdown(f"Confidence: {confidence.title()}")

    # Detailed agent outputs
    with st.expander("🔍 Detailed Agent Analysis"):
        tab1, tab2, tab3 = st.tabs(["Data Gatherer", "Preference Scorer", "Critic Validator"])

        with tab1:
            data_output = agent_outputs.get("data_gatherer", {})
            search_metadata = data_output.get("search_metadata", {})

            st.markdown("**Search Parameters:**")
            st.json({
                "query": search_metadata.get("query", "N/A"),
                "specialty": search_metadata.get("specialty", "N/A"),
                "location": search_metadata.get("location", "N/A"),
                "total_found": search_metadata.get("total_found", 0)
            })

        with tab2:
            score_output = agent_outputs.get("preference_scorer", {})
            scoring_metadata = score_output.get("scoring_metadata", {})

            st.markdown("**Scoring Summary:**")
            st.json({
                "total_providers": scoring_metadata.get("total_providers", 0),
                "scoring_method": scoring_metadata.get("scoring_method", "N/A"),
                "score_range": scoring_metadata.get("score_range", {})
            })

        with tab3:
            validation_output = agent_outputs.get("critic_validator", {})
            validation_results = validation_output.get("validation_results", {})

            # Show bias analysis
            if validation_results.get("bias_analysis"):
                bias_assessment = validation_results["bias_analysis"].get("bias_assessment", {})
                st.markdown("**Bias Assessment:**")
                st.markdown(f"**Severity:** {bias_assessment.get('severity', 'unknown').title()}")

                detected_biases = bias_assessment.get('detected_biases', [])
                if detected_biases:
                    st.markdown("**Detected Biases:**")
                    for bias in detected_biases:
                        st.markdown(f"• {bias}")

                st.markdown(f"**Explanation:** {bias_assessment.get('explanation', 'No explanation provided')}")

            # Show alternative perspectives
            alternative_rankings = validation_results.get("alternative_rankings", [])
            if alternative_rankings:
                st.markdown("**Alternative Ranking Perspectives:**")
                for alt in alternative_rankings:
                    st.markdown(f"**{alt.get('scenario_name', 'Unknown Scenario')}:** {alt.get('description', 'No description')}")


def render_validation_insights(workflow_results: Dict[str, Any]) -> None:
    """Render validation insights and alternative perspectives."""
    validation_output = workflow_results.get("agent_outputs", {}).get("critic_validator", {})
    validation_results = validation_output.get("validation_results", {})

    if not validation_results:
        return

    st.header("🛡️ Validation Insights")

    # Final recommendations from validator
    final_recommendations = validation_results.get("final_recommendations", {})
    if final_recommendations:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Recommendation Confidence")
            confidence = final_recommendations.get("recommendation_confidence", "medium")
            confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
            st.markdown(f"{confidence_color} **{confidence.title()} Confidence**")

            key_findings = final_recommendations.get("key_findings", [])
            if key_findings:
                st.markdown("**Key Findings:**")
                for finding in key_findings:
                    st.markdown(f"• {finding}")

        with col2:
            st.subheader("Important Considerations")
            considerations = final_recommendations.get("important_considerations", [])
            if considerations:
                for consideration in considerations:
                    st.info(consideration)

            user_guidance = final_recommendations.get("user_guidance", [])
            if user_guidance:
                st.markdown("**Guidance:**")
                for guidance in user_guidance:
                    st.markdown(f"• {guidance}")

    # Alternative ranking perspectives
    alternative_rankings = validation_results.get("alternative_rankings", [])
    if alternative_rankings:
        with st.expander("🔄 Alternative Ranking Perspectives"):
            for i, alt_ranking in enumerate(alternative_rankings):
                st.markdown(f"### {alt_ranking.get('scenario_name', f'Scenario {i+1}')}")
                st.markdown(alt_ranking.get('description', 'No description available'))

                # Show key insights
                insights = alt_ranking.get('key_insights', [])
                if insights:
                    for insight in insights:
                        st.markdown(f"💡 {insight}")

                st.divider()


def render_score_visualization(providers: List[Dict[str, Any]]) -> None:
    """Render provider score visualization."""
    if not providers:
        return

    st.header("📊 Provider Scoring Analysis")

    # Prepare data for visualization
    provider_data = []
    for provider in providers[:10]:  # Top 10 for visualization
        provider_data.append({
            "Name": provider.get("name", "Unknown")[:20] + "...",
            "Final Score": provider.get("final_score", 0),
            "Rating": provider.get("rating", 0),
            "AI Confidence": provider.get("ai_confidence", 50),
            "Distance": provider.get("distance", "N/A")
        })

    df = pd.DataFrame(provider_data)

    # Score comparison chart
    fig = px.bar(
        df,
        x="Name",
        y="Final Score",
        title="Provider Match Scores",
        color="Final Score",
        color_continuous_scale="viridis"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Rating vs AI Confidence scatter
    if len(df) > 1:
        fig2 = px.scatter(
            df,
            x="Rating",
            y="AI Confidence",
            size="Final Score",
            hover_name="Name",
            title="Provider Rating vs AI Confidence",
            labels={"Rating": "Provider Rating (1-5)", "AI Confidence": "AI Confidence %"}
        )
        st.plotly_chart(fig2, use_container_width=True)


def main():
    """Main application function."""
    init_session_state()
    render_header()

    config = get_config()
    rate_limiter = get_rate_limiter()

    # Authentication check
    authenticator = get_authenticator()

    # If not authenticated, show login form
    if not authenticator.require_authentication():
        authenticator.login_form()
        return

    # Check API keys
    if not check_api_keys():
        return

    # Sidebar configuration
    with st.sidebar:
        # User info
        username = st.session_state.get("username", "User")
        role = st.session_state.get("user_role", "user")
        st.markdown(f"### 👤 {username}")
        st.caption(f"Role: {role}")
        if st.button("🚪 Logout", use_container_width=True):
            authenticator.logout()

        st.markdown("---")
        st.header("⚙️ Configuration")

        # FHIR toggle
        st.session_state.fhir_enabled = st.toggle(
            "Use Insurance Directory (FHIR)",
            value=st.session_state.fhir_enabled,
            help="Query FHIR Provider Directory for verified in-network provider data"
        )

        # Show agent logs toggle (admins only)
        if authenticator.is_admin():
            st.session_state.show_agent_logs = st.toggle("Show Agent Logs", value=False)
        else:
            st.session_state.show_agent_logs = False
            st.caption("Agent logs are restricted to admin users.")

        # Clear results button
        if st.button("🗑️ Clear Results"):
            st.session_state.workflow_results = None
            st.session_state.search_executed = False
            st.rerun()

        # About section
        st.markdown("---")
        st.markdown("### About")
        st.markdown("CareCompass v1.0")
        st.markdown("Built with Streamlit, LangGraph, and multi-agent AI")

    # Main content
    search_params = render_search_form()

    # Detect if parameters changed and reset execution flag
    if search_params:
        if search_params_changed(search_params, st.session_state.last_search_params):
            st.session_state.search_executed = False
            st.session_state.last_search_params = search_params.copy()

    # Execute search if parameters provided
    if search_params and not st.session_state.search_executed:
        user = st.session_state.get("username", "anonymous")
        allowed, retry_after = rate_limiter.check(
            f"workflow:{user}",
            config.RATE_LIMIT_MAX_REQUESTS,
            config.RATE_LIMIT_WINDOW_SECONDS
        )
        if not allowed:
            st.warning(f"Rate limit exceeded. Try again in {retry_after} seconds.")
            log_audit_event(
                "rate_limit_exceeded",
                user=user,
                success=False,
                details={"retry_after": retry_after}
            )
            return

        log_audit_event(
            "workflow_started",
            user=user,
            details={
                "specialty": search_params.get("specialty"),
                "location": search_params.get("location"),
                "insurance": search_params.get("insurance")
            }
        )

        with st.spinner("🔄 CareCompass agents are working..."):
            try:
                # Set FHIR config based on toggle before creating orchestrator
                import os
                os.environ["FHIR_ENABLED"] = str(st.session_state.fhir_enabled).lower()

                # Create orchestrator and execute workflow
                orchestrator = create_orchestrator()

                # Use streaming execution to capture execution_log
                workflow_results = orchestrator.execute_workflow_streaming(
                    specialty=search_params["specialty"],
                    location=search_params["location"],
                    insurance=search_params["insurance"],
                    preferences=search_params["preferences"],
                    progress_callback=None  # No callback - timeline will be shown post-execution
                )

                st.session_state.workflow_results = workflow_results
                st.session_state.search_executed = True

                if workflow_results.get("success"):
                    provider_count = len(workflow_results.get("final_recommendations", []))
                    log_audit_event(
                        "workflow_completed",
                        user=user,
                        details={"providers": provider_count}
                    )
                else:
                    log_audit_event("workflow_failed", user=user, success=False)

                st.rerun()

            except Exception as e:
                logger.error(f"Workflow execution failed: {e}", exc_info=True)
                log_audit_event("workflow_failed", user=user, success=False, details={"error": str(e)})
                st.error("An error occurred while processing your search. Please try again later.")

                # Show details only in debug mode
                if config.DEBUG:
                    with st.expander("🐛 Error Details (Debug Mode)"):
                        st.code(str(e))

    # Display results if available
    if st.session_state.workflow_results:
        workflow_results = st.session_state.workflow_results

        if workflow_results.get("success"):
            recommendations = workflow_results.get("final_recommendations", [])

            if recommendations:
                st.header("🎯 Provider Recommendations")

                # Display provider cards
                for recommendation in recommendations:
                    provider = recommendation.get("provider", {})
                    rank = recommendation.get("rank", 0)
                    render_provider_card(provider, rank)

                # Visualizations
                providers = [rec.get("provider", {}) for rec in recommendations]
                render_score_visualization(providers)

                # Alternative perspectives
                if workflow_results.get("alternative_perspectives"):
                    render_alternative_perspectives(
                        providers,
                        workflow_results["alternative_perspectives"]
                    )

                # Agent workflow
                if st.session_state.show_agent_logs:
                    render_agent_workflow(workflow_results)

                    # Show execution timeline
                    execution_log = workflow_results.get("execution_log", [])
                    if execution_log:
                        st.markdown("---")
                        render_execution_timeline(execution_log)

                # Validation insights
                render_validation_insights(workflow_results)

            else:
                st.warning("No provider recommendations found. Try adjusting your search criteria.")

        else:
            st.error("Search failed. Please check your input and try again.")
            error_messages = workflow_results.get("error_messages", [])
            if error_messages:
                with st.expander("Error Details"):
                    for error in error_messages:
                        st.error(error)


if __name__ == "__main__":
    main()