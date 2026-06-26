"""Preference Scorer Agent for ranking healthcare providers based on user preferences using GPT-4o-mini."""

import logging
from typing import Dict, List, Any, Optional
import json
from openai import OpenAI

from utils.config import get_config
from utils.security import InputValidator, PromptSanitizer

logger = logging.getLogger(__name__)


def assess_insurance_confidence(insurance_list: List[str]) -> Dict[str, Any]:
    """Assess confidence in insurance data quality.

    Args:
        insurance_list: List of insurance names/types accepted by provider

    Returns:
        Dictionary with confidence score, quality level, and specific plan count
    """
    if not insurance_list:
        return {'confidence': 0.0, 'quality': 'missing', 'specific_plans': 0}

    vague_terms = ['most', 'major', 'many', 'various', 'accepts insurance']
    specific_plans = [
        'aetna', 'blue cross', 'blue shield', 'cigna', 'unitedhealth', 'united healthcare',
        'medicare', 'medicaid', 'humana', 'anthem', 'kaiser', 'wellcare', 'molina'
    ]

    insurance_text = " ".join(str(ins) for ins in insurance_list).lower()

    # Check for vague terms
    has_vague = any(term in insurance_text for term in vague_terms)

    # Check for specific plans
    specific_count = sum(1 for plan in specific_plans if plan in insurance_text)

    if specific_count >= 2:
        return {'confidence': 1.0, 'quality': 'high', 'specific_plans': specific_count}
    elif specific_count == 1:
        return {'confidence': 0.7, 'quality': 'medium', 'specific_plans': 1}
    elif has_vague:
        return {'confidence': 0.3, 'quality': 'low_vague', 'specific_plans': 0}
    else:
        return {'confidence': 0.5, 'quality': 'low', 'specific_plans': 0}


def calculate_rating_score_with_confidence(rating: float, review_count: Optional[int] = None) -> Dict[str, Any]:
    """Calculate rating score with confidence adjustment using Bayesian average.

    Low review counts are pulled toward a global average (3.5/5) to avoid over-weighting
    ratings with insufficient sample size.

    Args:
        rating: Provider rating (0-5)
        review_count: Number of reviews (None if not available)

    Returns:
        Dictionary with adjusted score, confidence level, and reliability
    """
    if rating <= 0:
        return {
            'score': 0,
            'confidence': 'no_rating',
            'adjusted_rating': 0,
            'original_rating': 0,
            'review_count': review_count or 0,
            'reliability': 'unknown'
        }

    # Default review count if missing
    if review_count is None:
        review_count = 15  # Assume moderate confidence for verified providers
        confidence_level = 'medium_assumed'
    elif review_count < 5:
        confidence_level = 'low'
    elif review_count < 20:
        confidence_level = 'medium'
    else:
        confidence_level = 'high'

    # Bayesian average calculation
    global_avg_rating = 3.5  # Conservative global average
    confidence_weight = 10   # Weight of prior belief

    # Calculate weighted rating (more reviews = closer to actual rating, fewer = closer to global avg)
    adjusted_rating = (
        (confidence_weight * global_avg_rating + review_count * rating) /
        (confidence_weight + review_count)
    )

    # Convert to 0-100 score
    score = (adjusted_rating / 5.0) * 100

    return {
        'score': round(score, 2),
        'confidence': confidence_level,
        'adjusted_rating': round(adjusted_rating, 2),
        'original_rating': rating,
        'review_count': review_count,
        'reliability': 'high' if review_count >= 20 else 'moderate' if review_count >= 5 else 'low'
    }


def calculate_missing_data_score(preference_weight: float, default_score: int = 50) -> int:
    """Calculate score for missing data based on user preference weight.

    When data is missing, penalize more heavily if user cares about that factor.

    Args:
        preference_weight: How important this factor is to user (0.0-1.0)
        default_score: Default neutral score (typically 50)

    Returns:
        Adjusted score for missing data
    """
    if preference_weight >= 0.5:
        # User prioritizes this factor - significant penalty
        return int(default_score * 0.4)  # 40% of default
    elif preference_weight >= 0.3:
        # Medium priority - moderate penalty
        return int(default_score * 0.7)  # 70% of default
    else:
        # Low priority - minimal penalty
        return default_score


def interpret_rating_status(rating: float, review_count: Optional[int]) -> Dict[str, Any]:
    """Distinguish between no rating vs poor rating.

    Args:
        rating: Provider rating (0-5)
        review_count: Number of reviews

    Returns:
        Dictionary with status, display text, scoring approach, and warning
    """
    if rating == 0 and (review_count is None or review_count == 0):
        return {
            'status': 'unrated',
            'display': 'No reviews yet',
            'score_approach': 'neutral_with_penalty',
            'base_score': 40,  # Slightly below neutral
            'warning': 'New provider or limited online presence'
        }
    elif rating == 0 and review_count and review_count > 0:
        # Unusual - might be data error
        return {
            'status': 'data_anomaly',
            'display': 'Rating unavailable',
            'score_approach': 'neutral',
            'base_score': 50,
            'warning': 'Rating data may be incomplete'
        }
    elif 0 < rating < 2.5 and review_count and review_count >= 5:
        return {
            'status': 'poor_quality',
            'display': f'{rating}/5.0 (⚠️ Low)',
            'score_approach': 'penalize',
            'base_score': (rating / 5.0) * 100,
            'warning': 'Below average patient ratings'
        }
    else:
        return {
            'status': 'valid_rating',
            'display': f'{rating}/5.0',
            'score_approach': 'normal',
            'base_score': (rating / 5.0) * 100,
            'warning': None
        }


class PreferenceScorerAgent:
    """Agent responsible for scoring and ranking healthcare providers based on user preferences."""

    def __init__(self):
        """Initialize the preference scorer with OpenAI client."""
        self.config = get_config()
        self.openai_client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            if not self.config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found in configuration")

            self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            logger.info("Preference scorer client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize preference scorer client: {e}")
            raise

    def _calculate_base_scores(self, providers: List[Dict[str, Any]], preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate base scores for providers using weighted algorithm.

        Args:
            providers: List of provider dictionaries
            preferences: User preference weights

        Returns:
            List of providers with base scores
        """
        scored_providers = []

        for provider in providers:
            score = 0.0
            score_breakdown = {}
            data_quality_flags = {}
            data_warnings = []

            # Rating score (0-100) with confidence adjustment
            rating = float(provider.get("rating", 0))
            review_count = provider.get("review_count", None)

            # Check rating status first
            rating_status = interpret_rating_status(rating, review_count)
            if rating_status['warning']:
                data_warnings.append(rating_status['warning'])

            # Calculate confidence-adjusted rating score
            rating_result = calculate_rating_score_with_confidence(rating, review_count)
            rating_score = rating_result['score']

            score_breakdown["rating"] = {
                "value": rating,
                "review_count": review_count,
                "adjusted_rating": rating_result.get('adjusted_rating', rating),
                "score": rating_score,
                "weight": preferences.get("rating_weight", 0.3),
                "confidence": rating_result.get('confidence', 'unknown'),
                "reliability": rating_result.get('reliability', 'unknown')
            }
            score += rating_score * preferences.get("rating_weight", 0.3)
            data_quality_flags['rating'] = rating_result.get('confidence', 'unknown')

            # Location/Distance score (0-100) with missing data penalty
            distance = provider.get("distance", None)
            if distance and isinstance(distance, (int, float, str)):
                try:
                    dist_value = float(str(distance).replace("mi", "").replace("miles", "").strip())
                    # Score decreases with distance (max 50 miles)
                    distance_score = max(0, 100 - (dist_value / 50 * 100))
                    data_quality_flags['distance'] = 'complete'
                except:
                    # Invalid distance format - penalize based on user preference
                    distance_score = calculate_missing_data_score(
                        preferences.get("location_weight", 0.4), 50
                    )
                    data_quality_flags['distance'] = 'invalid'
            else:
                # Missing distance - penalize based on how important location is to user
                distance_score = calculate_missing_data_score(
                    preferences.get("location_weight", 0.4), 50
                )
                data_quality_flags['distance'] = 'missing'

            score_breakdown["location"] = {
                "value": distance,
                "score": distance_score,
                "weight": preferences.get("location_weight", 0.4),
                "data_quality": data_quality_flags['distance']
            }
            score += distance_score * preferences.get("location_weight", 0.4)

            # Insurance score (0-100) with confidence assessment
            preferred_insurance = preferences.get("insurance", "").lower()
            insurance_accepted = provider.get("insurance_accepted", [])

            # Assess insurance data quality
            insurance_confidence = assess_insurance_confidence(insurance_accepted)

            if preferred_insurance and insurance_accepted:
                # User specified insurance AND provider has insurance data
                insurance_text = " ".join(str(ins) for ins in insurance_accepted).lower()

                if preferred_insurance in insurance_text:
                    # Base match score adjusted by confidence
                    base_score = 100
                    confidence_multiplier = insurance_confidence['confidence']
                    insurance_score = base_score * confidence_multiplier

                    # Add warning if vague
                    if insurance_confidence['quality'] == 'low_vague':
                        data_warnings.append("Vague insurance information - verify with provider")
                else:
                    insurance_score = 0

                data_quality_flags['insurance'] = insurance_confidence['quality']
            elif preferred_insurance and not insurance_accepted:
                # User wants specific insurance but provider has no data - penalize
                insurance_score = 25  # Low score when user cares but data missing
                data_quality_flags['insurance'] = 'missing'
            elif not preferred_insurance and insurance_accepted:
                # Provider has insurance info but user didn't specify
                # Score based on data quality (high quality = more bonus)
                insurance_score = 50 + (insurance_confidence['confidence'] * 20)  # 50-70 range
                data_quality_flags['insurance'] = insurance_confidence['quality']
            else:
                # No preference and no data - neutral
                insurance_score = 50  # Default neutral score
                data_quality_flags['insurance'] = 'missing'

            score_breakdown["insurance"] = {
                "value": insurance_accepted,
                "score": insurance_score,
                "weight": preferences.get("insurance_priority", 0.3),
                "confidence": insurance_confidence.get('confidence', 0.0),
                "quality": insurance_confidence.get('quality', 'unknown'),
                "specific_plans": insurance_confidence.get('specific_plans', 0)
            }
            score += insurance_score * preferences.get("insurance_priority", 0.3)

            # Experience bonus
            years_exp = provider.get("years_experience", None)
            if years_exp and isinstance(years_exp, (int, float)):
                exp_bonus = min(20, years_exp * 2)  # Max 20 points for 10+ years
                score += exp_bonus
                score_breakdown["experience_bonus"] = exp_bonus
            else:
                score_breakdown["experience_bonus"] = 0

            # FHIR network verification bonus
            fhir_metadata = provider.get("fhir_metadata", {})
            if fhir_metadata.get("network_verified"):
                network_bonus = 10
                score += network_bonus
                score_breakdown["network_verification_bonus"] = network_bonus
                data_quality_flags["network_verified"] = True
            else:
                score_breakdown["network_verification_bonus"] = 0

            provider_copy = provider.copy()
            provider_copy["base_score"] = round(score, 2)
            provider_copy["score_breakdown"] = score_breakdown
            provider_copy["data_quality_flags"] = data_quality_flags
            provider_copy["data_warnings"] = data_warnings
            provider_copy["rating_confidence"] = rating_result
            provider_copy["insurance_confidence"] = insurance_confidence
            scored_providers.append(provider_copy)

        return scored_providers

    def _generate_ai_rankings(self, providers: List[Dict[str, Any]], preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use GPT-4o-mini to analyze and rank providers with reasoning.

        Args:
            providers: List of scored provider dictionaries
            preferences: User preferences

        Returns:
            List of providers with AI rankings and reasoning
        """
        try:
            # Sanitize preferences first
            safe_preferences = InputValidator.validate_preferences(preferences)

            # Prepare provider data for AI analysis
            provider_summaries = []
            for i, provider in enumerate(providers):
                summary = {
                    "index": i,
                    "name": provider.get("name", "Unknown"),
                    "specialty": provider.get("specialty", ""),
                    "location": provider.get("location", ""),
                    "rating": provider.get("rating", 0),
                    "distance": provider.get("distance", "N/A"),
                    "insurance": provider.get("insurance_accepted", []),
                    "base_score": provider.get("base_score", 0),
                    "years_experience": provider.get("years_experience", "N/A"),
                    "network_verified": provider.get("fhir_metadata", {}).get("network_verified", False),
                    "data_source": provider.get("data_source", "tavily"),
                }
                provider_summaries.append(summary)

            # Use structured prompt with XML-style delimiters
            safe_notes = PromptSanitizer.escape_for_prompt(safe_preferences.get('notes', ''))
            safe_insurance = PromptSanitizer.escape_for_prompt(preferences.get('insurance', 'N/A'))

            prompt = f"""You are a healthcare provider matching expert. Analyze and rank these providers based on the user's preferences.

Your task is to provide fair, unbiased rankings based ONLY on the data and preferences provided.

<user_preferences>
Location Weight: {safe_preferences.get('location_weight', 0.4)} (importance of proximity)
Rating Weight: {safe_preferences.get('rating_weight', 0.3)} (importance of patient reviews)
Insurance Priority: {safe_preferences.get('insurance_priority', 0.3)} (importance of insurance acceptance)
Preferred Insurance: {safe_insurance}
Additional Notes: {safe_notes if safe_notes else 'None'}
</user_preferences>

<important_note>
Some providers have "network_verified": true, meaning their insurance network membership has been verified through an official FHIR Provider Directory (authoritative payer data). This is more reliable than self-reported or web-scraped insurance information. Give a slight preference to network-verified providers when insurance is important to the user.
</important_note>

<providers_data>
{json.dumps(provider_summaries, indent=2)}
</providers_data>

<task_instructions>
1. Analyze each provider considering ALL user preferences
2. Look beyond just base scores - consider qualitative factors
3. Rank providers from 1 (best) to {len(providers)} (worst)
4. For each provider, provide:
   - Final rank (1-{len(providers)})
   - AI confidence score (0-100)
   - Detailed reasoning (2-3 sentences explaining the ranking decision)
   - Key strengths and potential concerns
5. Do NOT include any explanatory text, ONLY return the JSON array
</task_instructions>

<output_format>
Return ONLY a JSON array with this structure:
[
  {{
    "provider_index": 0,
    "ai_rank": 1,
    "ai_confidence": 95,
    "reasoning": "This provider ranks highest because...",
    "strengths": ["Excellent ratings", "Accepts preferred insurance"],
    "concerns": ["Slightly farther distance"]
  }}
]
</output_format>

Ensure rankings are justified and consider the user's specific preference weights."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a healthcare provider ranking expert. Provide detailed, justified rankings based on user preferences."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )

            response_text = response.choices[0].message.content.strip()

            # Parse AI response
            try:
                # Extract JSON from potential markdown code blocks
                json_text = response_text
                if "```json" in response_text:
                    # Extract JSON from markdown code block
                    json_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    # Extract from generic code block
                    json_text = response_text.split("```")[1].split("```")[0].strip()

                ai_rankings = json.loads(json_text)

                if not isinstance(ai_rankings, list):
                    logger.warning("AI response was not a JSON array")
                    return providers

                # Apply AI rankings to providers
                ranked_providers = providers.copy()
                for ranking in ai_rankings:
                    provider_idx = ranking.get("provider_index")
                    if 0 <= provider_idx < len(ranked_providers):
                        ranked_providers[provider_idx].update({
                            "ai_rank": ranking.get("ai_rank", 999),
                            "ai_confidence": ranking.get("ai_confidence", 50),
                            "ai_reasoning": ranking.get("reasoning", "No reasoning provided"),
                            "ai_strengths": ranking.get("strengths", []),
                            "ai_concerns": ranking.get("concerns", [])
                        })

                # Sort by AI ranking
                ranked_providers.sort(key=lambda x: x.get("ai_rank", 999))

                logger.info(f"AI ranking completed for {len(ranked_providers)} providers")
                return ranked_providers

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI ranking response: {e}")
                logger.debug(f"AI response was: {response_text}")
                return providers

        except Exception as e:
            logger.error(f"AI ranking failed: {e}", exc_info=True)
            return providers

    def score_providers(self, providers: List[Dict[str, Any]], preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to score and rank healthcare providers.

        Args:
            providers: List of provider dictionaries from data gatherer
            preferences: User preference dictionary with weights

        Returns:
            Dictionary containing ranked providers and scoring metadata
        """
        try:
            logger.info(f"Starting provider scoring for {len(providers)} providers")

            if not providers:
                return {
                    "ranked_providers": [],
                    "scoring_metadata": {
                        "total_providers": 0,
                        "preferences_used": preferences,
                        "scoring_method": "weighted_algorithm_with_ai"
                    },
                    "status": "no_providers",
                    "message": "No providers to score"
                }

            # Step 1: Calculate base scores using weighted algorithm
            scored_providers = self._calculate_base_scores(providers, preferences)

            # Step 2: Generate AI rankings and reasoning
            ranked_providers = self._generate_ai_rankings(scored_providers, preferences)

            # Step 3: Add final composite score
            for provider in ranked_providers:
                # Combine base score with AI ranking (inverse weight for rank)
                ai_rank_score = (len(ranked_providers) - provider.get("ai_rank", len(ranked_providers))) * 10
                composite_score = (provider.get("base_score", 0) * 0.7) + (ai_rank_score * 0.3)

                provider["final_score"] = round(composite_score, 2)

            # Re-sort by final score to ensure rankings match scores
            ranked_providers.sort(key=lambda x: x.get("final_score", 0), reverse=True)

            # Assign final ranks based on sorted order
            for i, provider in enumerate(ranked_providers):
                provider["final_rank"] = i + 1

            result = {
                "ranked_providers": ranked_providers,
                "scoring_metadata": {
                    "total_providers": len(ranked_providers),
                    "preferences_used": preferences,
                    "scoring_method": "weighted_algorithm_with_ai",
                    "top_provider": ranked_providers[0]["name"] if ranked_providers else None,
                    "score_range": {
                        "highest": ranked_providers[0]["final_score"] if ranked_providers else 0,
                        "lowest": ranked_providers[-1]["final_score"] if ranked_providers else 0
                    }
                },
                "status": "success",
                "message": f"Successfully ranked {len(ranked_providers)} providers"
            }

            logger.info(f"Provider scoring completed: {result['message']}")
            return result

        except Exception as e:
            logger.error(f"Provider scoring failed: {e}", exc_info=True)
            return {
                "ranked_providers": [],
                "scoring_metadata": {
                    "total_providers": len(providers),
                    "preferences_used": preferences,
                    "scoring_method": "weighted_algorithm_with_ai"
                },
                "status": "error",
                "message": "Error scoring providers. Please try again."
            }

    def explain_scoring_methodology(self) -> Dict[str, Any]:
        """Explain the scoring methodology used by the agent.

        Returns:
            Dictionary explaining the scoring approach
        """
        return {
            "methodology": "Hybrid Weighted Algorithm + AI Ranking",
            "components": {
                "base_scoring": {
                    "description": "Weighted algorithm using user preferences",
                    "factors": [
                        "Provider rating (0-5 stars) normalized to 0-100",
                        "Distance/location proximity score (0-100)",
                        "Insurance acceptance match (0-100)",
                        "Years of experience bonus (up to 20 points)"
                    ]
                },
                "ai_ranking": {
                    "description": "GPT-4o-mini analyzes qualitative factors",
                    "model": "gpt-4o-mini",
                    "provides": [
                        "Detailed reasoning for each ranking decision",
                        "Confidence scores (0-100)",
                        "Identified strengths and concerns",
                        "Contextual analysis beyond numeric scores"
                    ]
                },
                "final_scoring": {
                    "description": "Composite score combining base algorithm (70%) and AI ranking (30%)",
                    "output": "Final ranked list with comprehensive explanations"
                }
            },
            "advantages": [
                "Combines quantitative and qualitative analysis",
                "Transparent scoring breakdown",
                "AI-generated reasoning for each decision",
                "Customizable preference weighting"
            ]
        }


def create_preference_scorer() -> PreferenceScorerAgent:
    """Factory function to create a PreferenceScorerAgent instance.

    Returns:
        PreferenceScorerAgent instance
    """
    return PreferenceScorerAgent()