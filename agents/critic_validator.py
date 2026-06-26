"""Critic Validator Agent for challenging and validating provider rankings using Claude Sonnet 4.5."""

import logging
from typing import Dict, List, Any, Optional
import json
import re
from anthropic import Anthropic

from utils.config import get_config
from utils.security import InputValidator

logger = logging.getLogger(__name__)


class CriticValidatorAgent:
    """Agent responsible for critically evaluating and validating provider rankings with sophisticated reasoning."""

    def __init__(self):
        """Initialize the critic validator with Anthropic client."""
        self.config = get_config()
        self.anthropic_client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Anthropic client."""
        try:
            if not self.config.ANTHROPIC_API_KEY:
                raise ValueError("Anthropic API key not found in configuration")

            self.anthropic_client = Anthropic(api_key=self.config.ANTHROPIC_API_KEY)
            logger.info("Critic validator client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize critic validator client: {e}")
            raise

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from markdown-wrapped responses.

        Args:
            response_text: Raw response text from Claude

        Returns:
            Cleaned JSON string
        """
        # Try to extract JSON from markdown code blocks
        if "```json" in response_text:
            # Extract JSON from markdown code block
            match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if match:
                return match.group(1).strip()
        elif "```" in response_text:
            # Extract from generic code block
            match = re.search(r'```\s*([\s\S]*?)\s*```', response_text)
            if match:
                return match.group(1).strip()

        # If no code blocks, try to find JSON object or array
        # Look for JSON object {...} or array [...]
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', response_text)
        if json_match:
            return json_match.group(1).strip()

        return response_text.strip()

    def _analyze_ranking_bias(self, ranked_providers: List[Dict[str, Any]], preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential biases in the current rankings.

        Args:
            ranked_providers: List of ranked provider dictionaries
            preferences: User preferences used for ranking

        Returns:
            Dictionary containing bias analysis
        """
        try:
            # Sanitize preferences first
            safe_preferences = InputValidator.validate_preferences(preferences)

            # Prepare data for analysis
            ranking_data = []
            for i, provider in enumerate(ranked_providers[:10]):  # Top 10 for analysis
                ranking_data.append({
                    "rank": i + 1,
                    "name": provider.get("name", "Unknown"),
                    "rating": provider.get("rating", 0),
                    "distance": provider.get("distance", "N/A"),
                    "insurance": provider.get("insurance_accepted", []),
                    "final_score": provider.get("final_score", 0),
                    "ai_reasoning": provider.get("ai_reasoning", "No reasoning"),
                    "years_experience": provider.get("years_experience", "N/A")
                })

            prompt = f"""As a critical healthcare analytics expert, analyze this provider ranking for potential biases, blind spots, and alternative perspectives.

<user_preferences>
{json.dumps(safe_preferences, indent=2)}
</user_preferences>

CURRENT TOP RANKINGS:
{json.dumps(ranking_data, indent=2)}

CRITICAL ANALYSIS REQUIRED:

1. BIAS DETECTION:
   - Are rankings overly influenced by any single factor?
   - Do preferences create unfair advantages/disadvantages?
   - Are there geographic or demographic biases?

2. BLIND SPOTS:
   - What important factors might be missing?
   - Are there hidden quality indicators not considered?
   - Could the ranking mislead patients?

3. ALTERNATIVE PERSPECTIVES:
   - How might different patient types rank these differently?
   - What if priorities were weighted differently?
   - Are there red flags in highly ranked providers?

4. RANKING VALIDITY:
   - Do top-ranked providers truly serve user needs?
   - Are lower-ranked providers unfairly penalized?
   - Is the ranking methodology sound?

IMPORTANT: Return ONLY valid JSON. No markdown, no explanations, just pure JSON.
Ensure all strings are properly escaped. Use double quotes for all keys and string values.

Return analysis as this exact JSON structure:
{{
  "bias_assessment": {{
    "detected_biases": ["bias1", "bias2"],
    "severity": "low",
    "explanation": "Brief explanation here"
  }},
  "blind_spots": {{
    "missing_factors": ["factor1", "factor2"],
    "impact": "Impact description",
    "recommendations": ["rec1", "rec2"]
  }},
  "alternative_scenarios": [
    {{
      "scenario": "Scenario description",
      "different_outcome": "Outcome description",
      "rationale": "Rationale explanation"
    }}
  ],
  "validity_concerns": {{
    "ranking_issues": ["issue1", "issue2"],
    "misleading_aspects": ["aspect1", "aspect2"],
    "confidence_level": "medium"
  }},
  "overall_assessment": "Brief assessment summary"
}}

Be thorough and critical. Return ONLY the JSON object, nothing else."""

            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=3500,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()
            # Extract JSON from markdown if needed
            cleaned_response = self._extract_json_from_response(response_text)

            try:
                bias_analysis = json.loads(cleaned_response)
                logger.info("Bias analysis completed successfully")
                return bias_analysis

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse bias analysis response: {e}")
                logger.error(f"Response was: {cleaned_response[:500]}...")
                return {
                    "bias_assessment": {"detected_biases": [], "severity": "unknown", "explanation": "Analysis failed"},
                    "blind_spots": {"missing_factors": [], "impact": "Unknown", "recommendations": []},
                    "alternative_scenarios": [],
                    "validity_concerns": {"ranking_issues": [], "misleading_aspects": [], "confidence_level": "low"},
                    "overall_assessment": "Critical analysis could not be completed"
                }

        except Exception as e:
            logger.error(f"Bias analysis failed: {e}")
            return {
                "bias_assessment": {"detected_biases": [], "severity": "unknown", "explanation": "Analysis failed"},
                "blind_spots": {"missing_factors": [], "impact": "Unknown", "recommendations": []},
                "alternative_scenarios": [],
                "validity_concerns": {"ranking_issues": [], "misleading_aspects": [], "confidence_level": "low"},
                "overall_assessment": f"Error during analysis: {str(e)}"
            }

    def _generate_alternative_rankings(self, ranked_providers: List[Dict[str, Any]], preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative ranking approaches and perspectives.

        Args:
            ranked_providers: Current ranked providers
            preferences: Original user preferences

        Returns:
            List of alternative ranking scenarios
        """
        try:
            # Sanitize preferences first
            safe_preferences = InputValidator.validate_preferences(preferences)

            # Focus on top providers for alternative analysis
            top_providers = ranked_providers[:8]
            provider_data = []

            for provider in top_providers:
                provider_data.append({
                    "name": provider.get("name", "Unknown"),
                    "rating": provider.get("rating", 0),
                    "location": provider.get("location", ""),
                    "distance": provider.get("distance", "N/A"),
                    "insurance": provider.get("insurance_accepted", []),
                    "years_experience": provider.get("years_experience", "N/A"),
                    "current_rank": provider.get("final_rank", 999),
                    "current_score": provider.get("final_score", 0)
                })

            prompt = f"""Generate alternative ranking perspectives for these healthcare providers.

<current_user_preferences>
{json.dumps(safe_preferences, indent=2)}
</current_user_preferences>

PROVIDERS TO RE-EVALUATE:
{json.dumps(provider_data, indent=2)}

Create 3 ALTERNATIVE RANKING scenarios:

1. "Quality-First Perspective": Prioritize clinical excellence over convenience
   - Heavy weight on ratings and experience
   - Minimal weight on distance
   - Consider years of practice, credentials

2. "Accessibility-First Perspective": Prioritize patient access and convenience
   - Heavy weight on location/distance
   - Strong insurance consideration
   - Consider appointment availability factors

3. "Balanced Healthcare Perspective": Consider factors often overlooked
   - Weight patient volume (not too busy, not too light)
   - Consider hospital affiliations
   - Balance new vs. experienced providers

For each scenario, provide:
- Adjusted preference weights
- Re-ranked provider order
- Reasoning for each provider's new position
- Key differences from original ranking

IMPORTANT: Return ONLY valid JSON array. No markdown, no extra text, just the JSON array.
Ensure all strings are properly escaped and use double quotes.

Return as JSON array (3 scenarios):
[
  {{
    "scenario_name": "Quality-First Perspective",
    "description": "Focus on clinical excellence",
    "adjusted_weights": {{
      "rating_weight": 0.6,
      "location_weight": 0.1,
      "insurance_priority": 0.2,
      "experience_weight": 0.1
    }},
    "reranked_providers": [
      {{
        "name": "Provider Name",
        "new_rank": 1,
        "original_rank": 3,
        "reasoning": "Brief reasoning here",
        "new_score": 95
      }}
    ],
    "key_insights": ["insight1", "insight2"]
  }}
]

Provide 3 scenarios. Return ONLY the JSON array, nothing else."""

            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=4000,  # Increased from 2500 to handle more providers
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()
            # Extract JSON from markdown if needed
            cleaned_response = self._extract_json_from_response(response_text)

            try:
                alternative_rankings = json.loads(cleaned_response)
                logger.info(f"Generated {len(alternative_rankings)} alternative ranking scenarios")
                return alternative_rankings

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse alternative rankings response: {e}")
                logger.error(f"Response preview (first 500 chars): {cleaned_response[:500]}")
                logger.error(f"Response preview (last 500 chars): {cleaned_response[-500:]}")

                # Try to fix common JSON issues
                try:
                    # Remove trailing commas before closing brackets/braces
                    fixed_response = re.sub(r',\s*([}\]])', r'\1', cleaned_response)
                    # Try parsing the fixed version
                    alternative_rankings = json.loads(fixed_response)
                    logger.info(f"Successfully parsed after fixing JSON syntax")
                    return alternative_rankings
                except:
                    logger.error(f"Could not recover from JSON error")
                    return []

        except Exception as e:
            logger.error(f"Alternative rankings generation failed: {e}")
            return []

    def _validate_top_recommendations(self, ranked_providers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the top recommended providers with detailed scrutiny.

        Args:
            ranked_providers: List of ranked providers

        Returns:
            Dictionary with validation results for top providers
        """
        try:
            top_3_providers = ranked_providers[:3]
            validation_data = []

            for provider in top_3_providers:
                validation_data.append({
                    "name": provider.get("name", "Unknown"),
                    "rank": provider.get("final_rank", 999),
                    "rating": provider.get("rating", 0),
                    "review_count": provider.get("review_count", 0),
                    "review_summary": provider.get("review_summary", "No reviews available"),
                    "review_sentiment": provider.get("review_sentiment", "unknown"),
                    "final_score": provider.get("final_score", 0),
                    "ai_reasoning": provider.get("ai_reasoning", ""),
                    "ai_confidence": provider.get("ai_confidence", 50),
                    "strengths": provider.get("ai_strengths", []),
                    "concerns": provider.get("ai_concerns", []),
                    "insurance": provider.get("insurance_accepted", []),
                    "location": provider.get("location", ""),
                    "distance": provider.get("distance", "N/A")
                })

            prompt = f"""As a senior healthcare quality auditor, rigorously validate these TOP 3 provider recommendations.

TOP PROVIDERS TO VALIDATE:
{json.dumps(validation_data, indent=2)}

Each provider includes:
- Basic info (name, rating, review_count)
- Review data (review_summary with patient feedback themes, review_sentiment: positive/mixed/negative)
- AI analysis (reasoning, confidence, strengths, concerns)
- Practical factors (location, distance, insurance)

VALIDATION CHECKLIST:

1. RECOMMENDATION QUALITY:
   - Is each provider truly suitable for the user's needs?
   - Are the rankings justified and logical?
   - Do AI confidence scores align with actual provider quality?
   - Does the review sentiment align with the rating and AI assessment?

2. REVIEW DATA ANALYSIS:
   - Are there red flags in the review summaries (e.g., "long wait times", "rude staff", "poor communication")?
   - Does review sentiment contradict the high rating or ranking?
   - Is the review_count sufficient to trust the rating? (Note: review counts may be incomplete from web sources, focus on relative differences)
   - Do review summaries provide meaningful context for differentiating providers?

3. RED FLAGS CHECK:
   - Any concerning patterns in the data or reviews?
   - Missing critical information that should disqualify providers?
   - Overconfidence in limited data?
   - Negative review sentiment despite high scores?

4. USER SAFETY:
   - Would you personally recommend these providers to a family member based on reviews and data?
   - Are there better alternatives that might have been overlooked?
   - Any risk factors from reviews that patients should know about?

5. RANKING ACCURACY:
   - Is the #1 provider truly the best choice considering reviews?
   - Are rankings 2 and 3 appropriately positioned?
   - Should any provider be excluded from top recommendations based on review feedback?

IMPORTANT: Return ONLY valid JSON. No markdown, no explanations, just pure JSON.
Ensure all strings are properly escaped. Use double quotes for all keys and string values.

Return detailed validation as this exact JSON structure:
{{
  "top_provider_validations": [
    {{
      "provider_name": "Provider Name",
      "rank": 1,
      "validation_status": "approved",
      "confidence_in_recommendation": "medium",
      "validation_notes": "Brief assessment",
      "red_flags": ["flag1", "flag2"],
      "recommendation_adjustments": "Brief adjustments",
      "patient_considerations": "Brief considerations"
    }}
  ],
  "overall_ranking_validity": {{
    "status": "validated",
    "confidence": "medium",
    "summary": "Brief overall assessment",
    "improvement_suggestions": ["suggestion1", "suggestion2"]
  }}
}}

Be extremely critical. Return ONLY the JSON object, nothing else."""

            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=3500,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()
            # Extract JSON from markdown if needed
            cleaned_response = self._extract_json_from_response(response_text)

            try:
                validation_results = json.loads(cleaned_response)
                logger.info("Top provider validation completed")
                return validation_results

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse validation response: {e}")
                return {
                    "top_provider_validations": [],
                    "overall_ranking_validity": {
                        "status": "error",
                        "confidence": "low",
                        "summary": "Validation could not be completed",
                        "improvement_suggestions": []
                    }
                }

        except Exception as e:
            logger.error(f"Top provider validation failed: {e}")
            return {
                "top_provider_validations": [],
                "overall_ranking_validity": {
                    "status": "error",
                    "confidence": "low",
                    "summary": f"Validation error: {str(e)}",
                    "improvement_suggestions": []
                }
            }

    def validate_rankings(self, ranked_providers: List[Dict[str, Any]], preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to critically validate and challenge provider rankings.

        Args:
            ranked_providers: List of ranked provider dictionaries
            preferences: User preferences used for original ranking

        Returns:
            Dictionary containing comprehensive validation results and critiques
        """
        try:
            # Sanitize preferences first
            safe_preferences = InputValidator.validate_preferences(preferences)

            logger.info(f"Starting critical validation of {len(ranked_providers)} ranked providers")

            if not ranked_providers:
                return {
                    "validation_results": {
                        "bias_analysis": {},
                        "alternative_rankings": [],
                        "top_provider_validation": {},
                        "final_recommendations": []
                    },
                    "validation_metadata": {
                        "total_providers_analyzed": 0,
                        "validation_method": "claude_sonnet_critical_analysis",
                        "validation_timestamp": "N/A"
                    },
                    "status": "no_providers",
                    "message": "No providers to validate"
                }

            # Step 1: Analyze potential biases and blind spots
            logger.info("Performing bias analysis...")
            bias_analysis = self._analyze_ranking_bias(ranked_providers, safe_preferences)

            # Step 2: Generate alternative ranking perspectives
            logger.info("Generating alternative ranking scenarios...")
            alternative_rankings = self._generate_alternative_rankings(ranked_providers, safe_preferences)

            # Step 3: Validate top provider recommendations
            logger.info("Validating top provider recommendations...")
            top_validation = self._validate_top_recommendations(ranked_providers)

            # Step 4: Generate final critical recommendations
            final_recommendations = self._generate_final_recommendations(
                ranked_providers, bias_analysis, alternative_rankings, top_validation
            )

            result = {
                "validation_results": {
                    "bias_analysis": bias_analysis,
                    "alternative_rankings": alternative_rankings,
                    "top_provider_validation": top_validation,
                    "final_recommendations": final_recommendations
                },
                "validation_metadata": {
                    "total_providers_analyzed": len(ranked_providers),
                    "validation_method": "claude_sonnet_critical_analysis",
                    "bias_severity": bias_analysis.get("bias_assessment", {}).get("severity", "unknown"),
                    "ranking_confidence": top_validation.get("overall_ranking_validity", {}).get("confidence", "unknown")
                },
                "status": "success",
                "message": f"Critical validation completed for {len(ranked_providers)} providers"
            }

            logger.info(f"Validation completed: {result['message']}")
            return result

        except Exception as e:
            logger.error(f"Rankings validation failed: {e}")
            return {
                "validation_results": {
                    "bias_analysis": {},
                    "alternative_rankings": [],
                    "top_provider_validation": {},
                    "final_recommendations": []
                },
                "validation_metadata": {
                    "total_providers_analyzed": len(ranked_providers),
                    "validation_method": "claude_sonnet_critical_analysis"
                },
                "status": "error",
                "message": f"Error during validation: {str(e)}"
            }

    def _generate_final_recommendations(
        self,
        ranked_providers: List[Dict[str, Any]],
        bias_analysis: Dict[str, Any],
        alternative_rankings: List[Dict[str, Any]],
        top_validation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final recommendations based on all validation analyses.

        Args:
            ranked_providers: Original ranked providers
            bias_analysis: Bias analysis results
            alternative_rankings: Alternative ranking scenarios
            top_validation: Top provider validation results

        Returns:
            Final recommendations dictionary
        """
        try:
            # Determine recommendation confidence
            bias_severity = bias_analysis.get("bias_assessment", {}).get("severity", "medium")
            validation_confidence = top_validation.get("overall_ranking_validity", {}).get("confidence", "medium")

            # Generate final recommendations
            recommendations = {
                "recommendation_confidence": "high" if bias_severity == "low" and validation_confidence == "high" else "medium",
                "key_findings": [],
                "user_guidance": [],
                "provider_recommendations": [],
                "important_considerations": []
            }

            # Add key findings
            if bias_analysis.get("bias_assessment", {}).get("detected_biases"):
                recommendations["key_findings"].append("Detected potential biases in ranking methodology")

            if alternative_rankings:
                recommendations["key_findings"].append(f"Generated {len(alternative_rankings)} alternative ranking perspectives")

            validation_status = top_validation.get("overall_ranking_validity", {}).get("status", "unknown")
            recommendations["key_findings"].append(f"Top provider validation status: {validation_status}")

            # Add user guidance
            if bias_severity in ["medium", "high"]:
                recommendations["user_guidance"].append("Consider the alternative ranking perspectives provided")

            recommendations["user_guidance"].append("Review detailed provider information beyond just rankings")

            if validation_confidence == "low":
                recommendations["user_guidance"].append("Exercise additional caution in provider selection")

            # Add provider recommendations
            for i, provider in enumerate(ranked_providers[:3]):
                validation_info = None
                for val in top_validation.get("top_provider_validations", []):
                    if val.get("rank") == i + 1:
                        validation_info = val
                        break

                provider_rec = {
                    "name": provider.get("name", "Unknown"),
                    "rank": i + 1,
                    "recommendation": "proceed with confidence" if validation_info and validation_info.get("validation_status") == "approved" else "consider carefully",
                    "key_considerations": validation_info.get("patient_considerations", "No specific considerations") if validation_info else "No validation available"
                }
                recommendations["provider_recommendations"].append(provider_rec)

            # Add important considerations
            blind_spots = bias_analysis.get("blind_spots", {}).get("missing_factors", [])
            if blind_spots:
                recommendations["important_considerations"].extend([f"Consider {factor}" for factor in blind_spots[:3]])

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate final recommendations: {e}")
            return {
                "recommendation_confidence": "low",
                "key_findings": ["Error generating recommendations"],
                "user_guidance": ["Manual review recommended"],
                "provider_recommendations": [],
                "important_considerations": []
            }


def create_critic_validator() -> CriticValidatorAgent:
    """Factory function to create a CriticValidatorAgent instance.

    Returns:
        CriticValidatorAgent instance
    """
    return CriticValidatorAgent()