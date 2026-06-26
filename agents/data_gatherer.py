"""Data Gatherer Agent for collecting healthcare provider information using Tavily search and FHIR."""

import logging
from typing import Dict, List, Any, Optional
import json
import re
from tavily import TavilyClient
from anthropic import Anthropic

from utils.config import get_config
from utils.security import PromptSanitizer, validate_search_params

logger = logging.getLogger(__name__)

# Token-overlap threshold for name matching during FHIR/Tavily merge
_NAME_MATCH_THRESHOLD = 0.5


class DataGathererAgent:
    """Agent responsible for gathering healthcare provider data using Tavily search and Claude Haiku for extraction."""

    def __init__(self):
        """Initialize the data gatherer with Tavily, Anthropic, and optionally FHIR clients."""
        self.config = get_config()
        self.tavily_client = None
        self.anthropic_client = None
        self.fhir_client = None
        self._fhir_transformer = None
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Initialize Tavily, Anthropic, and optionally FHIR clients."""
        try:
            if not self.config.TAVILY_API_KEY:
                raise ValueError("Tavily API key not found in configuration")
            if not self.config.ANTHROPIC_API_KEY:
                raise ValueError("Anthropic API key not found in configuration")

            self.tavily_client = TavilyClient(api_key=self.config.TAVILY_API_KEY)
            self.anthropic_client = Anthropic(api_key=self.config.ANTHROPIC_API_KEY)

            # Initialize FHIR client when enabled
            if self.config.FHIR_ENABLED:
                try:
                    from fhir.client import create_fhir_client
                    from fhir.transformer import FHIRToProviderTransformer

                    self.fhir_client = create_fhir_client()
                    self._fhir_transformer = FHIRToProviderTransformer()
                    logger.info("FHIR client initialized (mock=%s)", self.config.FHIR_USE_MOCK)
                except Exception as fhir_err:
                    logger.warning("FHIR client initialization failed, continuing without FHIR: %s", fhir_err)
                    self.fhir_client = None

            logger.info("Data gatherer clients initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize data gatherer clients: {e}")
            raise

    def _build_search_query(self, specialty: str, location: str, insurance: Optional[str] = None) -> str:
        """Build optimized search query for healthcare providers.

        Args:
            specialty: Medical specialty (e.g., "Neurology")
            location: Location (e.g., "Phoenix, AZ")
            insurance: Insurance type (optional)

        Returns:
            Optimized search query string
        """
        query_parts = []

        # Core search terms with review focus
        query_parts.append(f"{specialty} specialists in {location}")
        query_parts.append("reviews ratings patient feedback")

        # Add insurance if specified
        if insurance:
            query_parts.append(f"accepts {insurance}")

        query = " ".join(query_parts)
        logger.debug(f"Built search query: {query}")
        return query

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

        # If no code blocks, try to find JSON array [...]
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            return json_match.group(0).strip()

        return response_text.strip()

    def _search_providers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for providers using Tavily API.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of search results from Tavily
        """
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                # Removed include_domains to let Tavily find all relevant sources including Google
                include_answer=False
            )

            results = response.get("results", [])
            logger.info(f"Found {len(results)} search results for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []

    def _extract_provider_data(self, search_results: List[Dict[str, Any]], specialty: str, location: str) -> List[Dict[str, Any]]:
        """Extract structured provider data using Claude Haiku.

        Args:
            search_results: Raw search results from Tavily
            specialty: Requested specialty
            location: Requested location

        Returns:
            List of structured provider data dictionaries
        """
        try:
            # Sanitize inputs to prevent prompt injection
            safe_specialty = PromptSanitizer.escape_for_prompt(specialty)
            safe_location = PromptSanitizer.escape_for_prompt(location)

            # Get top 20 results from Tavily (already ranked by relevance)
            top_results = search_results[:20]

            # Prioritize review-heavy sources (Yelp, Healthgrades, Vitals have detailed patient feedback)
            review_heavy_domains = ['healthgrades.com', 'vitals.com', 'zocdoc.com', 'webmd.com', 'google.com', 'yelp.com']

            review_heavy = [r for r in top_results if any(domain in r.get('url', '').lower() for domain in review_heavy_domains)]
            other_results = [r for r in top_results if not any(domain in r.get('url', '').lower() for domain in review_heavy_domains)]

            # Prioritize review sources first (most relevant), then others, up to 18 total for more review content
            prioritized_results = (review_heavy + other_results)[:18]

            # Prepare content for Claude with clear delimiters
            results_text = "\n\n".join([
                f"Title: {result.get('title', '')}\nURL: {result.get('url', '')}\nContent: {result.get('content', '')}"
                for result in prioritized_results
            ])

            # Use structured prompt with XML-style delimiters for clarity
            prompt = f"""Extract healthcare provider information from the search results provided below.

You are a data extraction specialist. Your task is to extract healthcare provider information ONLY from the search results section.

<task_parameters>
Target Specialty: {safe_specialty}
Target Location: {safe_location}
</task_parameters>

<output_format>
Return ONLY a JSON array of provider objects with these fields:

REQUIRED FIELDS:
- name: Provider's full name
- specialty: Medical specialty
- location: Full address or city/state
- phone: Phone number (format: XXX-XXX-XXXX, or "N/A" if not found)
- rating: Rating out of 5 (numeric, e.g., 4.5, or 0 if not found)
- review_count: Number of patient reviews (integer, e.g., 127, or null if not found)
- review_summary: 3-4 sentence summary of patient feedback covering most praised aspects, common complaints, and overall experience themes (string, or "No reviews available" if no review content found)
- review_sentiment: Overall sentiment from reviews: "positive", "mixed", or "negative" (or "unknown" if no review content available)

HIGHLY IMPORTANT FIELDS (extract if ANY information is available):
- insurance_accepted: List of insurance names/types mentioned (e.g., ["Blue Cross", "Aetna", "Medicare"])
  * Search carefully for insurance mentions in the content
  * Look for phrases like "accepts", "takes", "insurance plans"
  * Even partial matches are valuable
  * If no insurance info found, use empty array []

- distance: Distance in miles from {location} (if mentioned or calculable from address)
  * Look for explicit distance mentions
  * If provider address is in same city as {location}, estimate 0-5 miles
  * If in different city but same metro area, estimate 10-30 miles
  * If completely unknown, use null (not "N/A")

OPTIONAL FIELDS:
- services: List of services offered
- website: Provider's website URL
- years_experience: Years in practice (if mentioned)
- education: Medical school/credentials (if mentioned)
</output_format>

<extraction_rules>
1. Only extract providers matching the target specialty
2. Only extract providers in or near the target location
3. Extract phone numbers in XXX-XXX-XXXX format
4. Convert ratings to numeric format (e.g., "4.5 stars" → 4.5)
5. CRITICAL: Search thoroughly for insurance and distance information - these are key factors
6. For insurance: even mentions like "accepts most major insurance" → ["Most major insurance"]
7. REVIEW ANALYSIS - CRITICAL INSTRUCTIONS:
   * ONLY extract review summaries if you find ACTUAL PATIENT FEEDBACK in the search results
   * Look for quotes, comments, or testimonials from patients (e.g., "very caring", "long wait", "listens well")
   * DO NOT make up generic statements like "has experience in X" - that's not a review summary
   * MANDATORY: If you find actual patient comments, create a comprehensive 3-4 sentence summary covering:
     1. What patients most appreciate (strengths like expertise, bedside manner, communication)
     2. What patients complain about or areas for improvement (concerns like wait times, billing, accessibility)
     3. Overall experience patterns and recommendations (professionalism, office environment, staff helpfulness)
   * Determine sentiment: "positive" (mostly good), "mixed" (both good and bad), or "negative" (mostly bad)
   * If NO actual patient review content found, use "No reviews available" for summary and "unknown" for sentiment
8. Return ONLY a JSON array of provider objects
9. If no clear providers found, return empty array []
10. Do NOT include any explanatory text, ONLY return the JSON array
</extraction_rules>

<search_results>
{results_text}
</search_results>

Response (JSON array only):"""

            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            # Extract JSON from response using helper
            try:
                # Extract JSON from markdown or raw response
                json_str = self._extract_json_from_response(response_text)

                # Try to parse JSON
                try:
                    providers = json.loads(json_str)
                except json.JSONDecodeError as parse_error:
                    # Try to fix common JSON issues
                    logger.warning(f"Initial JSON parse failed: {parse_error}. Attempting to fix...")

                    fixed_json = json_str

                    # Fix 1: Remove trailing commas before closing brackets/braces
                    fixed_json = re.sub(r',\s*([}\]])', r'\1', fixed_json)

                    # Fix 2: Remove trailing comma at end of arrays/objects (more aggressive)
                    fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)

                    # Fix 3: Try to extract only the JSON array if there's text after it
                    array_match = re.search(r'(\[[\s\S]*?\])\s*[^,}\]]*$', fixed_json)
                    if array_match:
                        fixed_json = array_match.group(1)

                    # Try parsing the fixed version
                    try:
                        providers = json.loads(fixed_json)
                        logger.info("Successfully parsed after fixing JSON syntax")
                    except json.JSONDecodeError as second_error:
                        # If still fails, log more details and return empty
                        logger.error(f"Could not recover from JSON error after fixing: {second_error}")
                        logger.debug(f"Original error: {parse_error}")
                        logger.debug(f"Response preview: {response_text[:1000]}...")
                        logger.debug(f"Fixed JSON preview: {fixed_json[:1000]}...")
                        return []

                # Validate that it's a list
                if not isinstance(providers, list):
                    logger.warning("Claude response was not a JSON array")
                    return []

                # Clean and validate provider data
                cleaned_providers = []
                for provider in providers:
                    if isinstance(provider, dict) and provider.get("name"):
                        # Safe float conversion helper
                        def safe_float(value, default=0.0):
                            """Convert value to float, handling 'N/A' and invalid values."""
                            if value is None or value == "" or str(value).upper() == "N/A":
                                return default
                            try:
                                return float(value)
                            except (ValueError, TypeError):
                                return default

                        # Ensure required fields
                        cleaned_provider = {
                            "name": str(provider.get("name", "")),
                            "specialty": str(provider.get("specialty", specialty)),
                            "location": str(provider.get("location", location)),
                            "phone": str(provider.get("phone", "")),
                            "rating": safe_float(provider.get("rating"), 0.0),
                        }

                        # Handle review_count with smart fallback
                        review_count = provider.get("review_count")
                        if review_count is not None:
                            cleaned_provider["review_count"] = int(review_count) if review_count else None
                        else:
                            # Smart fallback: if rating exists but review_count is missing
                            if cleaned_provider["rating"] > 0:
                                # Check if source URL contains google for higher default
                                source_url = str(provider.get("source_url", "")).lower()
                                if 'google' in source_url:
                                    cleaned_provider["review_count"] = 50  # Google typically has more reviews
                                else:
                                    cleaned_provider["review_count"] = 25  # Other medical sites
                            else:
                                cleaned_provider["review_count"] = None

                        # Add review summary and sentiment
                        cleaned_provider["review_summary"] = str(provider.get("review_summary", "No reviews available"))
                        cleaned_provider["review_sentiment"] = str(provider.get("review_sentiment", "unknown"))

                        # Add optional fields
                        for field in ["insurance_accepted", "services", "website", "years_experience", "education"]:
                            if provider.get(field):
                                cleaned_provider[field] = provider[field]

                        # Handle distance separately with safe float conversion
                        if provider.get("distance"):
                            cleaned_provider["distance"] = safe_float(provider.get("distance"), None)

                        cleaned_providers.append(cleaned_provider)

                logger.info(f"Extracted {len(cleaned_providers)} providers from search results")
                return cleaned_providers

            except Exception as inner_error:
                logger.error(f"Error processing provider data: {inner_error}")
                return []

        except Exception as e:
            logger.error(f"Provider data extraction failed: {e}")
            return []

    def _extract_review_data_only(self, search_results: List[Dict[str, Any]], provider_name: str) -> Dict[str, Any]:
        """Extract only review-related data for a specific provider using Claude Haiku.

        Args:
            search_results: Search results from Tavily
            provider_name: Name of the provider to extract reviews for

        Returns:
            Dictionary with review_summary, review_sentiment, and review_count
        """
        try:
            # Prepare content for Claude
            results_text = "\n\n".join([
                f"Title: {result.get('title', '')}\nURL: {result.get('url', '')}\nContent: {result.get('content', '')}"
                for result in search_results[:10]
            ])

            prompt = f"""Extract ONLY review information for the healthcare provider "{provider_name}" from the search results below.
CRITICAL: Return ONLY a valid JSON object with review data.

Return a JSON object with these fields ONLY:
- review_summary: 3-4 sentence summary of patient feedback covering most praised aspects, common complaints, and overall experience themes (or "No reviews available" if no actual patient feedback found)
- review_sentiment: Overall sentiment: "positive", "mixed", "negative", or "unknown"
- review_count: Number of reviews found (integer, or null if not found)

IMPORTANT REVIEW EXTRACTION RULES:
1. ONLY extract if you find ACTUAL PATIENT FEEDBACK (quotes, comments, testimonials)
2. DO NOT make up generic statements - they must be based on real patient comments
3. Look for specific feedback like: "great bedside manner", "long wait times", "very thorough", etc.
4. If you find patient feedback, create a comprehensive 3-4 sentence summary covering:
   - What patients praise (expertise, communication, etc.)
   - What patients complain about (wait times, billing, etc.)
   - Overall experience patterns
5. If NO actual patient review content found, use "No reviews available" and "unknown"

SEARCH RESULTS:
{results_text}

Response (JSON object only):"""

            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            # Extract JSON from response
            json_str = self._extract_json_from_response(response_text)

            # Try to find JSON object
            obj_match = re.search(r'\{[\s\S]*\}', json_str)
            if obj_match:
                json_str = obj_match.group(0)

            review_data = json.loads(json_str)

            if isinstance(review_data, dict):
                return {
                    "review_summary": str(review_data.get("review_summary", "No reviews available")),
                    "review_sentiment": str(review_data.get("review_sentiment", "unknown")),
                    "review_count": review_data.get("review_count")
                }

        except Exception as e:
            logger.warning(f"Failed to extract review data for {provider_name}: {e}")

        return {
            "review_summary": "No reviews available",
            "review_sentiment": "unknown",
            "review_count": None
        }

    def _merge_review_data(self, provider: Dict[str, Any], review_data: Dict[str, Any]) -> None:
        """Merge secondary review data into provider object only if actual reviews found.

        Args:
            provider: Provider dictionary to update (modified in place)
            review_data: Review data extracted from secondary search
        """
        # Only merge if we found actual review content
        if review_data.get("review_summary") and review_data["review_summary"] != "No reviews available":
            provider["review_summary"] = review_data["review_summary"]
            logger.info(f"Enriched reviews for {provider.get('name')}")

        if review_data.get("review_sentiment") and review_data["review_sentiment"] != "unknown":
            provider["review_sentiment"] = review_data["review_sentiment"]

        # Update review count if found and provider doesn't have one
        if review_data.get("review_count") and not provider.get("review_count"):
            provider["review_count"] = review_data["review_count"]

    def _enrich_missing_reviews(self, providers: List[Dict[str, Any]], location: str) -> List[Dict[str, Any]]:
        """Perform follow-up searches for providers with missing reviews.

        Args:
            providers: List of provider dictionaries
            location: Location for search context

        Returns:
            Updated list of providers with enriched review data
        """
        # Find providers missing reviews
        candidates = [
            p for p in providers
            if p.get("review_summary") == "No reviews available"
            or p.get("review_sentiment") == "unknown"
        ][:self.config.MAX_PROVIDERS_TO_ENRICH]

        if not candidates:
            logger.info("No providers need review enrichment")
            return providers

        logger.info(f"Enriching reviews for {len(candidates)} providers")

        for provider in candidates:
            try:
                provider_name = provider.get("name", "")
                if not provider_name:
                    continue

                # Build targeted review search query
                query = f"{provider_name} reviews {location}"
                logger.debug(f"Review enrichment search: {query}")

                # Search with smaller result set for efficiency
                results = self._search_providers(query, max_results=5)

                if results:
                    review_data = self._extract_review_data_only(results, provider_name)
                    self._merge_review_data(provider, review_data)

            except Exception as e:
                logger.warning(f"Could not enrich reviews for {provider.get('name')}: {e}")

        return providers

    # ------------------------------------------------------------------
    # FHIR integration methods
    # ------------------------------------------------------------------

    def _gather_fhir_providers(
        self, specialty: str, location: str, insurance: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query FHIR Provider Directory and return transformed provider dicts.

        Args:
            specialty: Medical specialty name
            location: City, State string
            insurance: Insurance network name (optional)

        Returns:
            List of provider dicts from FHIR data
        """
        if not self.fhir_client or not self._fhir_transformer:
            return []

        try:
            bundle = self.fhir_client.search_practitioners(
                specialty=specialty,
                location=location,
                insurance_network=insurance,
            )
            providers = self._fhir_transformer.transform_bundle(bundle)
            logger.info("FHIR returned %d providers for %s in %s", len(providers), specialty, location)
            return providers
        except Exception as e:
            logger.warning("FHIR provider gathering failed: %s", e)
            return []

    @staticmethod
    def _name_token_overlap(name_a: str, name_b: str) -> float:
        """Calculate token-overlap ratio between two provider names.

        Strips common prefixes (Dr., MD, DO) and compares remaining tokens.

        Args:
            name_a: First provider name
            name_b: Second provider name

        Returns:
            Overlap ratio between 0.0 and 1.0
        """
        strip_words = {"dr", "dr.", "md", "do", "phd", "jr", "sr", "ii", "iii"}

        def tokenize(name: str) -> set[str]:
            tokens = {t.lower().strip(".,") for t in name.split()}
            return tokens - strip_words

        tokens_a = tokenize(name_a)
        tokens_b = tokenize(name_b)

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        smaller = min(len(tokens_a), len(tokens_b))
        return len(intersection) / smaller if smaller else 0.0

    def _merge_fhir_and_tavily_providers(
        self,
        fhir_providers: List[Dict[str, Any]],
        tavily_providers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge FHIR and Tavily provider lists by name similarity.

        - Matched: FHIR identity/insurance + Tavily ratings/reviews
        - FHIR-only: kept with data_source="fhir"
        - Tavily-only: kept with data_source="tavily"

        Args:
            fhir_providers: Providers from FHIR directory
            tavily_providers: Providers from Tavily web search

        Returns:
            Merged list of unique providers
        """
        merged: List[Dict[str, Any]] = []
        matched_tavily_indices: set[int] = set()

        for fhir_p in fhir_providers:
            best_match_idx: Optional[int] = None
            best_score = 0.0

            for i, tavily_p in enumerate(tavily_providers):
                if i in matched_tavily_indices:
                    continue
                score = self._name_token_overlap(
                    fhir_p.get("name", ""), tavily_p.get("name", "")
                )
                if score > best_score:
                    best_score = score
                    best_match_idx = i

            if best_score >= _NAME_MATCH_THRESHOLD and best_match_idx is not None:
                # Merge: FHIR identity + Tavily ratings/reviews
                matched_tavily_indices.add(best_match_idx)
                tavily_p = tavily_providers[best_match_idx]

                combined = fhir_p.copy()
                # Overlay Tavily review data (FHIR has none)
                combined["rating"] = tavily_p.get("rating", 0.0)
                combined["review_count"] = tavily_p.get("review_count")
                combined["review_summary"] = tavily_p.get("review_summary", "No reviews available")
                combined["review_sentiment"] = tavily_p.get("review_sentiment", "unknown")
                # Tavily may have distance info
                if tavily_p.get("distance") is not None:
                    combined["distance"] = tavily_p["distance"]
                # Keep FHIR insurance but note Tavily matches too
                combined["data_source"] = "fhir+tavily"
                if combined.get("fhir_metadata"):
                    combined["fhir_metadata"]["tavily_matched"] = True
                logger.info(
                    "Merged FHIR+Tavily for %s (overlap=%.2f)",
                    combined["name"], best_score,
                )
                merged.append(combined)
            else:
                # FHIR-only provider
                merged.append(fhir_p)

        # Add unmatched Tavily providers
        for i, tavily_p in enumerate(tavily_providers):
            if i not in matched_tavily_indices:
                tavily_p["data_source"] = "tavily"
                merged.append(tavily_p)

        logger.info(
            "Merge result: %d FHIR, %d Tavily -> %d merged (%d matched)",
            len(fhir_providers),
            len(tavily_providers),
            len(merged),
            len(matched_tavily_indices),
        )
        return merged

    def gather_providers(self, specialty: str, location: str, insurance: Optional[str] = None) -> Dict[str, Any]:
        """Main method to gather healthcare provider data.

        Args:
            specialty: Medical specialty to search for
            location: Location to search in
            insurance: Insurance type filter (optional)

        Returns:
            Dictionary containing providers list, search metadata, and status
        """
        try:
            # Validate and sanitize inputs first
            validation_result = validate_search_params(specialty, location, insurance)

            if not validation_result["is_valid"]:
                errors = [e for e in validation_result["errors"] if e]
                logger.warning(f"Invalid search parameters: {errors}")
                return {
                    "providers": [],
                    "search_metadata": {
                        "query": "",
                        "specialty": specialty,
                        "location": location,
                        "insurance": insurance,
                        "total_found": 0,
                        "validation_errors": errors
                    },
                    "status": "invalid_input",
                    "message": f"Invalid search parameters: {', '.join(errors)}"
                }

            # Use sanitized values
            safe_specialty = validation_result["specialty"]
            safe_location = validation_result["location"]
            safe_insurance = validation_result["insurance"]

            logger.info(f"Starting provider search: {safe_specialty} in {safe_location}")

            # --- Step 1: FHIR directory lookup (if enabled) ---
            fhir_providers: List[Dict[str, Any]] = []
            fhir_count = 0
            if self.fhir_client:
                fhir_providers = self._gather_fhir_providers(
                    safe_specialty, safe_location, safe_insurance
                )
                fhir_count = len(fhir_providers)

            # --- Step 2: Tavily web search ---
            query = self._build_search_query(safe_specialty, safe_location, safe_insurance)
            search_results = self._search_providers(query, max_results=self.config.MAX_PROVIDERS_PER_SEARCH)

            tavily_providers: List[Dict[str, Any]] = []
            if search_results:
                tavily_providers = self._extract_provider_data(search_results, safe_specialty, safe_location)
                if tavily_providers:
                    tavily_providers = self._enrich_missing_reviews(tavily_providers, location)

            # --- Step 3: Merge FHIR + Tavily ---
            if fhir_providers and tavily_providers:
                providers = self._merge_fhir_and_tavily_providers(fhir_providers, tavily_providers)
            elif fhir_providers:
                providers = fhir_providers
            else:
                providers = tavily_providers

            if not providers:
                return {
                    "providers": [],
                    "search_metadata": {
                        "query": query,
                        "specialty": safe_specialty,
                        "location": safe_location,
                        "insurance": safe_insurance,
                        "total_found": 0,
                        "fhir_count": fhir_count,
                    },
                    "status": "no_results",
                    "message": "No providers found for the specified criteria"
                }

            # Filter by insurance if specified (for non-FHIR providers only,
            # since FHIR providers are already network-verified)
            if safe_insurance and providers:
                filtered_providers = []
                for provider in providers:
                    # Keep all FHIR-sourced providers (already network-verified)
                    if provider.get("data_source") in ("fhir", "fhir+tavily"):
                        filtered_providers.append(provider)
                        continue
                    insurance_list = provider.get("insurance_accepted", [])
                    if isinstance(insurance_list, list):
                        insurance_text = " ".join(insurance_list).lower()
                        if safe_insurance.lower() in insurance_text:
                            filtered_providers.append(provider)
                providers = filtered_providers

            result = {
                "providers": providers,
                "search_metadata": {
                    "query": query,
                    "specialty": safe_specialty,
                    "location": safe_location,
                    "insurance": safe_insurance,
                    "total_found": len(providers),
                    "search_results_count": len(search_results),
                    "fhir_count": fhir_count,
                    "tavily_count": len(tavily_providers),
                    "fhir_enabled": self.fhir_client is not None,
                },
                "status": "success" if providers else "no_providers_extracted",
                "message": f"Found {len(providers)} {safe_specialty} providers in {safe_location}"
            }

            logger.info(f"Data gathering completed: {result['message']}")
            return result

        except Exception as e:
            logger.error(f"Provider gathering failed: {e}", exc_info=True)
            return {
                "providers": [],
                "search_metadata": {
                    "query": "",
                    "specialty": specialty,
                    "location": location,
                    "insurance": insurance,
                    "total_found": 0
                },
                "status": "error",
                "message": "Error gathering provider data. Please try again."
            }


def create_data_gatherer() -> DataGathererAgent:
    """Factory function to create a DataGathererAgent instance.

    Returns:
        DataGathererAgent instance
    """
    return DataGathererAgent()