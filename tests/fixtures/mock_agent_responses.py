"""Mock API responses for agent tests."""

MOCK_TAVILY_SEARCH_RESPONSE = {
    "results": [
        {
            "title": "Dr. Emily Carter, Neurologist in Phoenix, AZ - HealthGrades",
            "url": "https://www.healthgrades.com/physician/emily-carter-xyz",
            "content": "Dr. Emily Carter is a neurologist in Phoenix, AZ. She has a 4.8-star rating from 150 reviews. Patients praise her for her listening skills and accurate diagnoses. She accepts BCBS and Aetna. Her office is located at 123 Health St, Phoenix, AZ."
        },
        {
            "title": "Dr. Ben Adams, MD - Neurology Specialist in Scottsdale, AZ",
            "url": "https://www.vitals.com/doctors/dr_ben_adams.html",
            "content": "Dr. Ben Adams is a neurology specialist in nearby Scottsdale, AZ. He has over 20 years of experience. His patients say he is compassionate and thorough. He accepts Medicare. Rating: 4.5 stars from 80 reviews."
        }
    ]
}

MOCK_CLAUDE_EXTRACTION_RESPONSE = """
[
    {
        "name": "Dr. Emily Carter",
        "specialty": "Neurology",
        "location": "123 Health St, Phoenix, AZ",
        "phone": "N/A",
        "rating": 4.8,
        "review_count": 150,
        "review_summary": "Patients praise her for her listening skills and accurate diagnoses.",
        "review_sentiment": "positive",
        "insurance_accepted": ["BCBS", "Aetna"],
        "distance": 2.5
    },
    {
        "name": "Dr. Ben Adams",
        "specialty": "Neurology",
        "location": "Scottsdale, AZ",
        "phone": "N/A",
        "rating": 4.5,
        "review_count": 80,
        "review_summary": "Patients say he is compassionate and thorough.",
        "review_sentiment": "positive",
        "insurance_accepted": ["Medicare"],
        "years_experience": 20,
        "distance": 12.0
    }
]
"""

MOCK_GATHER_PROVIDERS_RESULT = {
    "providers": [
        {
            "name": "Dr. Emily Carter",
            "specialty": "Neurology",
            "location": "123 Health St, Phoenix, AZ",
            "rating": 4.8,
            "insurance_accepted": ["BCBS", "Aetna"],
            "distance": 2.5
        },
        {
            "name": "Dr. Ben Adams",
            "specialty": "Neurology",
            "location": "Scottsdale, AZ",
            "rating": 4.5,
            "insurance_accepted": ["Medicare"],
            "years_experience": 20,
            "distance": 12.0
        }
    ],
    "status": "success"
}

MOCK_OPENAI_RESPONSE = """
[
  {
    "provider_index": 0,
    "scores": {"review_substance": 42, "red_flags": 27, "practical_access": 15},
    "evidence": {
      "review_substance": "Patients praise her for her listening skills and accurate diagnoses.",
      "red_flags": "no evidence",
      "practical_access": "Patients mention easy scheduling and a responsive office."
    },
    "ai_score": 84,
    "reasoning": "Dr. Carter has specific, consistent positive feedback and patients find the office easy to reach.",
    "strengths": ["Excellent ratings", "Responsive office"],
    "concerns": ["None"]
  },
  {
    "provider_index": 1,
    "scores": {"review_substance": 33, "red_flags": 27, "practical_access": 9},
    "evidence": {
      "review_substance": "Patients say he is compassionate and thorough.",
      "red_flags": "no evidence",
      "practical_access": "no evidence"
    },
    "ai_score": 69,
    "reasoning": "Dr. Adams has positive but generic feedback and no access signals either way.",
    "strengths": ["20+ years of experience"],
    "concerns": ["Greater distance"]
  }
]
"""

MOCK_RANKED_PROVIDERS = [
    {
        "name": "Dr. Emily Carter",
        "final_rank": 1,
        "final_score": 95.5,
        "ai_reasoning": "Top choice due to high ratings and close proximity."
    },
    {
        "name": "Dr. Ben Adams",
        "final_rank": 2,
        "final_score": 88.0,
        "ai_reasoning": "Great experience, but farther away."
    }
]

MOCK_BIAS_ANALYSIS_RESPONSE = """
{
  "bias_assessment": {
    "detected_biases": ["Proximity bias"],
    "severity": "low",
    "explanation": "The ranking slightly favors closer providers."
  },
  "blind_spots": {
      "missing_factors": ["Telehealth options"],
      "impact": "Could be important for some users.",
      "recommendations": ["Consider telehealth availability."]
  },
  "alternative_scenarios": [],
  "validity_concerns": {
      "ranking_issues": [],
      "misleading_aspects": [],
      "confidence_level": "high"
  },
  "overall_assessment": "The ranking is solid, with minor proximity bias."
}
"""

MOCK_ALTERNATIVE_RANKINGS_RESPONSE = """
[
  {
    "scenario_name": "Quality-First Perspective",
    "description": "Focus on clinical excellence over convenience",
    "reranked_providers": [
      {
        "name": "Dr. Ben Adams",
        "new_rank": 1,
        "original_rank": 2,
        "reasoning": "His 20+ years of experience are prioritized."
      }
    ]
  }
]
"""

MOCK_VALIDATION_RESPONSE = """
{
  "top_provider_validations": [
    {
      "provider_name": "Dr. Emily Carter",
      "rank": 1,
      "validation_status": "approved",
      "confidence_in_recommendation": "high",
      "validation_notes": "The high rating and positive reviews support this ranking."
    }
  ],
  "overall_ranking_validity": {
    "status": "validated",
    "confidence": "high",
    "summary": "The ranking is well-supported by the available data."
  }
}
"""

MOCK_SCORED_PROVIDERS_RESULT = {
    "ranked_providers": [
        {
            "name": "Dr. Emily Carter",
            "final_rank": 1,
            "final_score": 95.5,
            "ai_reasoning": "Top choice due to high ratings and close proximity.",
            # A scored provider carries the judge's rubric and the enrichment
            # outcome; a recommendation asserts both, so a fixture without them
            # is not a scored provider.
            "enrichment_outcome": "enriched",
            "ai_rubric": {"review_substance": 45.0, "red_flags": 28.0, "practical_access": 15.0},
        },
        {
            "name": "Dr. Ben Adams",
            "final_rank": 2,
            "final_score": 88.0,
            "ai_reasoning": "Great experience, but farther away.",
            "enrichment_outcome": "enriched",
            "ai_rubric": {"review_substance": 38.0, "red_flags": 26.0, "practical_access": 12.0},
        }
    ],
    "scoring_metadata": {
        "total_providers": 2,
        "preferences_used": {
            "location_weight": 0.4,
            "rating_weight": 0.3,
            "insurance_priority": 0.3
        },
        "scoring_method": "weighted_algorithm_with_ai",
        "top_provider": "Dr. Emily Carter",
        "score_range": {
            "highest": 95.5,
            "lowest": 88.0
        }
    },
    "status": "success",
    "message": "Successfully ranked 2 providers"
}

MOCK_VALIDATION_RESULT = {
    "validation_results": {
        "bias_analysis": {
            "bias_assessment": {
                "detected_biases": ["Proximity bias"],
                "severity": "low",
                "explanation": "The ranking slightly favors closer providers."
            },
            "blind_spots": {
                "missing_factors": ["Telehealth options"],
                "impact": "Could be important for some users.",
                "recommendations": ["Consider telehealth availability."]
            },
            "alternative_scenarios": [],
            "validity_concerns": {
                "ranking_issues": [],
                "misleading_aspects": [],
                "confidence_level": "high"
            },
            "overall_assessment": "The ranking is solid, with minor proximity bias."
        },
        "alternative_rankings": [],
        "top_provider_validation": {
            "top_provider_validations": [
                {
                    "provider_name": "Dr. Emily Carter",
                    "rank": 1,
                    "validation_status": "approved",
                    "confidence_in_recommendation": "high",
                    "validation_notes": "The high rating and positive reviews support this ranking."
                }
            ],
            "overall_ranking_validity": {
                "status": "validated",
                "confidence": "high",
                "summary": "The ranking is well-supported by the available data."
            }
        },
        "final_recommendations": {}
    },
    "validation_metadata": {
        "total_providers_analyzed": 2,
        "validation_method": "claude_sonnet_critical_analysis",
        "bias_severity": "low",
        "ranking_confidence": "high"
    },
    "status": "success",
    "message": "Critical validation completed for 2 providers"
}
