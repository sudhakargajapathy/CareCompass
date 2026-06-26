"""Mock API responses for Tavily, OpenAI, and Anthropic."""
import json


MOCK_TAVILY_SEARCH_RESULTS = [
    {
        "title": "Dr. Sarah Johnson, MD - Neurologist | Phoenix, AZ",
        "url": "https://www.healthgrades.com/physician/dr-sarah-johnson-xyz123",
        "content": "Dr. Sarah Johnson is a board-certified neurologist in Phoenix, AZ. Specializes in migraine treatment and epilepsy care. Rating: 4.8/5 from 127 patient reviews. Accepts Aetna, Blue Cross Blue Shield, UnitedHealthcare, Medicare. Phone: (602) 555-1234. Patients praise her thorough explanations and compassionate care.",
        "score": 0.95
    },
    {
        "title": "Sarah Johnson MD - Neurology Specialist Reviews",
        "url": "https://www.vitals.com/doctors/Dr_Sarah_Johnson.html",
        "content": "Dr. Sarah Johnson - Neurology specialist serving Phoenix area for 15 years. Expertise in migraine treatment, epilepsy, and MS management. 127 verified patient reviews with 4.8 average rating. Many note her expertise and ability to listen carefully to concerns. Located 2.3 miles from downtown Phoenix.",
        "score": 0.92
    },
    {
        "title": "Dr. Michael Chen - Neurologist Scottsdale",
        "url": "https://www.zocdoc.com/doctor/michael-chen-md",
        "content": "Michael Chen, MD - Neurology. Scottsdale, AZ 85251. Phone: (480) 555-5678. Rating 4.5/5 (89 reviews). Specializes in stroke prevention, Parkinson's disease, dementia evaluation. Accepts most major insurance. Praised for diagnostic skills and modern treatment approaches. 12 years experience.",
        "score": 0.89
    }
]


MOCK_CLAUDE_EXTRACTION_RESPONSE = {
    "id": "msg_01ABC123",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": json.dumps([
                {
                    "name": "Dr. Sarah Johnson",
                    "specialty": "Neurology",
                    "location": "Phoenix, AZ",
                    "phone": "(602) 555-1234",
                    "rating": 4.8,
                    "review_count": 127,
                    "review_summary": "Patients consistently praise Dr. Johnson for her thorough explanations and compassionate care. Many note her expertise in migraine treatment and her ability to listen carefully to concerns. Some mention longer wait times but feel the quality of care is worth it.",
                    "review_sentiment": "positive",
                    "insurance_accepted": "Accepts Aetna, Blue Cross Blue Shield, UnitedHealthcare, Medicare",
                    "distance": "2.3 miles",
                    "years_experience": 15,
                    "services": ["Migraine treatment", "Epilepsy care", "Multiple sclerosis management"]
                },
                {
                    "name": "Dr. Michael Chen",
                    "specialty": "Neurology",
                    "location": "Scottsdale, AZ",
                    "phone": "(480) 555-5678",
                    "rating": 4.5,
                    "review_count": 89,
                    "review_summary": "Dr. Chen is praised for his diagnostic skills and modern treatment approaches. Patients appreciate his use of cutting-edge technology. A few reviews mention communication could be more detailed, but overall satisfaction is high.",
                    "review_sentiment": "positive",
                    "insurance_accepted": "Accepts most major insurance plans",
                    "distance": "5.1 miles",
                    "years_experience": 12,
                    "services": ["Stroke prevention", "Parkinson's disease", "Dementia evaluation"]
                }
            ])
        }
    ],
    "model": "claude-3-5-haiku-20241022",
    "usage": {"input_tokens": 500, "output_tokens": 300}
}


MOCK_CLAUDE_INVALID_JSON_RESPONSE = {
    "id": "msg_01XYZ789",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Here is the provider data: {invalid json that won't parse..."
        }
    ],
    "model": "claude-3-5-haiku-20241022",
    "usage": {"input_tokens": 500, "output_tokens": 50}
}


MOCK_GPT_RANKING_RESPONSE = {
    "id": "chatcmpl-ABC123",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": json.dumps({
                    "rankings": [
                        {
                            "provider_name": "Dr. Lisa Patel",
                            "rank": 1,
                            "reasoning": "Highest rating (4.9) with extensive reviews (156), comprehensive insurance coverage, and strong patient feedback on communication and thoroughness."
                        },
                        {
                            "provider_name": "Dr. Sarah Johnson",
                            "rank": 2,
                            "reasoning": "Excellent rating (4.8) with good review count (127), closest location (2.3 miles), and specialized migraine expertise matching patient needs."
                        },
                        {
                            "provider_name": "Dr. Michael Chen",
                            "rank": 3,
                            "reasoning": "Strong rating (4.5) with modern treatment approaches, though slightly farther and fewer reviews than top choices."
                        }
                    ],
                    "overall_confidence": "high"
                })
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {"prompt_tokens": 800, "completion_tokens": 200, "total_tokens": 1000}
}


MOCK_CLAUDE_VALIDATION_RESPONSE = {
    "id": "msg_01VALIDATION",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": json.dumps({
                "bias_analysis": {
                    "location_bias": "Moderate - 60% of top providers within 5 miles, potentially overlooking excellent distant options",
                    "rating_bias": "Low - Good distribution of ratings with consideration of review counts",
                    "insurance_bias": "Present - Strong preference for comprehensive insurance acceptance may exclude quality providers with limited networks",
                    "blind_spots": ["Age diversity of providers not considered", "Availability/wait times not factored"]
                },
                "alternative_rankings": [
                    {
                        "perspective": "Quality-First",
                        "top_providers": ["Dr. Lisa Patel", "Dr. Sarah Johnson", "Dr. Michael Chen"],
                        "rationale": "Prioritizes highest ratings and review counts regardless of distance or insurance"
                    },
                    {
                        "perspective": "Accessibility-First",
                        "top_providers": ["Dr. Sarah Johnson", "Dr. Michael Chen", "Dr. Emily Rodriguez"],
                        "rationale": "Optimizes for proximity and insurance match, accepting slightly lower ratings"
                    }
                ],
                "validation_notes": {
                    "Dr. Lisa Patel": "Exceptional credentials and reviews justify top ranking despite distance",
                    "Dr. Sarah Johnson": "Optimal balance of quality, location, and insurance coverage",
                    "Dr. Michael Chen": "Strong technical skills but communication concerns noted in reviews"
                },
                "overall_confidence": "high",
                "recommendations": [
                    "Consider scheduling availability as a future ranking factor",
                    "May want to validate insurance acceptance directly with providers",
                    "Top 3 recommendations appear well-justified based on available data"
                ]
            })
        }
    ],
    "model": "claude-sonnet-4-20250514",
    "usage": {"input_tokens": 1200, "output_tokens": 400}
}


MOCK_CLAUDE_VALIDATION_MARKDOWN_WRAPPED = {
    "id": "msg_01MARKDOWN",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": """```json
{
    "bias_analysis": {
        "location_bias": "Low",
        "rating_bias": "Low"
    },
    "overall_confidence": "medium"
}
```"""
        }
    ],
    "model": "claude-sonnet-4-20250514",
    "usage": {"input_tokens": 1000, "output_tokens": 100}
}
