"""Sample provider data for testing."""

SAMPLE_PROVIDERS = [
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
    },
    {
        "name": "Dr. Emily Rodriguez",
        "specialty": "Neurology",
        "location": "Tempe, AZ",
        "phone": "(480) 555-9012",
        "rating": 4.2,
        "review_count": 45,
        "review_summary": "Patients value Dr. Rodriguez's attention to detail and follow-up care. Some note scheduling can be challenging due to high demand. Mixed feedback on bedside manner, though clinical competence is consistently praised.",
        "review_sentiment": "mixed",
        "insurance_accepted": "Cigna, Humana",
        "distance": "8.7 miles",
        "years_experience": 8,
        "services": ["Headache disorders", "Neuropathy treatment"]
    },
    {
        "name": "Dr. Robert Williams",
        "specialty": "Neurology",
        "location": "Mesa, AZ",
        "phone": "(480) 555-3456",
        "rating": 3.9,
        "review_count": 23,
        "review_summary": "Some patients report positive experiences with diagnosis accuracy, while others express concerns about rushed appointments. Clinical knowledge appears strong but communication style varies in patient feedback.",
        "review_sentiment": "mixed",
        "insurance_accepted": "Limited insurance accepted",
        "distance": "12.4 miles",
        "years_experience": 20,
        "services": ["General neurology"]
    },
    {
        "name": "Dr. Lisa Patel",
        "specialty": "Neurology",
        "location": "Chandler, AZ",
        "phone": "(480) 555-7890",
        "rating": 4.9,
        "review_count": 156,
        "review_summary": "Dr. Patel receives exceptional reviews for her patient-centered approach and clear communication. Patients highlight her thoroughness in testing and her ability to explain complex conditions simply. Wait times are reasonable and staff is friendly.",
        "review_sentiment": "positive",
        "insurance_accepted": "Accepts Aetna, Cigna, Blue Cross Blue Shield, UnitedHealthcare, Humana, Medicare, Medicaid",
        "distance": "15.2 miles",
        "years_experience": 18,
        "services": ["Comprehensive neurology", "Movement disorders", "Neuromuscular diseases", "Epilepsy"]
    }
]


PROVIDER_WITH_MISSING_DATA = {
    "name": "Dr. John Doe",
    "specialty": "Neurology",
    "location": "Phoenix, AZ",
    "phone": None,
    "rating": None,
    "review_count": 0,
    "review_summary": "No reviews available",
    "review_sentiment": "unknown",
    "insurance_accepted": None,
    "distance": None,
    "years_experience": None,
    "services": []
}


PROVIDER_LOW_REVIEWS = {
    "name": "Dr. Jane Smith",
    "specialty": "Neurology",
    "location": "Phoenix, AZ",
    "phone": "(602) 555-1111",
    "rating": 5.0,
    "review_count": 3,
    "review_summary": "Small number of reviews but all positive. Patients mention excellent care and attention.",
    "review_sentiment": "positive",
    "insurance_accepted": "Aetna",
    "distance": "3.5 miles",
    "years_experience": 5,
    "services": ["General neurology"]
}
