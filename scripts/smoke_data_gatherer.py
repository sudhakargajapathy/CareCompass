"""MANUAL smoke script — makes REAL Tavily + Anthropic calls and costs money.

Deliberately NOT under tests/ and deliberately not named test_*: it used to be
`test_data_gatherer.py` at the repo root, where `pytest .` collected it. That
ran the full live pipeline against real API keys, wrote to the real
./chroma_db (the autouse isolation fixture only covers tests/), and contained
no assertions at all, so it reported PASSED whether the pipeline worked or the
keys were revoked.

Run it deliberately:  python scripts/smoke_data_gatherer.py
"""
from agents.data_gatherer import create_data_gatherer
import json

def smoke_data_gatherer():
    """Test the data gatherer directly."""
    print("🔍 Testing Data Gatherer with Google review prioritization...\n")

    gatherer = create_data_gatherer()

    # Run search
    results = gatherer.gather_providers(
        specialty="Neurology",
        location="Scottsdale, AZ",
        insurance=None
    )

    if results.get("status") == "success":
        providers = results.get("providers", [])
        print(f"✅ Found {len(providers)} providers\n")

        for i, provider in enumerate(providers[:3], 1):
            print(f"{'='*80}")
            print(f"Provider {i}: {provider.get('name', 'N/A')}")
            print(f"Specialty: {provider.get('specialty', 'N/A')}")
            print(f"Rating: {provider.get('rating', 0)} ⭐ ({provider.get('review_count', 0)} reviews)")
            print(f"Sentiment: {provider.get('review_sentiment', 'unknown')}")

            summary = provider.get('review_summary', 'No reviews available')
            print(f"\n📝 Review Summary:")
            print(f"{summary}")

            # Count sentences in summary
            sentence_count = summary.count('.') + summary.count('!') + summary.count('?')
            print(f"\n📊 Summary Stats:")
            print(f"   - Sentences: {sentence_count}")
            print(f"   - Characters: {len(summary)}")
            print(f"   - Words: {len(summary.split())}")
            print(f"{'='*80}\n")
    else:
        print(f"❌ Error: {results.get('message', 'Unknown error')}")
        print(f"Status: {results.get('status')}")

if __name__ == "__main__":
    smoke_data_gatherer()
