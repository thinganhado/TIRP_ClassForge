import sys
import os

# Add the TIRP directory to the system path so we can import app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.database_loader import DatabaseLoader
from app.models.assistant import RuleBasedChatbot

def main():
    """Example of connecting to the database using app/config.py and training on all sources"""
    print("=" * 80)
    print("TIRP Database Connection Example")
    print("=" * 80)
    
    # Method 1: Using config.py (preferred) - trains on all data sources automatically
    print("\n[Method 1] Comprehensive training using config.py and all data sources")
    
    # Create chatbot with comprehensive training
    # It will automatically:
    # 1. Connect to database using config.py credentials
    # 2. Load all available data from database
    # 3. Fill in missing data from CSV files
    # 4. Generate synthetic data for any remaining missing essential data
    # 5. Train NLP models on the complete dataset
    print("\nInitializing chatbot with comprehensive training...")
    chatbot = RuleBasedChatbot()
    
    # Show what data sources were loaded
    print("\nLoaded Data Sources:")
    for key, value in chatbot.data_cache.items():
        if hasattr(value, 'shape'):
            print(f"  - {key}: {value.shape[0]} records")
        elif isinstance(value, list):
            print(f"  - {key}: {len(value)} items")
        elif isinstance(value, dict):
            print(f"  - {key}: {len(value)} items")
    
    # Test a few different query types
    print("\nTesting various query types:")
    
    # Academic query
    academic_query = "How can I optimize for academic performance?"
    print(f"\nAcademic Query: '{academic_query}'")
    response = chatbot.analyze_request(academic_query)
    print(f"Response: {response['message']}")
    
    # Wellbeing query
    wellbeing_query = "I need help with students who have low wellbeing"
    print(f"\nWellbeing Query: '{wellbeing_query}'")
    response = chatbot.analyze_request(wellbeing_query)
    print(f"Response: {response['message']}")
    
    # Bullying query
    bullying_query = "How can I address bullying in my classroom?"
    print(f"\nBullying Query: '{bullying_query}'")
    response = chatbot.analyze_request(bullying_query)
    print(f"Response: {response['message']}")
    
    # Social query
    social_query = "Help me improve social connections for isolated students"
    print(f"\nSocial Query: '{social_query}'")
    response = chatbot.analyze_request(social_query)
    print(f"Response: {response['message']}")
    
    # Combined query
    combined_query = "I need a balanced approach focusing on both wellbeing and academic performance"
    print(f"\nCombined Query: '{combined_query}'")
    response = chatbot.analyze_request(combined_query)
    print(f"Response: {response['message']}")
    
    print("\n" + "=" * 80)
    print("Example complete")
    print("=" * 80)

if __name__ == "__main__":
    main() 