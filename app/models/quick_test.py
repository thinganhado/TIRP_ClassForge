#!/usr/bin/env python
"""
Simple test script for the chatbot in app/models/assistant.py
"""
import os
import sys
import json

# Add the current directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the chatbot
try:
    from app.models.assistant import RuleBasedChatbot, chatbot
    print("✅ Successfully imported the chatbot")
except ImportError as e:
    print(f"❌ Error importing chatbot: {e}")
    sys.exit(1)

def test_single_query(query):
    """Test a single query and print detailed results"""
    print(f"\n🔍 Testing query: \"{query}\"")
    
    # Get original config
    original_config = chatbot.get_current_config()
    print("\nOriginal configuration:")
    for key, value in original_config.items():
        print(f"  {key}: {value}")
    
    # Process the query
    result = chatbot.analyze_request(query)
    
    # Check basic success
    if not result["success"]:
        print(f"\n❌ Failed to process query: {result['message']}")
        return
    
    # Check if config was modified
    if not result["is_modified"]:
        print("\n⚠️ No configuration changed")
        return
    
    # Check what parameters were modified
    print("\nResponse message:")
    print(f"  {result['message']}")
    
    print("\nModified configuration:")
    modified_params = []
    
    for key, new_value in result["modified_config"].items():
        if key in original_config and new_value != original_config[key]:
            modified_params.append(key)
            print(f"  ✓ {key}: {original_config[key]} -> {new_value}")
        else:
            print(f"  {key}: {new_value}")
    
    if modified_params:
        print(f"\n✅ Modified parameters: {', '.join(modified_params)}")
    else:
        print("\n⚠️ No parameters were actually changed")

def main():
    """Run a series of test queries"""
    print("🤖 Chatbot Quick Test")
    print("=" * 50)
    
    # Test academic query
    test_single_query("I want to focus on academic performance")
    
    # Test wellbeing query
    test_single_query("Student wellbeing should be the priority")
    
    # Test bullying query
    test_single_query("There's a bullying problem in my class")
    
    # Test social query
    test_single_query("I want to improve social dynamics")
    
    # Test friendship query
    test_single_query("Students need more friends in class")
    
    print("\n✨ Testing complete!")

if __name__ == "__main__":
    main()