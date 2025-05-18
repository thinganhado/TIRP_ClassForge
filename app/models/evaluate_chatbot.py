#!/usr/bin/env python
"""
Evaluation script for the RuleBasedChatbot in app/models/assistant.py.
This script tests the chatbot's accuracy and performance across different domains.
Target accuracy: 84%+

Usage:
python evaluate_chatbot.py [--verbose]
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chatbot_evaluation.log")
    ]
)
logger = logging.getLogger('chatbot_evaluator')

# Import the chatbot assistant module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from app.models.assistant import RuleBasedChatbot
    logger.info("Successfully imported RuleBasedChatbot")
except ImportError as e:
    logger.error(f"Error importing RuleBasedChatbot: {e}")
    sys.exit(1)


class ChatbotEvaluator:
    """Evaluates the accuracy and performance of the enhanced RuleBasedChatbot"""
    
    def __init__(self, chatbot_instance=None):
        """Initialize the evaluator with an optional chatbot instance"""
        
        # Create a chatbot instance if none provided
        self.chatbot = chatbot_instance or RuleBasedChatbot()
        
        # Enhanced test cases for each domain
        self.test_cases = {
            "academic": [
                "I want to focus on academic performance",
                "How can I improve student grades?",
                "Academic achievement is my priority",
                "I need to boost GPA across all classes",
                "Students need better test scores",
                "Make learning outcomes the top priority",
                "Can we prioritize academic factors?",
                "Help me optimize for better student achievement",
                "The school board wants to improve academic metrics",
                "Let's emphasize educational performance"
            ],
            "wellbeing": [
                "Student wellbeing should be the priority",
                "I'm concerned about mental health",
                "How can I reduce student stress?",
                "My students seem anxious about school",
                "Emotional wellbeing is important to me",
                "Can we focus more on student wellness?",
                "Mental health needs more emphasis",
                "I want to create emotionally supportive classrooms",
                "Psychological safety is my main concern",
                "Stress reduction should be our focus"
            ],
            "bullying": [
                "There's a bullying problem in my class",
                "How can I prevent harassment between students?",
                "I need to create a safer classroom environment",
                "Some students are being excluded by others",
                "Bullying prevention should be a top priority",
                "Help me address student conflicts",
                "I need to separate bullies from victims",
                "Some kids are getting picked on",
                "How can I improve classroom safety?",
                "Let's focus on reducing aggressive behavior"
            ],
            "social": [
                "I want to improve social dynamics",
                "There are issues with social hierarchies",
                "Some students have too much influence",
                "How can I manage peer pressure?",
                "Social interactions need balancing",
                "The social environment needs improvement",
                "Help me create better social balance",
                "I need to address social power dynamics",
                "Some students are socially dominant",
                "Can you help with peer interaction issues?"
            ],
            "friendship": [
                "Students need more friends in class",
                "How can I strengthen peer relationships?",
                "Some students are isolated from friend groups",
                "I want to ensure every student has a friend",
                "Friendship connections should be improved",
                "Keep friend groups together",
                "Help students maintain their friendships",
                "I want to prevent social isolation",
                "Let's prioritize social connections",
                "Can you optimize for better friendship networks?"
            ]
        }
        
        # Expected parameter changes per domain
        self.expected_changes = {
            "academic": [
                {"prioritize_academic": 1, "gpa_penalty_weight": 1}
            ],
            "wellbeing": [
                {"prioritize_wellbeing": 1, "wellbeing_penalty_weight": 1}
            ],
            "bullying": [
                {"prioritize_bullying": 1, "bully_penalty_weight": 1}
            ],
            "social": [
                {"prioritize_social_influence": 1, "influence_std_weight": 1, "isolated_std_weight": 1}
            ],
            "friendship": [
                {"prioritize_friendship": 1, "friend_inclusion_weight": 1, "friendship_balance_weight": 1, "min_friends_required": 1}
            ]
        }
        
        self.results = defaultdict(list)
        
    def reset_chatbot_config(self):
        """Reset the chatbot configuration to defaults for a clean test"""
        default_config = {
            "bully_penalty_weight": 50,
            "class_size": 30,
            "friend_inclusion_weight": 50,
            "friendship_balance_weight": 50,
            "gpa_penalty_weight": 50,
            "influence_std_weight": 50,
            "isolated_std_weight": 50,
            "max_classes": 6,
            "min_friends_required": 3,
            "prioritize_academic": 5,
            "prioritize_bullying": 5,
            "prioritize_friendship": 5,
            "prioritize_social_influence": 5,
            "prioritize_wellbeing": 5,
            "wellbeing_penalty_weight": 50
        }
        
        # Reset the chatbot's config
        with open(self.chatbot.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        # Reload the config
        self.chatbot.config = self.chatbot._load_config()
        
        # Clear interaction history
        self.chatbot.interaction_history = defaultdict(list)
        self.chatbot.last_domain = None
        
        logger.info("Reset chatbot configuration to defaults")
        
    def evaluate_domain(self, domain):
        """Evaluate the chatbot's performance for a specific domain"""
        logger.info(f"Evaluating domain: {domain}")
        domain_results = []
        
        # Get the test cases for this domain
        test_cases = self.test_cases.get(domain, [])
        
        # Reset the chatbot to default state
        self.reset_chatbot_config()
        
        # Process each test case
        for query in test_cases:
            # Process the query
            if hasattr(self.chatbot, 'process_input'):
                # Use new improved method
                result = self.chatbot.process_input(query)
                changed_params = result.get("changed_params", [])
                response = result.get("response", "")
                modified = bool(changed_params)
            else:
                # Fallback to old analyze_request method
                response, original_config, modified_config = self.chatbot.analyze_request(query)
                changed_params = [k for k in modified_config if original_config.get(k) != modified_config.get(k)]
                modified = original_config != modified_config
            
            # Check which expected parameters were changed
            expected_params = []
            for param_set in self.expected_changes.get(domain, []):
                expected_params.extend(param_set.keys())
            
            expected_params_changed = [p for p in expected_params if p in changed_params]
            
            # Determine success
            success = bool(expected_params_changed) and modified
            
            # Log the result
            logger.info(f"Query: '{query}'")
            logger.info(f"Success: {success}, Modified: {modified}")
            logger.info(f"Changed parameters: {changed_params}")
            logger.info(f"Expected parameters changed: {expected_params_changed}")
            logger.info(f"Response: {response}")
            logger.info("---")
            
            # Store the test result
            domain_results.append({
                "query": query,
                "success": success,
                "modified": modified,
                "changed_params": changed_params,
                "expected_params": expected_params,
                "expected_params_changed": expected_params_changed,
                "response": response
            })
        
        # Calculate the domain accuracy
        correct = sum(1 for r in domain_results if r["success"])
        total = len(domain_results)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        logger.info(f"Domain {domain} accuracy: {accuracy:.1f}% ({correct}/{total})")
        
        return {
            "accuracy": accuracy,
            "total": total,
            "details": domain_results
        }
    
    def run_evaluation(self):
        """Run a complete evaluation of the chatbot across all domains"""
        logger.info("Starting comprehensive chatbot evaluation")
        start_time = time.time()
        
        results = {}
        all_correct = 0
        all_total = 0
        
        # Evaluate each domain
        for domain in self.test_cases.keys():
            domain_result = self.evaluate_domain(domain)
            results[domain] = domain_result
            
            all_correct += sum(1 for r in domain_result["details"] if r["success"])
            all_total += domain_result["total"]
        
        # Calculate overall accuracy
        overall_accuracy = (all_correct / all_total * 100) if all_total > 0 else 0
        
        # Create the final results
        final_results = {
            "overall_accuracy": overall_accuracy,
            "domain_results": results
        }
        
        # Save the results to a JSON file
        results_file = "chatbot_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Also save a simplified summary
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_accuracy": overall_accuracy,
            "domain_accuracies": {d: results[d]["accuracy"] for d in results}
        }
        
        with open("chatbot_evaluation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        end_time = time.time()
        logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Overall accuracy: {overall_accuracy:.1f}%")
        logger.info(f"Results saved to {results_file}")
        
        return final_results
    
    def print_evaluation_summary(self, results):
        """Print a human-readable summary of the evaluation results"""
        print("\n" + "="*60)
        print("CHATBOT EVALUATION SUMMARY")
        print("="*60)
        
        overall_accuracy = results["overall_accuracy"]
        print(f"\nOverall Accuracy: {overall_accuracy:.1f}%")
        print("\nDomain Accuracies:")
        
        for domain, domain_results in results["domain_results"].items():
            accuracy = domain_results["accuracy"]
            correct = sum(1 for r in domain_results["details"] if r["success"])
            total = domain_results["total"]
            
            # Use colored output if available
            if accuracy >= 84:
                status = "✅"  # Target met
            elif accuracy >= 70:
                status = "⚠️"  # Close but not quite
            else:
                status = "❌"  # Needs improvement
                
            print(f"  {status} {domain.capitalize()}: {accuracy:.1f}% ({correct}/{total})")
        
        print("\nDetailed Results:")
        for domain, domain_results in results["domain_results"].items():
            print(f"\n  {domain.capitalize()} Domain:")
            for i, test in enumerate(domain_results["details"], 1):
                result = "✓" if test["success"] else "✗"
                print(f"    {i}. {result} '{test['query']}'")
                if not test["success"]:
                    print(f"       Expected: {', '.join(test['expected_params'])}")
                    print(f"       Changed: {', '.join(test['changed_params'])}")
        
        print("\n" + "="*60)
        
        # Summary statement
        if overall_accuracy >= 84:
            print("\n✅ Target accuracy of 84% achieved! The model is ready for deployment.")
        else:
            print(f"\n⚠️ Current accuracy ({overall_accuracy:.1f}%) is below the target (84%).")
            print("   Further improvements are needed in the following domains:")
            
            for domain, domain_results in results["domain_results"].items():
                if domain_results["accuracy"] < 84:
                    print(f"   - {domain.capitalize()}: {domain_results['accuracy']:.1f}%")
        
        print("\n" + "="*60)

def main():
    """Main function to run the evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate the chatbot performance")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create and run the evaluator
    evaluator = ChatbotEvaluator()
    results = evaluator.run_evaluation()
    
    # Print the summary
    evaluator.print_evaluation_summary(results)
    
    # Return success if accuracy target met
    return 0 if results["overall_accuracy"] >= 84 else 1

if __name__ == "__main__":
    sys.exit(main())
