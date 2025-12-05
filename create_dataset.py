#!/usr/bin/env python3
"""
Create Dataset from GeoQA3 JSON Files
Uses problem_parser.py to batch process problems with filtering and classification.
"""

import os
import sys
from problem_parser import ProblemParser, create_openai_api_function
from dotenv import load_dotenv

load_dotenv()

def main():
    """Main function to create dataset from JSON files."""
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not found in environment variables.")
        print("   Set it with: export OPENAI_API_KEY='your-api-key'")
        print("   Or add it to your ~/.zshrc or ~/.bashrc")
        print("\n   Without API key, classification and difficulty rating will not be available.")
        
        response = input("\nContinue without classification? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
        
        llm_function = None
    else:
        print("✓ OpenAI API key found")
        print("  Using GPT-4o-mini for parsing, classification, and difficulty rating\n")
        llm_function = create_openai_api_function(model="gpt-4.1-mini", api_key=api_key)
    
    # Initialize parser
    parser = ProblemParser(llm_api_function=llm_function)
    
    # Configuration
    input_dir = "data-5/GeoQA3/json"  # Change this to your input directory
    output_file = "ground_truth/geoqa3_dataset.json"  # Change this to your output file
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"✗ Error: Input directory not found: {input_dir}")
        print(f"  Please update the 'input_dir' variable in this script.")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created output directory: {output_dir}\n")
    
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"\nProcessing options:")
    print(f"  - Skip ambiguous references (∠1, ∠2, etc.): ✓")
    print(f"  - Clean text (remove '如图所示', questions): ✓")
    print(f"  - Classify problems: {'✓' if llm_function else '✗'}")
    print(f"  - Rate difficulty: {'✓' if llm_function else '✗'}")
    print()
    
    # Batch process
    stats = parser.batch_parse_directory(
        input_dir=input_dir,
        output_file=output_file,
        skip_ambiguous=True,  # Skip problems with ∠1, ∠2, etc.
        clean_text=True,  # Remove "如图所示" and question parts
        file_pattern="*.json"
    )
    
    print(f"\n✓ Dataset creation complete!")
    print(f"  Output saved to: {output_file}")


if __name__ == "__main__":
    main()


