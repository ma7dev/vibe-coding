#!/usr/bin/env python3
"""
Demo script to test the benchmark loading without requiring the full model
"""

import pandas as pd
import os

def test_benchmark_loading():
    """Test loading the benchmark data"""
    csv_path = "Pico-Saudi-LLMs-Benchmark/v0.01/Pico-Saudi-LLMs-Questions-v0.01.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Benchmark file not found: {csv_path}")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} questions from benchmark dataset")
        
        # Display dataset info
        print("\n📊 Dataset Summary:")
        print(f"   - Total questions: {len(df)}")
        print(f"   - Categories: {df['question_category'].unique().tolist()}")
        print(f"   - Questions per category:")
        for category, count in df['question_category'].value_counts().items():
            print(f"     • {category}: {count}")
        
        # Show sample questions
        print("\n📋 Sample Questions:")
        for idx, row in df.head(3).iterrows():
            print(f"\n{idx + 1}. Category: {row['question_category']}")
            print(f"   Question: {row['question']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading benchmark: {e}")
        return False

def simulate_model_responses(df):
    """Simulate model responses for testing"""
    print("\n🤖 Simulating model responses...")
    
    # Create mock responses
    responses = []
    for idx, row in df.iterrows():
        mock_response = f"هذا رد تجريبي للسؤال رقم {idx + 1} في فئة {row['question_category']}"
        responses.append({
            'question_id': row['question_id'],
            'question_category': row['question_category'],
            'question': row['question'],
            'response': mock_response,
            'timestamp': pd.Timestamp.now().isoformat()
        })
    
    # Save to CSV
    results_df = pd.DataFrame(responses)
    output_file = "demo_results.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"✓ Demo results saved to {output_file}")
    print(f"📊 Generated {len(results_df)} mock responses")
    
    return results_df

def main():
    """Main demo function"""
    print("🎯 Gemma-3 Saudi Benchmark Demo")
    print("=" * 40)
    
    # Test benchmark loading
    csv_path = "Pico-Saudi-LLMs-Benchmark/v0.01/Pico-Saudi-LLMs-Questions-v0.01.csv"
    df = pd.read_csv(csv_path) if test_benchmark_loading() else None
    
    if df is not None:
        # Simulate responses
        results = simulate_model_responses(df)
        
        print("\n✅ Demo completed successfully!")
        print("🔄 To run with actual Gemma-3 model:")
        print("   1. Run: ./setup.sh")
        print("   2. Run: source venv/bin/activate")
        print("   3. Run: python gemma3_saudi_benchmark.py")
    else:
        print("❌ Demo failed - check benchmark data")

if __name__ == "__main__":
    # Test if pandas is available
    try:
        import pandas as pd
        main()
    except ImportError:
        print("❌ pandas not found. Install with: pip install pandas")
        print("Or run the full setup: ./setup.sh")