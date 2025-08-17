#!/usr/bin/env python3
"""
Simple script to run Gemma-3 model on Pico-Saudi-LLMs-Benchmark dataset using Unsloth
"""

import pandas as pd
import torch
from datetime import datetime
import os
import sys

def check_requirements():
    """Check if required packages are installed"""
    try:
        import unsloth
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✓ All required packages are available")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install required packages:")
        print("pip install unsloth transformers torch pandas")
        return False

def load_benchmark_data(csv_path):
    """Load the benchmark questions from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} questions from benchmark dataset")
        return df
    except FileNotFoundError:
        print(f"❌ Could not find benchmark file: {csv_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading benchmark data: {e}")
        return None

def setup_gemma3_model():
    """Setup Gemma-3 270M GGUF model using Unsloth"""
    try:
        from unsloth import FastLanguageModel
        
        print("🔄 Loading Gemma-3 270M GGUF model using Unsloth...")
        
        # Load GGUF model with Unsloth optimizations
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gemma-3-270m-it-GGUF",  # Exact GGUF model specified
            max_seq_length=2048,
            dtype=None,  # Auto detection
            load_in_4bit=True,  # GGUF models work well with quantization
        )
        
        # Enable inference mode
        FastLanguageModel.for_inference(model)
        
        print("✓ Gemma-3 270M GGUF model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading GGUF model with Unsloth: {e}")
        print("Trying alternative loading method...")
        
        try:
            from unsloth import FastLanguageModel
            
            # Try without quantization
            print("🔄 Trying without quantization...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/gemma-3-270m-it-GGUF",
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=False,
            )
            
            FastLanguageModel.for_inference(model)
            print("✓ GGUF model loaded without quantization")
            return model, tokenizer
            
        except Exception as e2:
            print(f"❌ Error with alternative method: {e2}")
            print("Falling back to standard transformers...")
            
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                # Fallback to regular Gemma 2 270M
                model_name = "google/gemma-2-270m-it"
                print(f"🔄 Loading fallback {model_name} with transformers...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                print("✓ Fallback model loaded successfully")
                return model, tokenizer
                
            except Exception as e3:
                print(f"❌ Error loading fallback model: {e3}")
                return None, None

def generate_response(model, tokenizer, question, max_length=512):
    """Generate response for a single question"""
    try:
        # Format the prompt according to Gemma's chat template
        prompt = f"""<bos><start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
"""
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate response with Gemma 3 recommended settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=1.0,     # Recommended for Gemma 3
                top_k=64,           # Recommended for Gemma 3
                top_p=0.95,         # Recommended for Gemma 3
                repetition_penalty=1.0,  # Recommended for Gemma 3
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the model's response (after the last model turn)
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1].strip()
        
        return response
        
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return f"Error: {str(e)}"

def run_benchmark(model, tokenizer, questions_df, output_file="gemma3_results.csv"):
    """Run the model on all benchmark questions"""
    print(f"🔄 Running benchmark on {len(questions_df)} questions...")
    
    results = []
    
    for idx, row in questions_df.iterrows():
        question_id = row['question_id']
        category = row['question_category']
        question = row['question']
        
        print(f"Processing question {idx + 1}/{len(questions_df)} (ID: {question_id})")
        
        # Generate response
        response = generate_response(model, tokenizer, question)
        
        # Store result
        results.append({
            'question_id': question_id,
            'question_category': category,
            'question': question,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Print progress every 10 questions
        if (idx + 1) % 10 == 0:
            print(f"✓ Completed {idx + 1} questions")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"✓ Results saved to {output_file}")
    
    return results_df

def main():
    """Main function"""
    print("🚀 Gemma-3 270M GGUF Saudi LLMs Benchmark Runner")
    print("=" * 50)
    
    # Check if running with proper setup
    if not check_requirements():
        return
    
    # Load benchmark data
    csv_path = "Pico-Saudi-LLMs-Benchmark/v0.01/Pico-Saudi-LLMs-Questions-v0.01.csv"
    questions_df = load_benchmark_data(csv_path)
    
    if questions_df is None:
        return
    
    # Setup model
    model, tokenizer = setup_gemma3_model()
    
    if model is None or tokenizer is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Run benchmark
    print(f"🎯 Running benchmark with Gemma-3 270M GGUF")
    print(f"📋 Model: unsloth/gemma-3-270m-it-GGUF")
    print(f"📋 System prompt: 'You must provide all your responses exclusively in Arabic'")
    
    results_df = run_benchmark(model, tokenizer, questions_df)
    
    print("=" * 50)
    print("✅ Benchmark completed successfully!")
    print(f"📊 Processed {len(results_df)} questions")
    print("📁 Results saved to gemma3_results.csv")
    
    # Display sample results
    print("\n📋 Sample Results:")
    for idx, row in results_df.head(3).iterrows():
        print(f"\nQuestion {idx + 1}: {row['question']}")
        print(f"Response: {row['response'][:100]}...")

if __name__ == "__main__":
    main()