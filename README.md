# Gemma-3 270M GGUF Saudi LLMs Benchmark Runner

A simple Python script to run Google's Gemma-3 270M GGUF model on the [Pico-Saudi-LLMs-Benchmark](https://github.com/mznmel/Pico-Saudi-LLMs-Benchmark) dataset using Unsloth for optimized inference.

## Overview

This script:
- Loads the Pico-Saudi-LLMs-Benchmark dataset (56 questions in Arabic covering various topics)
- Runs Gemma-3 270M GGUF model using Unsloth for efficient inference
- Generates responses to all benchmark questions
- Saves results to a CSV file for analysis

## Features

- ✅ **Optimized Performance**: Uses Unsloth for faster inference and reduced memory usage
- ✅ **GGUF Format**: Highly optimized quantized model format for maximum efficiency
- ✅ **Lightweight Model**: 270M parameters - runs efficiently on modest hardware
- ✅ **Fallback Support**: Multiple fallback methods if GGUF loading fails
- ✅ **Progress Tracking**: Real-time progress updates during benchmark execution
- ✅ **Error Handling**: Robust error handling with informative messages
- ✅ **Arabic System Prompt**: Follows the benchmark's requirement for Arabic responses

## Quick Start

### 1. Setup Environment

Run the setup script to install all dependencies:

```bash
./setup.sh
```

Or manually install requirements:

```bash
pip install -r requirements.txt
```

### 2. Test Setup (Optional)

Run the demo script to test data loading without downloading the model:

```bash
# Install pandas only for demo
python3 -m venv demo_venv
source demo_venv/bin/activate
pip install pandas
python3 test_demo.py
```

### 3. Run Full Benchmark

```bash
# Activate the environment and run
source venv/bin/activate
python gemma3_saudi_benchmark.py
```

## Requirements

- Python 3.8+
- 1.5GB+ RAM (GGUF format is very memory efficient)
- ~1GB free disk space for GGUF model download
- Internet connection for initial model download
- GPU optional but recommended for faster inference

## Dataset

The script uses the **Pico-Saudi-LLMs-Benchmark v0.01** which contains:

- **56 questions** in Arabic
- **6 categories**: history, culture, literature, localDialects, reasoning, footballSport, music&art, media, coding
- Focus on Saudi culture and Arabic language understanding

### Sample Questions

| Category | Example Question |
|----------|------------------|
| History | اشرح الدور التاريخي لمدينة الدرعية في تأسيس الدولة السعودية الأولى |
| Culture | اشرح مفهوم 'العرضة' ودورها في المناسبات الوطنية السعودية |
| Reasoning | ما هو العدد الذي إذا ضربته في نفسه وأضفت إليه 2 يصبح الناتج 10؟ |

## Model Configuration

- **Model**: `unsloth/gemma-3-270m-it-GGUF` (270M parameters)
- **Format**: GGUF (highly optimized quantized format)
- **Max Sequence Length**: 2048 tokens
- **Context Length**: Up to 32K tokens supported
- **Generation Settings** (Gemma 3 optimized):
  - Max new tokens: 256
  - Temperature: 1.0
  - Top-k: 64
  - Top-p: 0.95
  - Repetition penalty: 1.0

## Output

The script generates:

1. **Console Output**: Real-time progress and sample results
2. **CSV File**: `gemma3_results.csv` with columns:
   - `question_id`: Unique identifier
   - `question_category`: Topic category
   - `question`: Original Arabic question
   - `response`: Model's Arabic response
   - `timestamp`: Generation timestamp

## Example Usage

```python
# The script can also be imported and used programmatically
from gemma3_saudi_benchmark import setup_gemma3_model, load_benchmark_data, run_benchmark

# Load model
model, tokenizer = setup_gemma3_model()

# Load questions
questions_df = load_benchmark_data("Pico-Saudi-LLMs-Benchmark/v0.01/Pico-Saudi-LLMs-Questions-v0.01.csv")

# Run benchmark
results = run_benchmark(model, tokenizer, questions_df, "my_results.csv")
```

## System Requirements

### Minimum
- 1.5GB RAM
- 1GB free disk space
- CPU-only (GGUF runs very efficiently)

### Recommended
- 3GB+ RAM
- GPU with 1GB+ VRAM (for faster inference)
- 3GB+ free disk space

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - The GGUF format is extremely memory efficient
   - Should run on most computers with 2GB+ RAM
   - Use CPU-only mode if GPU issues occur
   - Close other applications to free up RAM

2. **Model Download Fails**
   - Check internet connection
   - Verify Hugging Face access (some models require approval)
   - Clear Hugging Face cache: `rm -rf ~/.cache/huggingface/`

3. **GGUF Loading Issues**
   - Script has multiple fallback methods
   - Will try loading without quantization if needed
   - Falls back to standard transformers as last resort
   - Use conda environment: `conda create -n gemma python=3.9`

### Performance Tips

- **GPU**: Use GPU for 2-3x speedup over CPU
- **Memory**: GGUF format is very memory efficient
- **Storage**: Use SSD for faster model loading
- **Quantization**: GGUF format provides optimal speed/quality balance
- **Batch Processing**: Very fast inference, suitable for real-time use

## File Structure

```
.
├── gemma3_saudi_benchmark.py    # Main script
├── test_demo.py                # Demo script (no model required)
├── requirements.txt            # Python dependencies
├── setup.sh                   # Setup script
├── README.md                  # This file
├── venv/                      # Virtual environment (after setup)
├── Pico-Saudi-LLMs-Benchmark/ # Dataset (downloaded by setup)
│   └── v0.01/
│       └── Pico-Saudi-LLMs-Questions-v0.01.csv
├── gemma3_results.csv         # Output (generated after run)
└── demo_results.csv           # Demo output (from test_demo.py)
```

## Credits

- **Dataset**: [Pico-Saudi-LLMs-Benchmark](https://github.com/mznmel/Pico-Saudi-LLMs-Benchmark) by mznmel
- **Model**: [Gemma-3 270M GGUF](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF) by Google/Unsloth
- **Optimization**: [Unsloth](https://github.com/unslothai/unsloth) for efficient inference

## License

This project is provided as-is for educational and research purposes. Please respect the licenses of the underlying components:

- Gemma model license
- Benchmark dataset license
- Unsloth license
