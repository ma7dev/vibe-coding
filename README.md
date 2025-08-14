# Gemma-3 270M Saudi LLMs Benchmark Runner

A simple Python script to run Google's Gemma-3 270M model on the [Pico-Saudi-LLMs-Benchmark](https://github.com/mznmel/Pico-Saudi-LLMs-Benchmark) dataset using Unsloth for optimized inference.

## Overview

This script:
- Loads the Pico-Saudi-LLMs-Benchmark dataset (56 questions in Arabic covering various topics)
- Runs Gemma-3 270M model using Unsloth for efficient inference
- Generates responses to all benchmark questions
- Saves results to a CSV file for analysis

## Features

- ✅ **Optimized Performance**: Uses Unsloth for faster inference and reduced memory usage
- ✅ **Lightweight Model**: 270M parameters - runs efficiently on modest hardware
- ✅ **Fallback Support**: Automatically falls back to standard transformers if Unsloth fails
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
- 2GB+ RAM (CPU inference sufficient for 270M model)
- ~2GB free disk space for model downloads
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

- **Model**: `unsloth/gemma-2-270m-it` (270M parameters)
- **Precision**: Float16 for efficiency
- **Max Sequence Length**: 2048 tokens
- **Generation Settings**:
  - Max new tokens: 256
  - Temperature: 0.7
  - Top-p: 0.9

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
- 2GB RAM
- 2GB free disk space
- CPU-only (runs fine on modest hardware)

### Recommended
- 4GB+ RAM
- GPU with 2GB+ VRAM (for faster inference)
- 5GB+ free disk space

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - The 270M model should run on most modern computers
   - Use CPU-only mode if GPU issues occur
   - Close other applications to free up RAM

2. **Model Download Fails**
   - Check internet connection
   - Verify Hugging Face access (some models require approval)
   - Clear Hugging Face cache: `rm -rf ~/.cache/huggingface/`

3. **Unsloth Installation Issues**
   - Try installing without Unsloth (script will use fallback)
   - Use conda environment: `conda create -n gemma python=3.9`

### Performance Tips

- **GPU**: Use GPU for 2-3x speedup over CPU
- **Memory**: Close other applications to free up RAM
- **Storage**: Use SSD for faster model loading
- **Batch Processing**: The 270M model is fast enough for real-time inference

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
- **Model**: Gemma-3 270M by Google
- **Optimization**: [Unsloth](https://github.com/unslothai/unsloth) for efficient inference

## License

This project is provided as-is for educational and research purposes. Please respect the licenses of the underlying components:

- Gemma model license
- Benchmark dataset license
- Unsloth license
