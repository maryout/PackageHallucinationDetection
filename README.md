# Package Hallucination Detection

Evaluating whether LLMs suggest or use non-existent packages when generating Python and JavaScript .

## 📋 Table of Contents


- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Supported Models](#supported-models)
- [Results Analysis](#results-analysis)
- [Project Structure](#project-structure)



## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PackageHallucinationDetection.git
cd PackageHallucinationDetection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Required API Keys

Add the following to your `.env` file:

```env
OPENAI_API_KEY=your_openai_key_here
CLAUDE_API_KEY=your_claude_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
HUGGINGFACE_API_KEY=your_hf_token_here
```

## ⚙️ Configuration

### Models Configuration

The system supports various LLM providers configured in `config.py`:

**Commercial Models:**
- `gpt-4` (OpenAI)
- `claude-3-haiku-20240307` (Anthropic)
- `claude-sonnet-4` (Anthropic)
- `deepseek-chat` (DeepSeek)

**Open Source Models:**
- `codellama-7b-Instruct-hf` (HuggingFace)


## 📖 Usage

### Basic Experiment Run

Run an experiment with a specific model:

```bash
# Run with Claude Sonnet 4
python run_experiment.py --claude-sonnet-4

# Run with GPT-4
python run_experiment.py --gpt-4

# Run with DeepSeek
python run_experiment.py --deepseek-chaat
```

The script will:
1. Prompt you to select a programming language
2. Load prompts from the dataset
3. Generate code using the selected LLM
4. Analyze generated code for package hallucinations
5. Save results to `results/{language}/{model}/`



### Generate Comparative Plots

Create visualization comparing all models:

```bash
python plot_resultat.py
```
### Heuristic Analysis

Analyze results by individual heuristics:

```bash
python heuristic_analysis.py --model claude-3-haiku-20240307
```

## 🤖 Supported Models

| Provider | Model | API Name |
|----------|-------|----------|
| OpenAI | GPT-4 | `gpt-4` |
| Anthropic | Claude 3 Haiku | `claude-3-haiku-20240307` |
| Anthropic | Claude Sonnet 4 | `claude-sonnet-4-20250514` |
| DeepSeek | DeepSeek Chat | `deepseek-chat` |
| HuggingFace | CodeLlama 7B | `codellama/CodeLlama-7b-Instruct-hf` |



### Sample Output

```
FINAL REPORT - PYTHON
========================================
Total detected packages: 156
Hallucinated packages: 23
Valid packages: 133
Hallucination rate: 14.74%
```

## 📁 Project Structure

```
PackageHallucinationDetection/
├── config.py                 # Configuration and model settings
├── detector.py               # Core detection logic and heuristics
├── run_experiment.py         # Main experiment runner
├── utils.py                  # Utility functions for data processing
├── heuristic_analysis.py     # Per-heuristic analysis tool
├── plot_resultat.py          # Visualization generation
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (API keys)
├── data/                     # Dataset files
│   ├── python/
│   │   ├── master/valid-pypi.csv
│   │   └── prompts/
│   └── javascript/
│       ├── master/valid-npm.csv
│       └── prompts/
├── results/                  # Experiment outputs
├── results_analysis/         # Heuristic analysis outputs
└── scripts/                  # scripts used to generate the dataset 
```

### Custom Models

Add new models to `config.py`:

```python
MODELS_CONFIG = {
    'commercial': {
        'your-model': {
            'api_name': 'your-model-api-name',
            'provider': 'your-provider'
        }
    }
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
