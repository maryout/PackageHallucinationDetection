SUPPORTED_LANGUAGES = ['python', 'javascript']

MODELS_CONFIG = {
    'commercial': {
        'gpt-4': {'api_name': 'gpt-4', 'provider': 'openai'},
        'claude-3-haiku-20240307': {'api_name': 'claude-3-haiku-20240307', 'provider': 'claude'},
        'claude-sonnet-4': {'api_name': 'claude-sonnet-4-20250514', 'provider': 'claude'},
        'deepseek-chaat': {'api_name': 'deepseek-chat', 'provider': 'deepseek'}  # DeepSeek ajouté dans commercial
    },
    'open_source': {
        'codellama-7b-Instruct-hf': {'hf_name': 'codellama/CodeLlama-7b-Instruct-hf','provider': 'huggingface'}
    }
}


SYSTEM_MESSAGES = {
    'python': {
        'code_generation': "You are a coding assistant that generates Python code. Provide only the code and add additional explanatory text only when absolutely necessary. If no code is required to answer the question, simply reply 'None'.",
        'package_from_code': "You are a coding assistant that determines Python packages necessary to execute code. Respond with only a list of PyPI packages, separated by commas and no additional text or formatting. If there is no code provided, respond 'None', otherwise the response must begin with the name of a Python package.",
        'package_from_prompt': "You are a coding assistant that recommends Python packages that would be helpful to solve given problems. Respond with only a list of PyPI packages, separated by commas and no additional text or formatting. The response must begin with the name of a Python package."
    },
    'javascript': {
        'code_generation': "You are a coding assistant that generates JavaScript code. Provide only the code and add additional explanatory text only when absolutely necessary. If no code is required to answer the question, simply reply 'None'.",
        'package_from_code': "You are a coding assistant that determines JavaScript packages necessary to execute code. Respond with only a list of npm packages, separated by commas and no additional text or formatting. If there is no code provided, respond 'None', otherwise the response must begin with the name of an npm package.",
        'package_from_prompt': "You are a coding assistant that recommends JavaScript packages that would be helpful to solve given problems. Respond with only a list of npm packages, separated by commas and no additional text or formatting. The response must begin with the name of an npm package."
    }
}


# For the run file 
EXPERIMENT_CONFIG = {
    'max_prompts': 4999,
    'languages_to_test': ['python'],
    'rate_limit_delay': 1,
}

# Data paths per language
def get_data_paths(language):
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Language {language} not supported. Supported: {SUPPORTED_LANGUAGES}")
    
    return {
        'packages': f"./data/{language}/master/valid-{'pypi' if language == 'python' else 'npm'}.csv",
        'prompts': [
            f"./data/{language}/prompts/LLM_PKG_DESC.jsonl",
            f"./data/{language}/prompts/SO_QST.jsonl"
        ]
    }
