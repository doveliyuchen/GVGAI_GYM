import os 
from llm.utils.config import get_profile_config
from llm.visual.ollama_client import OllamaClient
from llm.visual.openai_client import OpenAIClient
from llm.visual.qwen_client import QwenClient
from llm.visual.portkey_client import PortkeyClient
from llm.text.deepseek_client import DeepseekClient

def create_client_from_config(profile: str): # profile is the CLI name, e.g., "portkey-4o-mini", "gemini"
    """
    Create a configured LLM client from profile name in llm_config.json.
    """
    config = get_profile_config(profile) # Loads the specific profile from llm_config.json
    
    params = config.get("parameters", {})
    system_prompt_from_config = params.get("system_prompt", None)

    # Determine client type: from profile's "client_type" field, or infer from profile name
    # Default to profile name itself if "client_type" is not specified in the config.
    client_type_from_config = config.get("client_type", profile) 

    # Normalize client_type for common Portkey naming conventions from CLI
    if profile.startswith("portkey-") or profile == "gemini":
        effective_client_type = "portkey"
    else:
        effective_client_type = client_type_from_config

    if effective_client_type == "ollama":
        client = OllamaClient(model=config["model"]) # Assumes "model" field is the direct model name
    elif effective_client_type == "openai":
        client = OpenAIClient(model=config["model"])
    elif effective_client_type == "qwen":
        client = QwenClient(model=config["model"])
    elif effective_client_type == "deepseek":
        client = DeepseekClient(model=config["model"])
    elif effective_client_type == "portkey": 
        # This branch handles profiles explicitly marked with client_type: "portkey"
        # OR profiles named "gemini" or starting with "portkey-"
        
        portkey_base_url = config.get("portkey_base_url")
        api_key_env_var = config.get("portkey_api_key_env_var") # Name of env var for general Portkey API key
        virtual_key_env_var = config.get("virtual_key_env_var") # Name of env var for this profile's specific virtual key
        actual_model_name = config.get("actual_model_name") # Model string Portkey backend expects

        if not all([portkey_base_url, api_key_env_var, virtual_key_env_var, actual_model_name]):
            raise ValueError(
                f"Profile '{profile}' is configured to use Portkey (client_type: 'portkey' or by name) "
                "but is missing one or more required fields in its llm_config.json entry: "
                "'portkey_base_url', 'portkey_api_key_env_var', 'virtual_key_env_var', 'actual_model_name'."
            )

        portkey_api_key = os.getenv(api_key_env_var)
        # The specific virtual key string is fetched from the environment variable
        # whose name is specified in the profile's 'virtual_key_env_var' field.
        specific_virtual_key = os.getenv(virtual_key_env_var)

        if not portkey_api_key:
            raise ValueError(f"Environment variable '{api_key_env_var}' for Portkey API key not found (for profile '{profile}').")
        if not specific_virtual_key:
            raise ValueError(f"Environment variable '{virtual_key_env_var}' for Portkey virtual key not found (for profile '{profile}').")

        client = PortkeyClient(
            actual_model_name=actual_model_name,
            portkey_api_key=portkey_api_key,
            virtual_key=specific_virtual_key,
            base_url=portkey_base_url,
            system_prompt=system_prompt_from_config,
            max_context_tokens=params.get("max_context_tokens", 8000) # From profile's parameters
        )
    else:
        raise ValueError(f"Unsupported profile name ('{profile}') or client_type ('{effective_client_type}') in llm_config.json.")

    # Set common optional parameters on the client instance
    # These are typically defined in LLMClientBase or implemented by specific clients
    if hasattr(client, 'temperature'):
        client.temperature = params.get("temperature") # Can be None if not set
    if hasattr(client, 'max_tokens'): # For response generation length
        client.max_tokens = params.get("max_tokens") # Can be None
    if hasattr(client, 'top_p'):
        client.top_p = params.get("top_p") # Can be None

    return client
