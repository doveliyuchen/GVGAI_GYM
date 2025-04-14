import re

class PromptFactory:
    def __init__(self):
        pass

    @staticmethod
    def vgdl_rules_to_eng(vgdl_rules):
        return f"""
            I need you to convert VGDL (Video Game Description Language) code into detailed natural language instructions that smaller language models could understand and reason with. The translation should be exhaustive, precise, and maintain all the functional details from the original VGDL.

            Please follow these guidelines:

            1. Maintain complete coverage of all game elements, including sprites, interactions, termination conditions, and level layouts.

            2. Translate technical VGDL syntax into clear, unambiguous English sentences that explicitly state relationships and mechanics.

            3. Organize information logically, with sections for:
            - Game objective and win/loss conditions
            - Player controls and capabilities
            - Game entities and their properties
            - Interaction rules between entities
            - Level layout and spatial relationships

            4. For level layouts, describe both the general structure and specific important positional relationships.

            5. Avoid technical jargon unless you also explain it in plain language.

            6. Include explicit numerical values where the VGDL uses them (scores, speeds, probabilities, etc.).

            7. Make causal relationships clear (e.g., "When the avatar touches an alien, the avatar is destroyed and the game ends in a loss").

            Here is the VGDL code to translate:

            {vgdl_rules}

            Your translation should be detailed enough that a model without direct knowledge of VGDL could accurately visualize the game, understand its mechanics, and reason about gameplay strategies. Please provide a response formatted as follows:

            ```Translation: 
             
            [Your translation goes here]```
        """

def retrieve_vgdl_translation(llm_response):    
    text = llm_response
    
    text = text.strip() # Strip leading/trailing whitespace
    text = text.replace("```", "") # Remove backticks
    
    # Remove "Translation:" if present
    if "Translation:" in text: text = text.split("Translation:", 1)[1]
    
    text = text.strip() # Strip leading/trailing whitespace

    # Only remove wrapping brackets if they exist
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    
    # Only remove wrapping angle brackets if they exist
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    
    text = text.strip() # Strip leading/trailing whitespace
    
    return text