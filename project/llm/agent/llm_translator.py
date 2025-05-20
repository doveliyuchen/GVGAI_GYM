from typing import Any, Optional, List
from llm.client import create_client_from_config

class VGDLTranslator:
    def __init__(self):
        pass

    def translate(self, vgdl_rules: str) -> str:
        """
        接收 VGDL 源码字符串，输出结构化自然语言说明。
        """
        # Step 1: 分段处理
        sections = self._split_sections(vgdl_rules)

        # Step 2: 分别翻译每一段
        results = []
        if "SpriteSet" in sections:
            results.append(self._translate_sprites(sections["SpriteSet"]))
        if "InteractionSet" in sections:
            results.append(self._translate_interactions(sections["InteractionSet"]))
        if "TerminationSet" in sections:
            results.append(self._translate_termination(sections["TerminationSet"]))
        
        return "\n\n".join(results)

    def _split_sections(self, vgdl_rules: str) -> dict:
        """
        将 VGDL 分为 SpriteSet, InteractionSet, TerminationSet 等模块
        """
        sections = {}
        current_key = None
        current_lines = []

        for line in vgdl_rules.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line in ["SpriteSet", "InteractionSet", "TerminationSet", "LevelMapping"]:
                if current_key:
                    sections[current_key] = current_lines
                current_key = line
                current_lines = []
            else:
                current_lines.append(line)
        
        if current_key:
            sections[current_key] = current_lines
        return sections

    def _translate_sprites(self, lines: List[str]) -> str:
        explanation = ["[Sprites]"]
        for line in lines:
            parts = line.strip().split(">")
            if len(parts) == 2:
                name, definition = parts
                explanation.append(f"Sprite '{name.strip()}' is a '{definition.strip()}'.")
            else:
                explanation.append(f"Sprite definition: {line}")
        return "\n".join(explanation)

    def _translate_interactions(self, lines: List[str]) -> str:
        explanation = ["[Interactions]"]
        for line in lines:
            tokens = line.strip().split()
            if len(tokens) == 3:
                sprite1, sprite2, effect = tokens
                explanation.append(f"When '{sprite1}' meets '{sprite2}', '{effect}' happens.")
            else:
                explanation.append(f"Interaction: {line}")
        return "\n".join(explanation)

    def _translate_termination(self, lines: List[str]) -> str:
        explanation = ["[Termination Conditions]"]
        for line in lines:
            explanation.append(f"- {line}")
        return "\n".join(explanation)

class LLMTranslator:

    def __init__(self, model_name: str):
        self.llm = create_client_from_config(model_name)
        self.translator = VGDLTranslator()  # 原始翻译器作为前处理

    def translate(self, vgdl_rules: str, level_layout: Optional[str] = None) -> str:
        translated = self.translator.translate(vgdl_rules)
        prompt = self._build_prompt(translated, level_layout)
        response = self.llm.query(prompt)
        print(response)
        return response

    def _build_prompt(self, translated: str, level: Optional[str] = None) -> str:
        prompt = (
            "You are an expert in video game analysis. Below is a description of a game's rules.\n"
            "Your task is to understand the game genre, mechanics, objective, win/loss conditions, and suggest a possible overall strategy. Make it in some understandable words and phrases.\n\n"
            f"=== Game Rules ===\n{translated.strip()}\n"
        )
        if level:
            prompt += f"\n=== Level Layout ===\n{level.strip()}\n"
        prompt += "\nPlease provide your analysis below:"
        return prompt
