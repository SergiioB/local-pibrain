#!/usr/bin/env python3
"""
Unified LLM client for Personal AI Node.
Uses llama.cpp directly for all LLM operations.
Configured for 27B model with non-thinking mode.
"""

import json
import hashlib
import subprocess
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import re
import yaml

# Paths - update to match your setup
LLAMA_CLI = Path("llama-cli")  # Or full path to llama-cli binary
MODELS_DIR = Path("models")     # Directory containing GGUF model files
DEFAULT_MODEL = "Qwen_Qwen3.5-27B-Q4_K_M.gguf"  # 27B for best quality
DB_PATH = Path(__file__).parent.parent.parent / "data" / "state.db"
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "models.yaml"


@dataclass
class LLMConfig:
    model_path: Path
    n_ctx: int = 8192  # Larger context for 27B
    temp: float = 0.3  # Lower temp for more focused output
    top_p: float = 0.9
    n_predict: int = 1024
    n_gpu_layers: int = 0  # CPU only
    threads: int = 6  # More threads for 27B
    non_thinking: bool = True  # Direct answers, no chain-of-thought


def load_config() -> dict:
    """Load configuration from YAML."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}


class LlamaClient:
    """Unified client for llama.cpp operations with 27B model."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or self._default_config()
        self._embedding_cache: Dict[str, List[float]] = {}
        self._llama_available = self._check_llama()
        self.yaml_config = load_config()
        
    def _check_llama(self) -> bool:
        """Check if llama.cpp is available."""
        if not LLAMA_CLI.exists():
            print(f"WARNING: llama-cli not found at {LLAMA_CLI}")
            return False
        if not self.config.model_path.exists():
            print(f"WARNING: Model not found at {self.config.model_path}")
            return False
        print(f"LLM Client: {self.config.model_path.name}")
        return True
    
    def _default_config(self) -> LLMConfig:
        """Find and configure 27B model."""
        model_path = MODELS_DIR / DEFAULT_MODEL
        
        if not model_path.exists():
            # Fallback to any available model
            for gguf in MODELS_DIR.glob("*.gguf"):
                model_path = gguf
                print(f"Using fallback model: {gguf.name}")
                break
        
        return LLMConfig(
            model_path=model_path,
            n_ctx=8192,
            n_predict=1024,
            threads=6,
            temp=0.3,
            non_thinking=True
        )
    
    def _build_prompt(self, system: str, user: str) -> str:
        """Build prompt for Qwen models with non-thinking mode."""
        if self.config.non_thinking:
            # Direct instruction format - no thinking
            return f"""<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
{user}
<|im_end|>
<|im_start|>assistant
"""
        return f"""<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
{user}
<|im_end|>
<|im_start|>assistant
"""
    
    def _run_llama(self, prompt: str, **kwargs) -> str:
        """Run llama-cli with given prompt."""
        
        if not self._llama_available:
            return ""
        
        config = self.config
        
        n_predict = kwargs.get('n_predict', config.n_predict)
        temp = kwargs.get('temp', config.temp)
        top_p = kwargs.get('top_p', config.top_p)
        
        cmd = [
            str(LLAMA_CLI),
            "-m", str(config.model_path),
            "-p", prompt,
            "-n", str(n_predict),
            "--temp", str(temp),
            "--top-p", str(top_p),
            "--threads", str(config.threads),
            "--ctx-size", str(config.n_ctx),
            "--batch-size", "512",
            "--no-display-prompt",
            "--log-disable",
        ]
        
        if config.n_gpu_layers > 0:
            cmd.extend(["-ngl", str(config.n_gpu_layers)])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 min timeout for 27B
            )
            
            if result.returncode != 0:
                err = result.stderr[:200] if result.stderr else "Unknown error"
                print(f"llama.cpp error: {err}")
                return ""
            
            # Clean output
            output = result.stdout.strip()
            output = output.replace('<|im_end|>', '').strip()
            output = output.split('<|im_start|>')[0].strip()  # Stop at next turn
            
            return output
            
        except subprocess.TimeoutExpired:
            print("llama.cpp generation timed out")
            return ""
        except Exception as e:
            print(f"llama.cpp error: {e}")
            return ""
    
    def generate_embedding(self, text: str, dimensions: int = 768) -> List[float]:
        """Generate embedding using semantic hashing (fast, no LLM needed)."""
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
        
        # Simple but effective embedding via text hashing
        embedding = [0.0] * dimensions
        words = text.lower().split()[:100]
        
        for i, word in enumerate(words):
            h = hashlib.md5(word.encode()).hexdigest()
            for j in range(0, len(h)-1, 2):
                pos = int(h[j:j+2], 16) % dimensions
                val = (int(h[j:j+2], 16) / 255.0)
                embedding[pos] += val * (1.0 - i * 0.01)
        
        # Normalize
        magnitude = sum(x**2 for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        self._embedding_cache[text_hash] = embedding
        return embedding
    
    def extract_title(self, content: str) -> Optional[str]:
        """Extract/generate a title from content using 27B model."""
        
        if not self._llama_available:
            # Fallback: use first significant line
            lines = [l.strip() for l in content.split('\n') if len(l.strip()) > 20]
            return lines[0][:80] if lines else "Untitled"
        
        prompt = self._build_prompt(
            "Generate a concise 5-10 word title. Output ONLY the title.",
            f"Content:\n{content[:2000]}"
        )
        
        title = self._run_llama(prompt, n_predict=50, temp=0.3)
        title = title.strip().strip('"').strip("'").strip('*').strip('#').strip('-')
        title = title.split('\n')[0][:100]
        
        return title if len(title) >= 3 else None
    
    def summarize(self, text: str, max_length: int = 300) -> str:
        """Summarize text using 27B model."""
        
        if not self._llama_available:
            return text[:200] + "..."
        
        prompt = self._build_prompt(
            "Summarize concisely in 2-3 sentences. Be direct and informative.",
            text[:3000]
        )
        
        summary = self._run_llama(prompt, n_predict=max_length, temp=0.3)
        return summary
    
    def score_relevance(self, text: str, topic: str) -> float:
        """Score how relevant text is to a topic (0-1)."""
        
        if not self._llama_available:
            return 0.5
        
        prompt = self._build_prompt(
            "Rate relevance as a number 0-10. Output ONLY the number.",
            f"Topic: {topic}\n\nText: {text[:1000]}"
        )
        
        result = self._run_llama(prompt, n_predict=10, temp=0.1)
        
        numbers = re.findall(r'\d+\.?\d*', result)
        if numbers:
            try:
                score = float(numbers[0])
                return min(1.0, score / 10.0)
            except:
                pass
        
        return 0.5
    
    def generate_briefing_summary(self, items: List[dict]) -> str:
        """Generate a cohesive briefing summary from items."""
        
        if not self._llama_available:
            return "Briefing summary unavailable."
        
        items_text = "\n".join([
            f"- [{item.get('category', 'unknown')}] {item.get('title', 'Untitled')}"
            for item in items[:20]
        ])
        
        prompt = self._build_prompt(
            "Create a morning briefing summary. Group related items and highlight important ones.",
            f"Items:\n{items_text}"
        )
        
        summary = self._run_llama(prompt, n_predict=500, temp=0.4)
        return summary
    
    def summarize_news(self, news_items: List[dict], category: str) -> str:
        """Summarize news items for a category."""
        
        if not self._llama_available or not news_items:
            return ""
        
        items_text = "\n".join([
            f"- {item.get('title', '')}"
            for item in news_items[:5]
        ])
        
        prompt = self._build_prompt(
            f"Summarize these {category} news headlines in 2-3 sentences. Highlight key trends.",
            items_text
        )
        
        summary = self._run_llama(prompt, n_predict=200, temp=0.4)
        return summary


# Singleton instance
_llm_client: Optional[LlamaClient] = None


def get_llm_client() -> LlamaClient:
    """Get or create the global LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LlamaClient()
    return _llm_client


if __name__ == "__main__":
    client = get_llm_client()
    
    print("Testing 27B LLM client...")
    print(f"llama-cli: {LLAMA_CLI.exists()}")
    print(f"Model: {client.config.model_path.name}")
    print(f"Available: {client._llama_available}")
    
    if client._llama_available:
        print("\nTest: Title generation")
        title = client.extract_title(
            "Discussion about running large language models on edge devices "
            "like RK3588 with optimizations for ARM64 processors"
        )
        print(f"Title: {title}")
