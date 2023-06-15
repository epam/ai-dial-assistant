from typing import Dict, Tuple

from utils.printing import print_info
from utils.text import indent

# Prices in dollars per 1000 tokens from https://openai.com/pricing#language-models
MODEL_PRICES: Dict[str, Tuple[float, float] | float] = {
    # ChatGPT
    "gpt-3.5-turbo": 0.002,
    "gpt-3.5-turbo-0301": 0.002,
    # Davinci
    "davinci": 0.02,
    "text-davinci-003": 0.02,
    "text-davinci-002": 0.02,
    "code-davinci-002": 0.02,
    # GPT-4 Completion
    "gpt-4": (0.03, 0.06),
    "gpt-4-0314": (0.03, 0.06),
    "gpt-4-32k": (0.06, 0.12),
    "gpt-4-32k-0314": (0.06, 0.12),
}


class TokenCounter:
    # token_usage for the latest request
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_cost: float = 0

    # Cumulative metrics
    tokens_so_far: int = 0
    number_of_requests: int = 0
    total_cost: float = 0

    def print(self):
        print_info("\nToken usage:\n" + indent(str(self), 2))

    def update(self, model_name: str, prompt_ts: int, completion_ts: int):
        self.prompt_tokens = prompt_ts
        self.completion_tokens = completion_ts
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.request_cost = (
            self.get_token_price(model_name, 0) * self.prompt_tokens
            + self.get_token_price(model_name, 1) * self.completion_tokens
        )

        self.tokens_so_far += self.total_tokens
        self.number_of_requests += 1
        self.total_cost += self.request_cost

    @staticmethod
    def get_token_price(model_name: str, idx: int):
        # Model price is per 1000 tokens
        price: Tuple[float, float] | float = MODEL_PRICES.get(model_name, 0.0)
        price1: float = price[idx] if isinstance(price, tuple) else price
        return price1 / 1000.0

    def __str__(self) -> str:
        pairs = [
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
            ("total_tokens", f"{self.total_tokens:<7} ({self.request_cost:.3f}$)"),
            ("tokens_so_far", f"{self.tokens_so_far:<7} ({self.total_cost:.3f}$)"),
            ("number_of_requests", self.number_of_requests),
        ]
        longest_key = max(len(pair[0]) for pair in pairs)

        lines = [f"{pair[0]:<{longest_key}} : {pair[1]}" for pair in pairs]

        return "\n".join(lines)
