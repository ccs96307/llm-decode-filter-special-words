# LLM Decode Filter

This project introduces a tool designed to filter specific sensitive words during the text generation process using large language models (LLMs). The tool allows users to detect certain sensitive words and dynamically roll back the generation to filter out these words, ensuring that the generated text avoids predefined sensitive content while maintaining coherence.

## Features

- **Sensitive Word Detection**: Automatically detects sensitive words during the decoding process.
- **Rollback Mechanism**: If a sensitive word is detected, the generation is rolled back to exclude that word.
- **Token Masking**: After rolling back, the detected sensitive word's tokens are masked to prevent them from being regenerated.
- **Historical Masking**: Tokens that were previously masked are remembered at each decoding step, preventing them from being regenerated in subsequent steps.
- **Flexible Configuration**: Allows users to define a list of sensitive words and manage the decoding process dynamically.

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Hugging Face's `transformers` library

### Install dependencies

```bash
pip install torch transformers
```

## Usage

### Example Code

Below is a sample code snippet showcasing how to use the tool:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Define sensitive words
sensitive_words = ["listen", "example_sensitive_word"]

# Input prompt
input_text = "Let's discuss a topic."

# Tokenize input
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Call the custom generation function
generated_text = custom_generate_with_special_words_filter(
    input_ids=input_ids,
    special_words=sensitive_words,
    max_length=50
)

print(generated_text)
```

### Function Explanation

```python
def custom_generate_with_special_words_filter(
    input_ids: torch.Tensor,
    special_words: List[str],
    max_length: int = 20,
) -> str:
    # Historical mask list used to record masked tokens at each decoding step
    masked_tokens_history = {}

    for step in range(max_length):
        # Generate new tokens
        outputs = model(input_ids, return_dict=True, use_cache=True)
        logits = outputs.logits[:, -1, :]

        # Check if there are already masked tokens at the current step
        if step in masked_tokens_history:
            for token_id in masked_tokens_history[step]:
                logits[:, token_id] = -float("inf")  # Mask previously invalid tokens

        # Generate the token
        generated_token = torch.argmax(logits, dim=-1)
        combined_ids = torch.cat((input_ids, generated_token.unsqueeze(0)), dim=-1)
        combined_text = tokenizer.decode(combined_ids[0], skip_special_tokens=True)

        # Check for each sensitive word
        for special_word in special_words:
            if special_word in combined_text:
                # Convert the sensitive word into tokens
                special_word_tokenized = tokenizer(special_word, return_tensors="pt", add_special_tokens=False).input_ids
                special_word_length = special_word_tokenized.shape[1]
                print(f"Detect special word: {special_word}\n Start roll-back process...")

                # Roll back to before the sensitive word
                rollbacks_ids = combined_ids[:, :-special_word_length]
                input_ids = rollbacks_ids
                print(f"Roll back from {combined_ids.shape[1]} to {rollbacks_ids.shape[1]}")

                # Recalculate logits based on the rolled-back sequence
                outputs = model(input_ids, return_dict=True, use_cache=True)
                logits = outputs.logits[:, -1, :]  # Recalculate logits based on rolled-back input

                # Only mask the first token of the sensitive word
                first_token_id = special_word_tokenized[0, 0]
                print(f"Masking token: {tokenizer.decode(first_token_id)}")
                logits[:, first_token_id] = -float("inf")  # Mask the first token of the sensitive word

                # Update the historical mask list to record the token at this step
                if step not in masked_tokens_history:
                    masked_tokens_history[step] = set()
                masked_tokens_history[step].add(first_token_id)

        # Generate the next token
        new_generated_token = torch.argmax(logits, dim=-1)
        print(f"Generated token: {tokenizer.decode(new_generated_token)}")
        input_ids = torch.cat((input_ids, new_generated_token.unsqueeze(0)), dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
```

### Parameters

- `input_ids`: The tokenized input prompt.
- `special_words`: A list of words you want to filter during generation.
- `max_length`: The maximum number of tokens to be generated.

### How It Works
1. **Token Generation**: For each step, the model generates a new token.
2. **Sensitive Word Detection**: After generating each token, the tool checks whether the combined sequence contains any of the sensitive words.
3. **Rollback and Masking**: If a sensitive word is detected, the generation process is rolled back, and the detected word's first token is masked to prevent it from being regenerated.
4. **Historical Masking**: If a token has already been masked at a particular step, it will remain masked for future steps.

### Output

The tool will return a generated text that avoids the predefined sensitive words, maintaining the fluency and relevance of the generated output.


## TODO List
- [ ] Custom decode prototype
- [ ] Support batch decode
- [ ] Support transformers library rollback
- [ ] Support vLLM


## Contribution

If you'd like to contribute to this project:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License.

