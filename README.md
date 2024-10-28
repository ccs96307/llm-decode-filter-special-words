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
response = custom_generate_with_fsm_filter(
    input_ids=inputs.input_ids,
    fsm_processor=fsm_processor,
    max_length=50,
)
```


### Function Explanation

```python
def custom_generate_with_fsm_filter(
    input_ids: torch.Tensor,
    fsm_processor: FSMProcessor,
    max_length: int = 20,
) -> str:
    # Historical mask list used to record masked tokens at each decoding step
    masked_tokens_history = {}
    past_key_values = None
    steps = 0

    while steps < max_length:
        steps += 1

        # Generate new token with kv cache
        outputs = model(input_ids, past_key_values=past_key_values, return_dict=True, use_cache=True)
        logits = outputs.logits[:, -1, :]

        # Update kv cache
        past_key_values = outputs.past_key_values

        # Check if there are already masked tokens at the current step
        if steps in masked_tokens_history:
            for masked_token_id in masked_tokens_history[steps]:
                logits[:, masked_token_id] = -float("inf")
        else:
            masked_tokens_history[steps] = set()

        # Decode the generated token
        generated_token_id = torch.argmax(logits, dim=-1).item()
        combined_ids = torch.cat((input_ids, torch.tensor([[generated_token_id]], device=input_ids.device)), dim=-1)

        # Check FSM for sensitive sequences
        if fsm_processor.detect(generated_token_id):
            # Detected a sensitive sequence, initiate rollback
            rollback_length = fsm_processor.partial_match_state + 1 if fsm_processor.partial_match_state is not None else 1
            steps = steps - rollback_length + 1
            rollbacks_ids = combined_ids[:, :-rollback_length]
            input_ids = rollbacks_ids
            print(f"Rollback detected. Rolling back from step {steps + rollback_length} to step {steps}")

            # Reset FSM state
            fsm_processor.curr_state = 0
            fsm_processor.partial_match_state = None

            # Reset past_key_values when rolling back
            past_key_values = None

            # Recalculate logits based on rolled-back sequence
            outputs = model(input_ids, return_dict=True, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # Mask the first token of the sensitive sequence
            first_token_id = generated_token_id
            print(f"Masking token id: {first_token_id}, Masking token: {tokenizer.decode(first_token_id)}")

            # Update the historical mask list to record the token at this step
            masked_tokens_history[steps].add(first_token_id)

            for masked_token_id in masked_tokens_history[steps]:
                logits[:, masked_token_id] = -float("inf")

            # Generate the token again after masking
            generated_token_id = torch.argmax(logits, dim=-1).item()

        # Update input_ids with the generated token
        input_ids = torch.cat((input_ids, torch.tensor([[generated_token_id]], device=input_ids.device)), dim=1)

        print(f"Step {steps}: ID: {generated_token_id} Generated token: {tokenizer.decode(generated_token_id)}")

        if generated_token_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=False)
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
- [X] Custom decode prototype
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

