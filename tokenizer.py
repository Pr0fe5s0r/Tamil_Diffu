from transformers import AutoTokenizer

def main():
    # Load the multilingual tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base")
    # List all special tokens and their vocab IDs
    print("Special tokens in the tokenizer:")
    special_tokens = tokenizer.special_tokens_map
    for name, token in special_tokens.items():
        if token is not None:
            if isinstance(token, list):
                for t in token:
                    token_id = tokenizer.convert_tokens_to_ids(t)
                    print(f"{name}: '{t}' (ID: {token_id})")
            else:
                token_id = tokenizer.convert_tokens_to_ids(token)
                print(f"{name}: '{token}' (ID: {token_id})")
    # Also print all additional special tokens if present
    if hasattr(tokenizer, "additional_special_tokens") and tokenizer.additional_special_tokens:
        for t in tokenizer.additional_special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(t)
            print(f"additional_special_token: '{t}' (ID: {token_id})")

if __name__ == "__main__":
    main()
