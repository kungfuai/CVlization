"""
Test the fine-tuned Moondream2 model.
"""

import argparse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./outputs/moondream2_ft', help='Path to fine-tuned model')
    parser.add_argument('--image', required=True, help='Path to test image')
    parser.add_argument('--question', default='What is the text in this captcha image?', help='Question to ask')
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading image: {args.image}")
    image = Image.open(args.image)

    print(f"Question: {args.question}")

    # Encode image and ask question
    encoded = model.encode_image(image)
    answer = model.answer_question(encoded, args.question, tokenizer)

    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
