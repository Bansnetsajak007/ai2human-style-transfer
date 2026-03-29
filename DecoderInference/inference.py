from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "D:\HumanizerResearch\DecoderInference\HumanizerModel",
    tokenizer_name = "D:\HumanizerResearch\DecoderInference\HumanizerTokenizer",
    max_seq_length = 768,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)

inference_prompt = """### Instruction:
Rewrite the following AI-generated text to sound natural and human-written.
Follow these rules strictly:
- Keep the same meaning and facts as the original
- Use a conversational, flowing tone — avoid stiff or robotic phrasing
- Vary sentence length naturally (mix short and long sentences)
- Remove overly formal or academic vocabulary where possible
- Do NOT add new information, opinions, or examples not in the original
- Do NOT remove any key points from the original

### Input:
{}

### Response:
"""

def humanize(ai_text):
    inputs = tokenizer(
        [inference_prompt.format(ai_text.strip())],
        return_tensors = "pt"
    ).to("cpu")

    streamer = TextStreamer(tokenizer, skip_prompt=True)

    _ = model.generate(
        **inputs,
        streamer          = streamer,
        max_new_tokens    = 512,    # increased since your inputs are ~600 tokens
        do_sample         = True,
        temperature       = 0.7,
        repetition_penalty = 1.1,
        use_cache         = True,
    )

humanize("Your AI text here...")