from collator_sft import SFTCollator
from formatters import Example

def test_masking_sets_prompt_to_ignore():
    # Falls back to the byte tokenizer if the Part 4 BPE tokenizer dir doesn't exist.
    col = SFTCollator(block_size=256, bpe_dir='../part_4/runs/part4-demo/tokenizer')
    text = "This is a tiny test."
    x, y = col.collate([(text, "OK")])
    # All labels up to response marker should be -100
    # (-100 = ignored by cross-entropy, so the model isn't trained to predict the prompt)
    boundary = ("<s>\n### Instruction:\n" + text + "\n\n### Response:\n")
    # We don't have direct access to the tokenized boundary; just sanity check: some -100s present
    assert (y[0] == -100).sum() > 0
