from formatters import Example, format_example, format_prompt_only

def test_template_contains_markers():
    ex = Example("Say hi","Hello!")
    s = format_example(ex)
    # The section markers are what the model keys off to know where the answer starts.
    assert "### Instruction:" in s and "### Response:" in s
    p = format_prompt_only("Explain transformers.")
    # Prompt-only text must end right where the response should begin (optionally with the </s> marker).
    assert p.endswith("### Response:\n") or p.endswith("### Response:\n</s>")