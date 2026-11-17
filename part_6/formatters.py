"""Prompt/response formatting utilities (6.1).
We keep a very simple template with clear separators.
"""
from dataclasses import dataclass

# A "chat template". The model learns during SFT that text after "### Response:" is an answer to
# the instruction, and that "</s>" means "answer finished". At inference we give it everything up
# to "### Response:\n" and let it write the rest. (Real chat models use the same idea with their
# own special tokens, e.g. <|im_start|>assistant.)
# <s> / </s> are BPE special tokens (begin / end of sequence).
template = (
    "<s>\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}</s>"
)

@dataclass
class Example:
    instruction: str
    response: str


def format_example(ex: Example) -> str:
    # Full training string: prompt + answer + end marker.
    return template.format(instruction=ex.instruction.strip(), response=ex.response.strip())


def format_prompt_only(instruction: str) -> str:
    # Empty response -> ends in "### Response:\n</s>". Callers strip the "</s>" so the model
    # continues from right after "### Response:\n".
    return template.format(instruction=instruction.strip(), response="")