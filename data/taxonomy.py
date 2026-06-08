# We need paired examples of safe vs unsafe instruction-following. The goal is to teach the model to refuse harmful requests.

UNSAFE_CATEGORIES = [
    "violence", "deception", "illegal_activity", "privacy_violation", "self-harm",
]

SAFE_CATEGORIES = [
    "cooking", "coding_help", "writing_assistance", "general_knowledge", "math_tutoring",
]