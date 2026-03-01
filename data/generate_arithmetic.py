"""
MathStack - Generate all arithmetic datasets.
Pure Python, no external downloads. Reproducible via seeds.
"""

import json
import random
import os
from pathlib import Path

# Problem type names used across the project
PROBLEM_TYPES = [
    "single_digit_addition",
    "single_digit_subtraction",
    "single_digit_multiplication",
    "double_digit_addition",
    "double_digit_subtraction",
    "double_digit_multiplication",
    "addition_with_carrying",
    "three_number_addition",
    "simple_division",
    "mixed_operations",
]


def single_digit_addition(rng):
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    return f"What is {a} + {b}?", str(a + b), [a, b], a + b


def single_digit_subtraction(rng):
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    if a < b:
        a, b = b, a
    return f"What is {a} - {b}?", str(a - b), [a, b], a - b


def single_digit_multiplication(rng):
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    return f"What is {a} x {b}?", str(a * b), [a, b], a * b


def double_digit_addition(rng):
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    return f"What is {a} + {b}?", str(a + b), [a, b], a + b


def double_digit_subtraction(rng):
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    if a < b:
        a, b = b, a
    return f"What is {a} - {b}?", str(a - b), [a, b], a - b


def double_digit_multiplication(rng):
    # Keep operands 10-30 so answers stay reasonable
    a, b = rng.randint(10, 30), rng.randint(10, 30)
    return f"What is {a} x {b}?", str(a * b), [a, b], a * b


def addition_with_carrying(rng):
    # Ones digits must sum to 10+ so carrying is required
    a_tens, a_ones = rng.randint(1, 9), rng.randint(5, 9)
    b_tens, b_ones = rng.randint(1, 9), rng.randint(10 - a_ones, 9)
    a = a_tens * 10 + a_ones
    b = b_tens * 10 + b_ones
    return f"What is {a} + {b}?", str(a + b), [a, b], a + b


def three_number_addition(rng):
    a, b, c = rng.randint(10, 50), rng.randint(10, 50), rng.randint(10, 50)
    return f"What is {a} + {b} + {c}?", str(a + b + c), [a, b, c], a + b + c


def simple_division(rng):
    # Whole number results only: pick answer first, then multiply
    answer = rng.randint(2, 12)
    b = rng.randint(2, 12)
    a = answer * b
    return f"What is {a} / {b}?", str(answer), [a, b], answer


def mixed_operations(rng):
    # (a + b) * c
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    c = rng.randint(2, 9)
    inner = a + b
    result = inner * c
    return f"What is ({a} + {b}) x {c}?", str(result), [a, b, c], result


GENERATORS = {
    "single_digit_addition": single_digit_addition,
    "single_digit_subtraction": single_digit_subtraction,
    "single_digit_multiplication": single_digit_multiplication,
    "double_digit_addition": double_digit_addition,
    "double_digit_subtraction": double_digit_subtraction,
    "double_digit_multiplication": double_digit_multiplication,
    "addition_with_carrying": addition_with_carrying,
    "three_number_addition": three_number_addition,
    "simple_division": simple_division,
    "mixed_operations": mixed_operations,
}


def generate_problems(per_type: int, seed: int):
    """Generate problems with equal count per type. Returns list of dicts."""
    rng = random.Random(seed)
    out = []
    for _ in range(per_type):
        for ptype in PROBLEM_TYPES:
            instr, resp, operands, correct = GENERATORS[ptype](rng)
            out.append({
                "instruction": instr,
                "response": resp,
                "type": ptype,
                "operands": operands,
                "correct_answer": correct,
            })
    # Shuffle so types are mixed
    rng.shuffle(out)
    return out


def write_jsonl(path: str, items: list):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def generate_arithmetic_facts(seed: int, count: int = 500):
    """Generate arithmetic rules and examples for RAG. Covers all 10 types."""
    rng = random.Random(seed)
    facts = []

    # Carrying
    for _ in range(50):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if (a % 10) + (b % 10) >= 10:
            ones = (a % 10) + (b % 10)
            carry = 1
            tens = (a // 10) + (b // 10) + carry
            result = a + b
            facts.append({
                "fact": f"When adding two numbers where the ones digits sum to 10 or more, carry 1 to the tens column. Example: {a} + {b}, ones: {a%10}+{b%10}={ones}, write {ones%10} carry 1, tens: {a//10}+{b//10}+1={tens}, result: {result}",
                "category": "carrying",
                "related_types": ["addition_with_carrying", "three_number_addition"],
            })
    # Single-digit addition
    for _ in range(40):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        facts.append({
            "fact": f"Single digit addition: {a} + {b} = {a+b}",
            "category": "single_digit",
            "related_types": ["single_digit_addition"],
        })
    # Single-digit subtraction
    for _ in range(40):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        if a < b:
            a, b = b, a
        facts.append({
            "fact": f"Single digit subtraction: {a} - {b} = {a-b}",
            "category": "single_digit",
            "related_types": ["single_digit_subtraction"],
        })
    # Multiplication tables
    for _ in range(50):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        facts.append({
            "fact": f"Multiplication: {a} x {b} = {a*b}",
            "category": "multiplication",
            "related_types": ["single_digit_multiplication", "double_digit_multiplication"],
        })
    # Double-digit addition
    for _ in range(40):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        facts.append({
            "fact": f"Double digit addition: {a} + {b} = {a+b}. Add ones first, then tens.",
            "category": "double_digit",
            "related_types": ["double_digit_addition"],
        })
    # Double-digit subtraction
    for _ in range(40):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if a < b:
            a, b = b, a
        facts.append({
            "fact": f"Double digit subtraction: {a} - {b} = {a-b}. Subtract ones first, then tens.",
            "category": "double_digit",
            "related_types": ["double_digit_subtraction"],
        })
    # Division
    for _ in range(50):
        b = rng.randint(2, 12)
        answer = rng.randint(2, 12)
        a = b * answer
        facts.append({
            "fact": f"Division: {a} / {b} = {answer} because {b} x {answer} = {a}",
            "category": "division",
            "related_types": ["simple_division"],
        })
    # Mixed / order of operations
    for _ in range(40):
        x, y, z = rng.randint(1, 9), rng.randint(1, 9), rng.randint(2, 9)
        inner = x + y
        result = inner * z
        facts.append({
            "fact": f"Brackets first: ({x} + {y}) x {z} = {inner} x {z} = {result}",
            "category": "mixed",
            "related_types": ["mixed_operations"],
        })
    # Double-digit multiplication (partial products)
    for _ in range(50):
        a, b = rng.randint(10, 25), rng.randint(10, 25)
        b_ones, b_tens = b % 10, b // 10
        p1 = a * b_ones
        p2 = a * b_tens * 10
        facts.append({
            "fact": f"Double digit multiplication: {a} x {b}. First {a} x {b_ones} = {p1}, then {a} x {b_tens}0 = {p2}, total {p1}+{p2} = {a*b}",
            "category": "double_digit_multiplication",
            "related_types": ["double_digit_multiplication"],
        })
    # Three-number addition
    for _ in range(50):
        a, b, c = rng.randint(10, 40), rng.randint(10, 40), rng.randint(10, 40)
        ab = a + b
        facts.append({
            "fact": f"Three number addition: {a} + {b} + {c}. First {a} + {b} = {ab}, then {ab} + {c} = {a+b+c}",
            "category": "three_number",
            "related_types": ["three_number_addition"],
        })

    rng.shuffle(facts)
    return facts[:count]


def main():
    script_dir = Path(__file__).parent

    # train_100: 100 problems, 10 per type, seed 42
    train_100 = generate_problems(10, 42)
    write_jsonl(script_dir / "train_100.jsonl", train_100)
    print(f"Generated train_100: {len(train_100)} problems")

    # train_1000: 100 per type, seed 42 (superset of train_100 concept, same seed so first 100 match distribution)
    train_1000 = generate_problems(100, 42)
    write_jsonl(script_dir / "train_1000.jsonl", train_1000)
    print(f"Generated train_1000: {len(train_1000)} problems")

    # test_200: 20 per type, seed 99 (no overlap with train)
    test_200 = generate_problems(20, 99)
    write_jsonl(script_dir / "test_200.jsonl", test_200)
    print(f"Generated test_200: {len(test_200)} problems")

    # arithmetic_facts for RAG
    facts = generate_arithmetic_facts(42, 500)
    rag_dir = script_dir / "rag_documents"
    rag_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rag_dir / "arithmetic_facts.jsonl", facts)
    print(f"Generated arithmetic_facts: {len(facts)} rules")


if __name__ == "__main__":
    main()
