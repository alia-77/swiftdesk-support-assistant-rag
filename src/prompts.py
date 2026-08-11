def zero_shot_prompt(customer_issue):
    return f"""
You are an IT support assistant.

Write a short, clear, polite support reply to the customer.

Customer issue:
{customer_issue}

Reply:
"""


def few_shot_prompt(customer_issue, examples):
    examples_text = "\n\n".join(
        [
            f"Customer issue:\n{example['customer_issue']}\n"
            f"Support reply:\n{example['reference_reply']}"
            for example in examples
        ]
    )

    return f"""
You are an IT support assistant.

Use the examples below as guidance for writing a short,
clear, polite support reply.

Examples:

{examples_text}

Now write a support reply for this customer issue:

{customer_issue}

Reply:
"""


def reasoned_prompt(customer_issue):
    return f"""
You are an IT support assistant.

Analyze the customer's issue carefully and identify the
main problem before drafting the response.

Then write a short, clear, polite support reply.
Do not expose your internal reasoning.

Customer issue:
{customer_issue}

Reply:
"""


def rag_prompt(customer_issue, examples):
    examples_text = "\n\n".join(
        [
            f"Previous customer issue:\n{example['customer_issue']}\n"
            f"Approved support reply:\n{example['reference_reply']}"
            for example in examples
        ]
    )

    return f"""
You are an IT support assistant.

Draft a short, clear, polite support reply for the customer.

Use the previous support examples below as reference.
Base your response on information that is relevant to the
customer's issue. Do not invent technical details.

Previous support examples:

{examples_text}

Customer issue:

{customer_issue}

Reply:
"""

