import random

def nshot_chats(nshot_data: list, n: int, question: str, prompt: str = "Let's think step by step.") -> dict:

    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        return f'Answer: {s}'

    chats = [
        # {
        #     "role": "system",
        #     "content": "You are a grade school math problem solver. At the end, you MUST write the answer as an integer after '####'. Let's think step by step.",
        # },
    ]

    random.seed(42)
    for qna in random.sample(nshot_data, n):
        chats.append(
            {"role": "user", "content": question_prompt(qna["question"])})
        chats.append(
            {"role": "assistant", "content": answer_prompt(qna["answer"])})

    chats.append({
        "role": "user", 
        "content": question_prompt(question) + " " 
        + prompt 
        + " At the end, you MUST write the answer as an integer after '####'."
        })

    return chats


def extract_num_from_ans(answer: str):

    answer = answer.split('####')[-1].strip()

    for remove_char in [',', '$', '%', 'g']:
        answer = answer.replace(remove_char, '')

    try:
        return int(answer)
    except ValueError:
        return answer
