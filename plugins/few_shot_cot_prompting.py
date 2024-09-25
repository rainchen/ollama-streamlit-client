# COT prompting paper: https://arxiv.org/abs/2201.11903
# fmt: off
def process(model: str, messages: list, system_prompt: str, model_params: dict, generate_func: callable, **kwargs):
    few_shot_cot_prompt = """
You solve problem thinking step by step.
Example:
Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?
Let's think step by step.
Answer:
Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.
His team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers
They scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.
All together his team scored 50+24+10= 84 points
Mark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.
His opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.
They also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.
All together Mark's opponents scored 100+12+5=117 points
The total score for the game is both team's scores added together, so it is 84+117=201 points
The answer is 201
""".strip()
    # add cot_prompt into system prompt once only
    if few_shot_cot_prompt not in system_prompt:
        if system_prompt.strip() != "":
            system_prompt = system_prompt + "\n" + few_shot_cot_prompt
        else:
            system_prompt = few_shot_cot_prompt

    zero_shot_cot_prompt = "Let's think step by step."
    messages = messages + [{"role": "user", "content": zero_shot_cot_prompt}]

    for chunk in generate_func(model, messages, system_prompt, model_params):
        yield chunk

    # st.session_state.messages.extend(messages)
