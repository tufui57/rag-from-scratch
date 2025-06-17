from prepare import llm
from rag_from_scratch_5 import get_retriever
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


### Part 8: Step Back
def main(question = "What is task decomposition for LLM agents?"):
    print(f"Question\n {question}")
    retriever=get_retriever()
    examples = [
        {
            "input": "Could the members of The Police perform lawful arrests?",
            "output": "what can the members of The Police do?",
        },
        {
            "input": "Jan Sindel’s was born in what country?",
            "output": "what is Jan Sindel’s personal history?",
        },
    ]
    # We now transform these to example messages
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
            ),
            # Few shot examples
            few_shot_prompt,
            # New question
            ("user", "{question}"),
        ]
    )
    generate_queries_step_back = prompt | llm | StrOutputParser()
    generate_queries_step_back.invoke({"question": question})
    # Response prompt 
    response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

    # {normal_context}
    # {step_back_context}

    # Original Question: {question}
    # Answer:"""
    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

    chain = (
        {
            # Retrieve context using the normal question
            "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
            # Retrieve context using the step-back question
            "step_back_context": generate_queries_step_back | retriever,
            # Pass on the question
            "question": lambda x: x["question"],
        }
        | response_prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"question": question})


if __name__ == "__main__":
    res=main()
    print(res)