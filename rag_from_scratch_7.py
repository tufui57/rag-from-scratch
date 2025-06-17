from prepare import llm
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from rag_from_scratch_5 import get_retriever


def format_qa_pair(question, answer):
    """Format Q and A pair"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

### Part 7
def main(question = "What are the main components of an LLM-powered autonomous agent system?"):
    print(f"Question\n {question}")
    retriever = get_retriever()
    # Decomposition
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)

    # Chain
    generate_queries_decomposition = ( 
        prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

    questions = generate_queries_decomposition.invoke({"question":question})
    print(f"Question decomposition\n {questions}")

    # Answer recursively
    # Prompt
    template = """Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """

    decomposition_prompt = ChatPromptTemplate.from_template(template)

    q_a_pairs = ""
    for q in questions:
        
        rag_chain = (
        {"context": itemgetter("question") | retriever, 
        "question": itemgetter("question"),
        "q_a_pairs": itemgetter("q_a_pairs")} 
        | decomposition_prompt
        | llm
        | StrOutputParser())

        answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
        q_a_pair = format_qa_pair(q,answer)
        q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair

    return answer


if __name__ == "__main__":
    res=main()
    print(res)