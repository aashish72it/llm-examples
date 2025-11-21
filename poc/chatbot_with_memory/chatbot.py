import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()


GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MAX_TOKENS = os.environ["MAX_TOKENS_HISTORY"]
LLM_MODEL = os.environ["LLM_MODEL"]
SYSTEM_PROMPT = os.environ["SYSTEM_PROMPT"] 

llm = ChatGroq(
    model = LLM_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=int(MAX_TOKENS),
)

parser = StrOutputParser()


prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"), 
    ("user", "{input}"),                       
])


chain = prompt | llm | parser


def chatbot():
    history = []  # [("user", "..."), ("assistant", "..."), ...]
    print("Genie is ready to chat! How can I help you today? (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Exiting chatbot. Goodbye!")
            break

        llm_response = chain.invoke({"input": user_input, "history": history})
        print(f"Genie: {llm_response}\n")

        # Update history
        ## only below values allowed: 
        # ['human', 'user', 'ai', 'assistant', 'function', 'tool', 'system', 'developer']
        ## otherwise you will get below error
        # ValueError: Unexpected message type: 'llm'. 
        # Use one of 'human', 'user', 'ai', 'assistant', 'function', 'tool', 'system', or 'developer'.  
        history.append(("user", user_input))
        history.append(("assistant", llm_response))


if __name__ == "__main__":
    chatbot()


