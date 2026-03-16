# ==========================================================
# AI Developer Assistant Agent
# Code Understanding + Debugging + Documentation
# ==========================================================

# Install dependencies
# pip install langchain langchain-openai

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool


# ==========================================================
# Code Analysis Tool
# ==========================================================

def analyze_code(code):

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
You are an expert AI developer assistant.

Analyze the following code.

Provide:

1. Code Explanation
2. Bugs or Errors
3. Corrected Code
4. Documentation (docstring)
5. Suggested Improvements
6. Possible Unit Tests

Code:
{code}
"""

    response = llm.invoke(prompt)

    return response.content


# ==========================================================
# Documentation Tool
# ==========================================================

def generate_docs(code):

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
Generate professional documentation for the following code.

Include:

Function description
Parameters
Return values
Example usage

Code:
{code}
"""

    response = llm.invoke(prompt)

    return response.content


# ==========================================================
# Tools
# ==========================================================

tools = [

    Tool(
        name="CodeAnalyzer",
        func=analyze_code,
        description="Analyze code and detect bugs or improvements."
    ),

    Tool(
        name="DocGenerator",
        func=generate_docs,
        description="Generate documentation for code."
    )

]


# ==========================================================
# LLM
# ==========================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ==========================================================
# Memory
# ==========================================================

memory = ConversationBufferMemory(memory_key="chat_history")


# ==========================================================
# Agent Setup
# ==========================================================

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)


# ==========================================================
# Run Dev Assistant
# ==========================================================

if __name__ == "__main__":

    print("\nPaste your code below (type END to finish):\n")

    lines = []

    while True:
        line = input()
        if line == "END":
            break
        lines.append(line)

    code = "\n".join(lines)

    result = agent.run(
        f"Analyze this code and improve it:\n\n{code}"
    )

    print("\n========== AI DEV ASSISTANT ==========\n")
    print(result)