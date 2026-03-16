# ==========================================================
# Autonomous Founder / CEO Research Agent
# Uses LangChain + Web Search + Memory
# ==========================================================

# Install dependencies first:
# pip install langchain langchain-openai duckduckgo-search beautifulsoup4 requests

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType


# ==========================================================
# Web Search Tool
# ==========================================================

def web_search(query):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(r["href"])
    return str(results)


# ==========================================================
# Webpage Reader Tool
# ==========================================================

def read_webpage(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = [p.text for p in soup.find_all("p")]
        content = " ".join(paragraphs)
        return content[:4000]
    except:
        return "Unable to read webpage."


# ==========================================================
# Define Tools
# ==========================================================

tools = [

    Tool(
        name="WebSearch",
        func=web_search,
        description="Search the internet for information about founders or CEOs."
    ),

    Tool(
        name="ReadWebpage",
        func=read_webpage,
        description="Extract content from a webpage URL."
    )

]


# ==========================================================
# LLM Setup
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
# Agent Initialization
# ==========================================================

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)


# ==========================================================
# Research Task
# ==========================================================

def run_research(person):

    prompt = f"""
You are an autonomous research agent.

Research the founder or CEO: {person}

Steps:
1. Search the web
2. Open relevant pages
3. Collect context
4. Summarize findings

Return structured output:

Name
Company
Background
Career History
Major Achievements
Leadership Style
Interesting Facts
"""

    result = agent.run(prompt)

    print("\n========== FINAL RESEARCH REPORT ==========\n")
    print(result)


# ==========================================================
# Run Agent
# ==========================================================

if __name__ == "__main__":

    person = input("Enter Founder or CEO name: ")

    run_research(person)