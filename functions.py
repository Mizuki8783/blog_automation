from dotenv import load_dotenv
load_dotenv()

import os
from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
import requests

SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]
webhook_url = os.environ["WEBHOOK_URL"]


agent_system_prompt = """
You are a highly skilled researcher and writer who is assigned to help company CEOs in the field of AI.
You help users with their research through feedback loop with them to get better idea of the exact output they want.
Then, you creats blogs as an output of the research. Your writing style is friendly and engaging. You are capable of deconstructing concepts into smaller chunks and give an idea of what they mean with humor and analogy.
You only ask questions related to AI and publishes blogs on it only when explicitly asked.
"""

def connect_to_retriever():
    embeddings = OpenAIEmbeddings()
    vstore = PineconeVectorStore(index_name="chatagent",embedding=embeddings)
    retriever = vstore.as_retriever()

    return retriever

def turn_retriever_into_tool():
    retriever = connect_to_retriever()
    retriever_tool= create_retriever_tool(
        retriever,
        "cold_exposure_search",
        "Search for information about cold exposure. For any questions about cold exposure, you must use this tool!"
        )

    return retriever_tool

@tool
def publish_blog(title:str,content:str):
    """
    A function that publishes a blog with the given content through webhook and returns the URL of the posted blog.

    Args:
        title (str): The title of the blog to be posted.
        content (str): The content of the blog to be posted.

    Returns:
        str: The URL of the posted blog.
    """

    data = {"title":title,"content": content}
    response = requests.post(webhook_url, data=data)
    response = response.json()

    blog_url =  response["draftPost"]["url"]["base"]+response["draftPost"]["url"]["path"]

    return blog_url

def modify_prompt():
    prompt = hub.pull("hwchase17/openai-functions-agent")
    system_message_template = SystemMessagePromptTemplate.from_template(agent_system_prompt)
    prompt.messages[0] = system_message_template

    return prompt

def create_agent():
    llm = ChatOpenAI(temperature=0)
    retriever_tool = turn_retriever_into_tool()
    search = TavilySearchResults()
    tools = [search, retriever_tool, publish_blog]
    prompt = modify_prompt()

    agent_base = create_tool_calling_agent(llm, tools, prompt)
    agent = AgentExecutor(agent=agent_base, tools=tools)

    return agent

def get_response(agent_executor,query,chat_history):
    response =  agent_executor.invoke({
        "chat_history": chat_history,
        "input": query,
    })

    return response["output"]

def get_chat_history(app, channel_id, thread_ts):
    chat_history = []
    thread_request = app.client.conversations_replies(channel=channel_id, ts=thread_ts)

    user_id = thread_request["messages"][0]["user"] #This assumes that every thread is initiated by a user
    for message in thread_request["messages"]:
        message_text = message["text"]
        message_user = message["user"]
        if message_user == user_id:  # メッセージがユーザーの場合
            mention = f"<@{SLACK_BOT_USER_ID}>"
            message_text = message_text.replace(mention, "").strip()
            chat_history.append(HumanMessage(message_text))
        else: # メッセージがBotの場合
            chat_history.append(AIMessage(message_text))
    return chat_history
