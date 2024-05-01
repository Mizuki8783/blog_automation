import os
import requests
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain import hub
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()
SLACK_BOT_USER_ID = os.getenv("SLACK_BOT_USER_ID")
webhook_url = os.getenv("WEBHOOK_URL")


agent_system_prompt = """
You are a highly skilled researcher and writer in the field of AI. You ONLY answer questions related to AI and do not answer questions out side of the topic.
You help users with their research through feedback loop to get better ideas of the exact output they want.
You also craft blog post based on the research for the users. Here are some of the rules on blog writing.
1. The blog should only be published only after the draft gets approved. Until then, you need to keep modifying the draft.
2. The blog should be written in HTML.
"""

@tool
def publish_blog(title:str,content:str):
    """
    Publishes a blog with the given title and content.

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
    """
    Pulls the template prompt for agents from Langchain hub, updating the system message template,
    and returning the modified prompt.
    """
    prompt = hub.pull("hwchase17/openai-functions-agent")
    system_message_template = SystemMessagePromptTemplate.from_template(agent_system_prompt)
    prompt.messages[0] = system_message_template

    return prompt

def create_agent():
    """
    Creates an agent with the specified tools.

    Returns:
        AgentExecutor: The created agent.
    """
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    search = TavilySearchResults()
    tools = [search, publish_blog]
    prompt = modify_prompt()

    agent_base = create_tool_calling_agent(llm, tools, prompt)
    agent = AgentExecutor(agent=agent_base, tools=tools)

    return agent

def get_response(agent_executor,query,chat_history):
    """
    Retrieves the response from the agent executor based on the provided query and chat history.

    Args:
        agent_executor: The executor for the agent.
        query: The input query for the agent.
        chat_history: The chat history of the conversation.

    Returns:
        The output response from the agent.
    """
    response =  agent_executor.invoke({
        "chat_history": chat_history,
        "input": query,
    })

    return response["output"]

def get_chat_history(app, channel_id, thread_ts):
    """
    Retrieves chat history for a given channel and thread.

    Args:
        app: The Slack app instance.
        channel_id: The ID of the channel.
        thread_ts: The timestamp of the thread.

    Returns:
        list: A list of messages in the chat history.
    """

    chat_history = []
    thread_request = app.client.conversations_replies(channel=channel_id, ts=thread_ts)

    user_id = thread_request["messages"][0]["user"] #This assumes that every thread is initiated by a user
    for message in thread_request["messages"]:
        message_text = message["text"]
        message_user = message["user"]
        # when message is from the user
        if message_user == user_id:
            mention = f"<@{SLACK_BOT_USER_ID}>"
            message_text = message_text.replace(mention, "").strip()
            chat_history.append(HumanMessage(message_text))
        # when message is from the agent
        else:
            chat_history.append(AIMessage(message_text))
    return chat_history
