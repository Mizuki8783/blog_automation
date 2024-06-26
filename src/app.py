import os
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request
from functions import create_agent, get_response, get_chat_history

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize the Flask app
# Flask is a web application framework written in Python
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

#Initialize agent
agent = create_agent()

@app.event("app_mention")
def handle_mentions(body, say):
    """
    Event listener for mentions in Slack.
    When the bot is mentioned, this function processes the text and sends a response.

    Args:
        body (dict): The event data received from Slack.
        say (callable): A function for sending a response to the channel.
    """
    channel_id = body["event"]["channel"]
    text = body["event"]["text"]
    message_ts = body["event"]["ts"]
    thread_ts = body["event"].get("thread_ts")

    mention = f"<@{SLACK_BOT_USER_ID}>"
    query = text.replace(mention, "").strip()

    if thread_ts == None:
        chat_history = []
        response = get_response(agent,query,chat_history)
        say(response, thread_ts=message_ts)
    else:
        chat_history = get_chat_history(app,channel_id,thread_ts)
        response = get_response(agent,query,chat_history)
        say(response, thread_ts=thread_ts)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """
    Route for handling Slack events.
    This function passes the incoming HTTP request to the SlackRequestHandler for processing.

    Returns:
        Response: The result of handling the request.
    """

    return handler.handle(request)


# Run the Flask app
if __name__ == "__main__":
    flask_app.run()
