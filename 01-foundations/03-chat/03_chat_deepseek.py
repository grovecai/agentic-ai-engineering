"""
Interactive Chat (DeepSeek)

Demonstrates an interactive chat loop with client-managed message history.
DeepSeek's Responses API is stateless, so each request sends the full history.
"""

import os

from common import OpenAITokenTracker, setup_logging
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI, OpenAIError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Load environment variables from root .env file
load_dotenv(find_dotenv())

# Configure logging
logger = setup_logging(__name__)


class ChatSession:
    """
    Chat agent that maintains conversation history and encapsulates
    all chat logic including message management and API interaction.
    """

    def __init__(self, model: str, token_callback: OpenAITokenTracker):
        """
        Initialize the chat session.
        """
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.token_callback = token_callback
        self.messages: list[dict[str, str]] = []
        self.model = model

    def send_message(self, user_message: str) -> str:
        """
        Send a message and get a response using the Responses API.
        """
        # Keep the pending turn out of saved history if the API call fails.
        request_messages = [*self.messages, {"role": "user", "content": user_message}]

        logger.info("Sending message to DeepSeek (history length: %d)", len(request_messages))

        # Make API call with full message history using Responses API
        response = self.client.responses.create(
            model=self.model,
            temperature=0.1,
            max_output_tokens=2048,
            input=request_messages,
            reasoning={"effort": "none"},
        )

        # Track token usage
        if response.usage:
            self.token_callback.track(response.usage)

        # Save both sides of a completed turn for the next stateless request.
        assistant_message = response.output_text or ""
        if not assistant_message:
            raise ValueError("DeepSeek returned no text response")
        self.messages.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )

        return assistant_message


def main() -> None:
    """
    Main orchestration function that handles user interaction and coordinates the chat flow.
    """

    # Rich console for beautiful output
    console = Console()
    # Create token tracker and chat session
    token_tracker = OpenAITokenTracker()
    chat = ChatSession(os.getenv("DEEPSEEK_MODEL", "deepseek-flash"), token_tracker)

    # Welcome message
    console.print(
        Panel(
            "[bold cyan]Welcome to DeepSeek Chat![/bold cyan]\n\n"
            "Type your messages and press Enter.\n"
            "Type 'quit' or 'exit' to end the conversation.",
            title="Chat Session",
        )
    )

    # Chat loop
    while True:
        # Get user input
        console.print("\n[bold green]You:[/bold green] ", end="")
        user_input = input().strip()

        if user_input.lower() in ["quit", "exit", ""]:
            console.print("\n[yellow]Ending chat session...[/yellow]")
            break

        # Send message and get response
        try:
            response = chat.send_message(user_input)

            # Display response with markdown formatting
            console.print("\n[bold blue]DeepSeek:[/bold blue]")
            console.print(Markdown(response))

        except (OpenAIError, ValueError) as e:
            logger.error("Error during chat: %s", e)
            console.print(f"\n[red]Error: {e}[/red]")
            break

    # Report final token usage
    console.print()
    token_tracker.report()
    console.print(f"\n[dim]Total messages exchanged: {len(chat.messages)}[/dim]")


if __name__ == "__main__":
    main()
