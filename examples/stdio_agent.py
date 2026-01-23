import argparse
import traceback
from typing import (
    Any,
    Dict,
    List,
)

from chat_factory import (
    ChatFactory,
    ChatModel,
)
from chat_factory.utils.factory import configure_logging
from dotenv import (
    find_dotenv,
    load_dotenv,
)
from mcp_multi_server.utils import print_capabilities_summary
from utils.agent import SYSTEM_MESSAGE
from utils.stdio_mcp import (
    display_mcp_content,
    display_mcp_resource_result,
    get_prompt_arguments,
    get_template_variables,
)
from utils.tools import tools


load_dotenv(find_dotenv())


def chat(verbose: bool = False) -> None:
    configure_logging(level="INFO" if verbose else "WARNING")

    openai_model = ChatModel(model_name="gpt-5.2", provider="openai")
    # anthropic_model = ChatModel(model_name="claude-sonnet-4-5", provider="anthropic")
    # google_model = ChatModel(model_name="gemini-2.5-flash", provider="google")
    # deepseek_model = ChatModel(model_name="deepseek-chat", provider="deepseek")
    # groq_model = ChatModel(model_name="openai/gpt-oss-120b", provider="groq")
    # ollama_model = ChatModel(model_name="deepseek-r1:7b", provider="ollama", api_key="unused")

    try:
        with ChatFactory(
            generator_model=openai_model,
            system_prompt=SYSTEM_MESSAGE,
            tools=tools,
            generator_kwargs={"reasoning_effort": "none"},
            mcp_config_path="mcp_config.json",
            display_content=display_mcp_content,
        ) as factory:

            factory.set_mcp_logging_level(level="CRITICAL")
            print_capabilities_summary(factory.mcp_client)  # type: ignore

            messages: List[Dict[str, Any]] = []
            print("Multi-Server MCP Chat Client")
            print("Type 'exit' or 'quit' to end the conversation\n")

            query = input("> ")

            while query.lower() not in ("exit", "quit"):

                # Add user message, prompt or resource
                if query.startswith("+prompt:"):
                    prompt_name = query[len("+prompt:") :].strip()
                    prompt_messages = factory.instantiate_prompt(
                        prompt_name=prompt_name,
                        get_prompt_arguments=get_prompt_arguments,
                        display_content=display_mcp_content,
                    )
                    if not prompt_messages:
                        print(f"Prompt '{prompt_name}' not found.")
                    else:
                        messages.extend(prompt_messages)
                    query = input("> ")
                    continue

                if query.startswith("+resource:"):
                    resource_name = query[len("+resource:") :].strip()
                    resource_messages = factory.instantiate_resource(
                        resource_name=resource_name,
                        get_template_variables=get_template_variables,
                        display_result=display_mcp_resource_result,
                    )
                    if not resource_messages:
                        print(f"Resource '{resource_name}' not found.")
                    else:
                        if verbose:
                            print("****Retrieved resource content (displayed above)\n")
                        messages.extend(resource_messages)
                    query = input("> ")
                    continue

                reply = factory.chat(message=query, history=messages)
                messages.append({"role": "assistant", "content": reply})

                # Print assistant response
                print(f"\n\033[34m{reply}\033[0m\n")

                # Get next user input
                query = input("> ")

    except FileNotFoundError as e:
        print(f"Configuration error: {e}")
    except Exception:
        print("An error occurred:")
        traceback.print_exc()


def main() -> None:
    """Parse command-line arguments and run the chat client."""
    parser = argparse.ArgumentParser(
        description="Multi-server MCP chat client with OpenAI integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (tool calls and results)",
    )

    args = parser.parse_args()
    chat(verbose=args.verbose)


if __name__ == "__main__":
    main()
