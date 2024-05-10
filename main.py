import gradio as gr
import asyncio
import argparse
from typing import List
from src.llm import RAG_LLM


def main(md_file_path: str):
    """Main function to run the Gradio interface with the specified markdown file path."""
    llm_interface = RAG_LLM(md_file_path)

    async def process_message_async(message: str, history: List[List[str]]) -> str:
        """Asynchronous function to process the prompt and obtain a response."""
        try:
            response = await asyncio.wait_for(
                llm_interface.ainvoke(message, history), timeout=15
            )  # 15-second timeout
            history.append([message, response])
            return response
        except asyncio.TimeoutError:
            return "Request timed out."

    # Define a synchronous wrapper that calls the async function
    def process_message_sync(message: str, history: List[List[str]]):
        """Wrapper to call the asynchronous function synchronously for Gradio."""
        return asyncio.run(process_message_async(message, history))

    # Define the Gradio chat interface with a concurrency limit of 10
    iface = gr.ChatInterface(
        fn=process_message_sync,
        concurrency_limit=10,
    )

    # Launch the Gradio interface
    iface.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Specify the path to the markdown file containing Q&A"
    )
    parser.add_argument("--path", type=str, help="Path to the md file containing Q&A")
    args = parser.parse_args()

    main(args.path)
