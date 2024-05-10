from typing import List
from src.llm import RAG_LLM

if __name__ == "__main__":

    import gradio as gr
    import asyncio

    llm_interface = RAG_LLM("taxonomy_faqs_cleaned.md")

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
