import os
from typing import Dict, List

from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

from .async_vectorize_document import QdrantLayer
from .prompts import rag_template

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

llm = ChatOpenAI(temperature=0.7, model="gpt-4")


class RAG_LLM:
    """
    LLM class using which uses RAG with Qdrant.

    Attributes:
        qdrant_layer (QdrantLayer): A reference to the document retriever class.
        llm (ChatOpenAI): An instance of ChatOpenAI configured with temperature and model.
        retriever (QdrantLayer): An instance for accessing and retrieving the most
                                 relevant documents.
        retrieved_rag_samples (int): Number of top documents to retrieve for prompt
                                     augmentation.
    """

    qdrant_layer = QdrantLayer
    llm = ChatOpenAI(temperature=0.7, model="gpt-4")

    def __init__(self, path, retrieved_rag_samples=3):
        """
        Initializes the LLMInterface instance, ensuring the provided path is valid.

        Args:
            path (str): Path to the document storage for retrieval.
            retrieved_rag_samples (int): Number of top documents to retrieve for prompt
                                         augmentation.

        Raises:
            FileNotFoundError: If the provided path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find any path at {path}")

        self.retriever = self.qdrant_layer(path)
        self.retrieved_rag_samples = retrieved_rag_samples

    async def ainvoke(
        self,
        message: str,
        history: List[List[str]],
    ) -> str:
        """
        Asynchronously processes a user message by retrieving relevant documents
        and using them to augment the LLM's response.

        Args:
            message (str): User's input message.
            history (List[List[str]]): List of past conversation history.

        Returns:
            str: The LLM-generated response.
        """
        # Retrieve the queries closest to the user's prompt (RAG).
        closest_queries = await self.retriever.retrieve(
            message, self.retrieved_rag_samples
        )

        closest_queries = self.reformat_closest_queries(closest_queries)

        prompt = self.prepare_prompt(
            prompt_template=rag_template,
            documents=closest_queries,
            message=message,
        )
        prompt = self.langchain_format(prompt, history)

        response = await self.llm.ainvoke(prompt)
        return response.content

    def langchain_format(self, message: str, history=[]) -> List[List]:
        """
        Converts conversation history to LangChain message format.

        Args:
            message (str): The current input message.
            history (List[List[str]]): List of previous user and AI messages.

        Returns:
            List[List]: Messages formatted according to LangChain schema.
        """
        history_langchain_format = []
        for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))
        history_langchain_format.append(HumanMessage(content=message))
        return history_langchain_format

    def prepare_prompt(self, prompt_template: str, **placeholders):
        """
        Prepares a prompt by applying placeholders to a given template.

        Args:
            prompt_template (str): The base template string.
            **placeholders: Named placeholders to be inserted into the template.

        Returns:
            str: The formatted prompt string.
        """
        prompt = PromptTemplate.from_template(template=prompt_template)
        return prompt.format(**placeholders)

    def reformat_closest_queries(self, queries: List[Dict]) -> str:
        """
        Reformats retrieved Qdrant queries to a format more comprehensible by the LLM.

        Args:
            queries (List[Dict]): List of dictionaries containing "question" and "answer".

        Returns:
            str: Reformatted string of queries and answers.
        """
        result_str = ""
        for item in queries:
            question = item["question"]
            answer = item["answer"]
            result_str += f"{question}\n{answer}\n\n"
        # Removes extra whitespace.
        return result_str.strip()
