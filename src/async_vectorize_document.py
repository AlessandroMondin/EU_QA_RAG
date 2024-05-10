import hashlib
import re
import os

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import models, AsyncQdrantClient
from .logger import logger

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)


class QdrantLayer:
    """
    This class handles the communication with Qdrant vector database.
    It processes a document, by checking that its structure follows a given schema,
    (QA & "###" to indicate questions and the following text indicating the corresponding
    answer). Afterwards, are computed the embeddings of each 'question' which are loaded
    together with the corresponding answer. The class also handles the retrieval of the
    top-k closest "questions&answers" w.r.t. a new given string (sentence), see the
    `retrieve` method.
    """

    client = OpenAI()
    EMBEDDING_MODEL = "text-embedding-3-small"
    MAX_INPUT_TOKEN = 8191

    _qdrant = None

    @classmethod
    async def get_qdrant(cls):
        """
        Returns an instance of the Qdrant client, initializing it if it hasn't been
        already.

        Returns:
            AsyncQdrantClient: An instance of the asynchronous Qdrant client.
        """
        if cls._qdrant is None:
            cls._qdrant = AsyncQdrantClient("localhost", port=6333)
        return cls._qdrant

    def __init__(self, file_path):
        """
        Initializes the QdrantLayer instance, validating the provided file path and
        setting up documents.

        Args:
            file_path (str): The path to the Markdown file containing questions and
                             answers.
        """
        # Initialize Qdrant instance attributes
        self._qdrant_collection = None

        # Validate file and set up documents
        documents, document_encryption = self._validate_and_load_md_file(file_path)
        self.documents = documents
        self.document_encryption = document_encryption

    async def setup_qdrant(self):
        """
        Sets up the Qdrant collection if it doesn't already exist.
        Creates a collection and uploads document embeddings if necessary.
        """
        qdrant_client = await self.get_qdrant()

        # Check if the collection already exists
        response = await qdrant_client.get_collections()
        collections = [c.name for c in response.collections]
        if self.document_encryption not in collections:
            # Create collection if not found
            await qdrant_client.create_collection(
                collection_name=self.document_encryption,
                vectors_config=models.VectorParams(
                    size=len(self.get_embedding("test")),  # Set vector size dynamically
                    distance=models.Distance.COSINE,
                ),
            )

            logger.info("Uploading embeddings to QDRANT")
            await qdrant_client.upload_points(
                collection_name=self.document_encryption,
                points=[
                    models.PointStruct(
                        id=idx,
                        vector=self.get_embedding(doc["question"]),
                        payload=doc,
                    )
                    for idx, doc in enumerate(self.documents[:2])
                ],
            )
        else:
            logger.info("Collection already exists. Skipping setup.")

    def get_embedding(self, text):
        """
        Generates an embedding for the provided text using the configure OpenAI's
        sentence embedder specified as class attribute.

        Args:
            text (str): The input text to be embedded.

        Returns:
            list: A list of floats representing the text's embedding vector.
        """
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.EMBEDDING_MODEL)
            .data[0]
            .embedding
        )

    async def retrieve(self, message: str, retrieve_n_queries=5):
        """
        Searches for documents similar to the provided message using the Qdrant
        collection.

        Args:
            message (str): The search query text.
            retrieve_n_queries (int): The number of closest matching documents to
                                      retrieve.

        Returns:
            list: A list of document payloads matching the search query.
        """
        qdrant_client = await self.get_qdrant()
        hits = await qdrant_client.search(
            collection_name=self.document_encryption,
            query_vector=self.get_embedding(message),
            limit=retrieve_n_queries,
        )
        hits = [hit.payload for hit in hits]
        return hits

    def _validate_and_load_md_file(self, path: str):
        """
        Parses and validates the Markdown file to extract questions and answers.
        Returns a list of dictionaries with 'question' and 'answer' keys.
        Raises a ValueError if any section is not formatted correctly.

        Additionally, returns an SHA-256 hash of the concatenated first and last
        questions.

        Args:
            path (str): The path to the Markdown file.

        Returns:
            tuple: A list of dictionaries containing questions and answers, and the
            encryption hash.
        """
        documents = []
        with open(path, "r") as file:
            content = file.read()

        # Remove any unwanted portions between '(![]' and '=)'
        # like link/images
        pattern = r"\(!\[\].*?=\)"
        content = re.sub(pattern, "", content)

        # Split the file into sections using '###' as the delimiter
        sections = re.split(r"###\s*", content)
        for section in sections[1:]:  # Skip the initial empty string
            # Extract the question and answer from each section
            lines = section.strip().splitlines()
            if not lines or len(lines) < 2:
                logger.warning(f"Invalid question found: see {lines} in the document")
                continue

            question = lines[0].strip()
            answer = "\n".join(lines[1:]).strip()
            documents.append({"question": question, "answer": answer})

        if not documents:
            raise ValueError(
                "Markdown file must contain at least one valid question and answer section."
            )

        # Retrieve the first and last question
        first_question = documents[0]["question"]
        last_question = documents[-1]["question"]

        # Concatenate and compute the SHA-256 hash
        concatenated_questions = first_question + last_question
        hash_object = hashlib.sha256(concatenated_questions.encode())
        document_encryption = hash_object.hexdigest()

        return documents, document_encryption
