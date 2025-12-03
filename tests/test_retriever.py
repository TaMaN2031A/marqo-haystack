import random

import marqo
import pytest
from haystack.dataclasses import Document

from marqo_haystack import MarqoDocumentStore, MarqoRetriever


@pytest.fixture
def document_store() -> MarqoDocumentStore:
    mq = marqo.Client()
    test_index = "test-haystack-document-store"
    mq.delete_index(test_index)
    return MarqoDocumentStore(collection_name=test_index, vector_dimension=768)


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(content="Hello world", embedding=[random.random() for _ in range(768)]),  # noqa: S311
        Document(content="Test document 1", embedding=[random.random() for _ in range(768)]),  # noqa: S311
        Document(content="Another test document", embedding=[random.random() for _ in range(768)]),  # noqa: S311
    ]


@pytest.mark.integration
def test_marqo_retriever_run_text(document_store, sample_documents):
    document_store.write_documents(sample_documents)
    retriever = MarqoRetriever(document_store=document_store, top_k=2)
    result = retriever.run("Hello world")
    assert "documents" in result
    assert len(result["documents"]) <= 2
    assert isinstance(result["documents"][0], Document)


@pytest.mark.integration
def test_marqo_retriever_run_vector(document_store, sample_documents):
    document_store.write_documents(sample_documents)
    retriever = MarqoRetriever(document_store=document_store, top_k=2)
    query_vector = [random.random() for _ in range(768)]  # noqa: S311
    result = retriever.run(query_vector)
    assert "documents" in result
    assert len(result["documents"]) <= 2
    assert isinstance(result["documents"][0], Document)
