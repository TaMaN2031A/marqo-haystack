from typing import Any, Dict, List, Optional, Union

from haystack import component
from haystack.dataclasses import Document

from marqo_haystack import MarqoDocumentStore


@component
class MarqoRetriever:
    """
    A component for retrieving documents from an MarqoDocumentStore with query.
    """

    def __init__(self, document_store: MarqoDocumentStore, filters: Optional[Dict[str, Any]] = None, top_k: int = 10):
        """
        Create a retriever component. Usually you pass some basic configuration
        parameters to the constructor.

        Args:
            document_store (MarqoDocumentStore): An instance of a MarqoDocumentStore
            filters (Optional[Dict[str, Any]], optional): A dictionary with filters to narrow down the search space.
                Defaults to None.
            top_k (int, optional): Maximum number of results to return.
        """
        self.filters = filters
        self.top_k = top_k
        self.document_store = document_store

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: Union[str | List[float]],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ):
        """
        Run the retriever on a query.

        Args:
            query (List[str]): The query or the query embedding
            filters (Optional[Dict[str, Any]], optional): A dictionary with filters to narrow down the search space.
            Defaults to None.
            top_k (Optional[int], optional): The maximum number of documents to retrieve. Defaults to None.
        """

        if not top_k:
            top_k = self.top_k

        if not filters:
            filters = self.filters

        return {"documents": self.document_store.search(query, top_k, filters=filters)}
