from typing import Any, Dict, List, Optional, Union

from haystack import DeserializationError, component, default_from_dict, default_to_dict
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
        self._filters = filters
        self._top_k = top_k
        self._document_store = document_store

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
            top_k = self._top_k

        if not filters:
            filters = self._filters

        return {"documents": self._document_store.search(query, top_k, filters=filters)}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        docstore = self._document_store.to_dict()
        return default_to_dict(
            self,
            document_store=docstore,
            filters=self._filters,
            top_k=self._top_k,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LanceDBFTSRetriever":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        if "document_store" not in init_params:
            err = "Missing 'document_store' in serialization data"
            raise DeserializationError(err)
        if "type" not in init_params["document_store"]:
            err = "Missing 'type' in document store's serialization data"
            raise DeserializationError(err)
        data["init_parameters"]["document_store"] = MarqoDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)
