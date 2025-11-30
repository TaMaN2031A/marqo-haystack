import base64
import logging
import struct
from typing import Any, Dict, List, Optional, Union, Iterable
import math
import random

import marqo
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.dataclasses import Document
from haystack.dataclasses.byte_stream import ByteStream

from marqo_haystack.errors import MarqoDocumentStoreFilterError

logger = logging.getLogger(__name__)


class MarqoDocumentStore(DocumentStore):
    """
    A MarqoDocumentStore document store for Haystack.
    """

    def __init__(
        self,
        vector_dimension: int = 512,
        collection_name: str = "documents",
        url: str = "http://localhost:8882",
        api_key: Optional[str] = None,
        settings_dict: Optional[Dict[str, Any]] = None,
        client_batch_size: int = 4,
    ):
        """Initialise the document store

        Args:
            collection_name (str, optional): The name of your collection, known as an 'index' in Marqo. Defaults to "documents".
            url (_type_, optional): The URL for Marqo, if using the cloud then use https://api.marqo.ai. Defaults to "http://localhost:8882".
            api_key (Optional[str], optional): Your Marqo Cloud API key (only required for cloud). Defaults to None.
            settings_dict (Optional[Dict[str, Any]], optional): A settings dictionary for creation of the index if running Marqo locally. Defaults to None.
            client_batch_size (int, optional): The client batch size for adding documents, set this higher (16-32) if using a GPU. Defaults to 4.

        Raises:
            ValueError: If collection_name is not an existing index and you are using Marqo cloud then an error will be raised.
        """
        self._vector_dimension = vector_dimension
        self._marqo_client = marqo.Client(url=url, api_key=api_key)
        self.client_batch_size = client_batch_size
        self._collection = collection_name
        self._settings_dict = settings_dict
        # A document store will receive embedding if any, but it is not to create it
        self._model = "no_model"
        self._model_properties = {"type": self._model, "dimensions": self._vector_dimension}
        seed = 42
        random.seed(seed)
        self._dummy_vector = [random.random() * 0.01 for _ in range(self._vector_dimension)]

        indexes = {idx["indexName"] for idx in self._marqo_client.get_indexes()["results"]}
        if self._collection not in indexes:
            if not api_key:
               # self._marqo_client.create_index(self._collection, settings_dict=settings_dict)
                self._marqo_client.create_index(self._collection, model=self._model,
                                      model_properties=self._model_properties, settings_dict=settings_dict)
            else:
                raise ValueError(
                    "If using this integration with Marqo Cloud you must create your index ahead of time, specify your index name as the collection_name in the MarqoDocumentStore constructor."
                )
        else:
            print(f"Index {self._collection} already exists, skipping index creation.")

        self._index = self._marqo_client.index(self._collection)


    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        return self._index.get_stats()["numberOfDocuments"]

    def count_vectors(self) -> int:
        """
        Returns how many vectors are present in the document store.
        """
        return self._index.get_stats()["numberOfVectors"]

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Returns at most 1,000 documents that match the filter

        Args:
            filters (Optional[Dict[str, Any]], optional): Filters to apply. Defaults to None.

        Raises:
            MarqoDocumentStoreFilterError: If the filter is invalid or not supported by this class.

        Returns:
            List[Document]: A list of matching documents.
        """

        if not isinstance(filters, dict) and filters is not None:
            msg = "Filters must be a dictionary or None"
            raise MarqoDocumentStoreFilterError(msg)

        filter_string = self._convert_filters(filters)
        print("Filter string: ", filter_string)
        results = self._index.search(
            {"customVector": {"content": "", "vector": self._dummy_vector}},
            filter_string=filter_string,
            limit=1000
        )
        hits = []
        for r in results["hits"]:
            r.pop("_score")
            hits.append(r)

        return self._get_result_to_documents(hits)

    def _escape_special_filter(self, filter_value: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Escape special characters in filter values
        """
        special_chars = {"+", "-", "&&", "||", "!", "(", ")", "{", "}", "[", "]", "^", '"', "~", "*", "?", ":", "\\"}
        if isinstance(filter_value, list):
            return [self._escape_special_filter(v) for v in filter_value]

        if not isinstance(filter_value, str):
            return filter_value

        if any(c in filter_value for c in special_chars):
            for c in special_chars:
                filter_value = filter_value.replace(c, f"\\{c}")
        return filter_value

    def _convert_filters(self, f: dict) -> str | None:
        """
        Convert haystack filters to marqo filter string capturing all boolean operators
        """
        if f is None or f == {}:
            return None

        if "operator" in f and "conditions" in f:
            op = f["operator"].upper()
            sub_filters = [self._convert_filters(c) for c in f["conditions"]]
            return f"({f' {op} '.join(sub_filters)})"

        field = f["field"]
        operator = f["operator"]
        value = f["value"]
        doc_key = "__meta_" + field.split("meta.")[1] if field.startswith("meta.") else field

        if value is None:
            raise MarqoDocumentStoreFilterError(f"Value cannot be None for {operator} operator")

        if operator == "==":
            return f"{doc_key}:({value})"
        elif operator == "!=":
            return f"NOT {doc_key}:({value})"
        elif operator == "in":
            return "(" + " OR ".join(f"{doc_key}:({v})" for v in value) + ")"
        elif operator == "not in":
            return "(" + " AND ".join(f"NOT {doc_key}:({v})" for v in value) + ")"
        elif operator in {">", ">=", "<", "<="}:
            if not isinstance(value, (int, float)):
                raise MarqoDocumentStoreFilterError(f"Value {value} must be int or float for range filters")
            if operator == ">":
                return f"{doc_key}:[{value + value * 1e-16} TO *]"
            elif operator == ">=":
                return f"{doc_key}:[{value} TO *]"
            elif operator == "<":
                return f"{doc_key}:[* TO {value - value * 1e-16}]"
            elif operator == "<=":
                return f"{doc_key}:[* TO {value}]"
            return None
        else:
            raise MarqoDocumentStoreFilterError(f"Unsupported operator {operator}")

    def get_documents_by_id(self, ids: List[str]) -> List[Document]:
        """
        Returns documents with given ids.
        """
        results = self._index.get_documents(document_ids=ids)["results"]
        results = [r for r in results if r["_found"]]
        return self._get_result_to_documents(results)

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int:
        """Writes documents into the Marqo index.

        Args:
            documents (List[Document]): A list of documents to add
            policy (DuplicatePolicy, optional): Not used, ignore.

        Raises:
            ValueError: If the documents are not a list of the Document object.
        """
        if (
            not isinstance(documents, Iterable)
            or isinstance(documents, str)
            or any(not isinstance(doc, Document) for doc in documents)
        ):
            err = "Please provide a list of Documents."
            raise ValueError(err)

        marqo_docs = []
        for d in documents:
            if d.content is None:
                logger.warning(
                    f"Document {d.id} has no content. "
                    "This document will be skipped"
                )
                continue
            d = self._prepare_document(d)
            marqo_docs.append(d)
        self._index.add_documents(
            documents=marqo_docs,
            client_batch_size=self.client_batch_size,
            mappings={"content_custom_vector": {"type": "custom_vector"}},
            tensor_fields=["content_custom_vector"]
        )

    def delete_documents(self, document_ids: List[str]) -> None:
        """Deletes documents from the index. If the document doesn't exist then it is ignored.

        Args:
            document_ids (List[str]): A list of document IDs to delete.
        """
        self._index.delete_documents(ids=document_ids)

    def search(
        self, queries: List[Union[str, List[float]]], top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[List[Document]]:
        """Perform a search for a list of queries.

        Args:
            queries (List[Union[str, Dict[str, float]]]): A list of queries.
            top_k (int): The number of results to return.
            filters (Optional[Dict[str, Any]], optional): Filters to apply during search. Defaults to None.

        Returns:
            List[List[Document]]: A list of matching documents for each query.
        """
        results = []
        for query_or_query_embedding in queries:
            if isinstance(query_or_query_embedding, str):
                result = self._index.search(
                    q={"content": query_or_query_embedding, "vector": self._dummy_vector},
                    limit=top_k,
                    filter_string=self._convert_filters(filters)
                )
            else:
                result = self._index.search(
                    q={"content": "", "vector": query_or_query_embedding},
                    limit=top_k,
                    filter_string=self._convert_filters(filters)
                )

            results.append(result)

        return self._query_result_to_documents(results)

    def _prepare_document(self, d: Document) -> Dict[str, Any]:
        """
        Change the document in a way we can better store it into Marqo.
        """
        marqo_doc = {}
        haystack_doc = d.to_dict(flatten=False)
        custom_vector = {"vector": self._dummy_vector, "content": None}

        marqo_doc["_id"] = d.id
        for key, value in haystack_doc.items():
            if key == "meta":
                for key_, value_ in value.items():
                    if value_ is not None:
                        marqo_doc["__meta_" + key_] = value_
            elif key == "content": # cannot be None
                custom_vector["content"] = value
            elif key == "embedding":
                if value is not None:
                    custom_vector["vector"] = value
                    # Not redundant, didn't find a way to make marqo return the embeddings of the custom vector
                    binary = struct.pack(f'{self._vector_dimension}f', *value)
                    encoded = base64.b64encode(binary).decode()
                    marqo_doc["emb_raw"] = encoded
            elif value is not None:
                marqo_doc[key] = value

        marqo_doc["content_custom_vector"] = custom_vector

        return marqo_doc

    def _get_result_to_documents(self, marqo_documents: List[Dict[str, Any]]) -> List[Document]:
        """
        Helper function to convert Marqo results into Haystack Documents
        """
        documents = []
        for marqo_doc in marqo_documents:
            # prepare meta
            haystack_doc: Dict[str, Any] = {}
            meta: Dict[str, Any] = {}
            # stored it independently because custom vector didn't return it
            embedding: Dict[str, Any] = {}

            for k in marqo_doc:
                if k.startswith("__meta_"):
                    new_k = k.replace("__meta_", "")
                    meta[new_k] = marqo_doc[k]
                elif k == "content_custom_vector":
                    haystack_doc["content"] = marqo_doc[k]
                elif k == "_id":
                    haystack_doc["id"] = marqo_doc[k]
                elif k == "emb_raw":
                    embedding[k] = marqo_doc[k]
                elif k == "_score" or k == "_highlights":
                    continue
                else:
                    haystack_doc[k] = marqo_doc[k]

            haystack_doc["meta"] = meta
            if len(embedding) > 0:
                decoded = base64.b64decode(embedding["emb_raw"])
                vector = list(struct.unpack(f'{self._vector_dimension}f', decoded))
                haystack_doc["embedding"] = vector
            documents.append(Document().from_dict(haystack_doc))

        return documents

    def _query_result_to_documents(self, result: Dict[str, Any]) -> List[List[Document]]:
        """
        Helper function to convert Marqo results into Haystack Documents
        """
        retrievals = []

        for r in result:
            converted_hits = self._get_result_to_documents(r["hits"])
            retrievals.append(converted_hits)
        return retrievals
