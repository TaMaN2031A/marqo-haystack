import base64
import logging
import random
import struct
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Union

import marqo
from haystack import Document, default_from_dict, default_to_dict
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy

from marqo_haystack.errors import MarqoDocumentStoreFilterError

logger = logging.getLogger(__name__)


class MarqoDocumentStore(DocumentStore):
    """
    A MarqoDocumentStore document store for Haystack.
    """

    def __init__(
        self,
        vector_dimension: int,
        collection_name: str = "documents",
        url: str = "http://localhost:8882",
        api_key: Optional[str] = None,
        settings_dict: Optional[Dict[str, Any]] = None,
        client_batch_size: int = 4,
    ):
        """
        Initialise a Marqo document store.

        Args:
            vector_dimension (int): The dimensionality of the vectors stored.
            collection_name (str, optional): The Marqo collection name. Defaults to "documents".
            url (str, optional): Marqo server URL. Defaults to "http://localhost:8882".
            api_key (Optional[str], optional): API key for Marqo Cloud. Defaults to None.
            settings_dict (Optional[Dict[str, Any]], optional): Optional index settings dictionary for local Marqo.
                Defaults to None.
            client_batch_size (int, optional): Batch size for writing documents. Defaults to 4.

        Raises:
            ValueError: If the collection does not exist in Marqo Cloud and no API key is provided.
        """

        self._vector_dimension = vector_dimension
        self._url = url
        self._api_key = api_key
        self._marqo_client = marqo.Client(url=url, api_key=api_key)
        self._client_batch_size = client_batch_size
        self._collection_name = collection_name
        self._settings_dict = settings_dict
        # A document store will receive embedding if any, but it is not to create it
        self._model = "no_model"
        self._model_properties = {"type": self._model, "dimensions": self._vector_dimension}
        seed = 42
        random.seed(seed)
        # S311: ignoring because it is not suitable for cryptographic purposes
        self._dummy_vector = [random.random() * 0.01 for _ in range(self._vector_dimension)]  # noqa: S311

        indexes = {idx["indexName"] for idx in self._marqo_client.get_indexes()["results"]}
        if self._collection_name not in indexes:
            if not api_key:
                self._marqo_client.create_index(
                    self._collection_name,
                    model=self._model,
                    model_properties=self._model_properties,
                    settings_dict=settings_dict,
                )
            else:
                error_msg = (
                    "If using this integration with Marqo Cloud you must create your index ahead of time, "
                    "specify your index name as the collection_name in the MarqoDocumentStore constructor."
                )
                raise ValueError(error_msg)
        else:
            logger.info("Index %s already exists, skipping index creation.", self._collection_name)

        self._index = self._marqo_client.index(self._collection_name)

    def count_documents(self) -> int:
        """
        Return the number of documents currently stored in the Marqo index.

        Returns:
            int: Number of documents in the collection.
        """
        return self._index.get_stats()["numberOfDocuments"]

    def count_vectors(self) -> int:
        """
        Return the number of vectors currently stored in the Marqo index.

        Returns:
            int: Number of vectors in the collection.
        """
        return self._index.get_stats()["numberOfVectors"]

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Retrieve up to 1000 documents matching the given filters.

        Args:
            filters (Optional[Dict[str, Any]]): Dictionary specifying filter conditions.

        Raises:
            MarqoDocumentStoreFilterError: If the filters are invalid or unsupported.

        Returns:
            List[Document]: List of matching Haystack Document objects.
        """

        if not isinstance(filters, dict) and filters is not None:
            msg = "Filters must be a dictionary or None"
            raise MarqoDocumentStoreFilterError(msg)

        filter_string = self._convert_filters(filters)
        results = self._index.search(
            {"customVector": {"content": "", "vector": self._dummy_vector}}, filter_string=filter_string, limit=1000
        )
        hits = []
        for r in results["hits"]:
            #   r.pop("_score")
            hits.append(r)

        return self._get_result_to_documents(hits)

    def _escape_special_filter(self, filter_value: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Escape special characters in a filter string or list of filter strings.

        Args:
            filter_value (Union[str, List[str]]): Filter value(s) to escape.

        Returns:
            Union[str, List[str]]: Escaped filter value(s), preserving the input type.
        """
        special_chars = {"+", "-", "&&", "||", "!", "(", ")", "{", "}", "[", "]", "^", '"', "~", "*", "?", ":", "\\"}
        if isinstance(filter_value, list):
            escaped_list: List[str] = []
            for v in filter_value:
                ev = self._escape_special_filter(v)
                if isinstance(ev, list):
                    escaped_list.extend(ev)
                else:
                    escaped_list.append(ev)
            return escaped_list

        if not isinstance(filter_value, str):
            return filter_value

        if any(c in filter_value for c in special_chars):
            for c in special_chars:
                filter_value = filter_value.replace(c, f"\\{c}")
        return filter_value

    def _convert_filters(self, f: Optional[Dict[str, Any]] = None) -> str | None:
        """
        Convert a Haystack filter dictionary into a Marqo filter string.

        Handles nested boolean operators (AND, OR, NOT) and range filters.

        Args:
            f (Optional[Dict[str, Any]]): Filter dictionary in Haystack format.

        Raises:
            MarqoDocumentStoreFilterError: If the filter is invalid, missing keys, or has unsupported operators.

        Returns:
            Optional[str]: Marqo-compatible filter string, or None if no filter is provided.
        """
        if f is None or f == {}:
            return None

        if "operator" in f and "conditions" in f:
            op = f["operator"].upper()
            if not isinstance(f["conditions"], list):
                msg = f"Conditions must be a list, got {f['conditions']}"
                raise MarqoDocumentStoreFilterError(msg)
            if op == "NOT" and len(f["conditions"]) > 1:
                sub_filters = [f"NOT {self._convert_filters(c)}" for c in f["conditions"]]
                # Used OR because this is the behaviour expected in the tests
                return f"({' OR '.join(sub_filters)})"
            else:
                sub_filters = [s for s in (self._convert_filters(c) for c in f["conditions"]) if s is not None]
                return f"({f' {op} '.join(sub_filters)})"

        required_keys = ("field", "operator", "value")
        if not all(k in f for k in required_keys):
            msg = f"Condition is missing one of the keys: {f}"
            raise MarqoDocumentStoreFilterError(msg)

        field = f["field"]
        operator = f["operator"]
        value = f["value"]
        doc_key = "__meta_" + field.split("meta.")[1] if field.startswith("meta.") else field

        if value is None:
            msg = "Value cannot be None"
            raise MarqoDocumentStoreFilterError(msg)
        # try to parse isoformat dates
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value).timestamp()
            except ValueError:
                pass

        if operator == "==":
            return f"{doc_key}:({value})"
        elif operator == "!=":
            return f"NOT {doc_key}:({value})"
        elif operator in {"in", "not in"}:
            if not isinstance(value, list):
                msg = f"Value {value} must be a list for '{operator}' operator"
                raise MarqoDocumentStoreFilterError(msg)

            if operator == "in":
                return "(" + " OR ".join(f"{doc_key}:({v})" for v in value) + ")"
            else:  # "not in"
                return "(" + " AND ".join(f"NOT {doc_key}:({v})" for v in value) + ")"

        elif operator in {">", ">=", "<", "<="}:
            if not isinstance(value, (int, float)):
                msg = f"Value {value} must be int, float or iso_date for range filters"
                raise MarqoDocumentStoreFilterError(msg)
            if operator == ">":
                return f"{doc_key}:[{value + abs(value) * 1e-16} TO *]"
            elif operator == ">=":
                return f"{doc_key}:[{value} TO *]"
            elif operator == "<":
                return f"{doc_key}:[* TO {value - abs(value) * 1e-16}]"
            elif operator == "<=":
                return f"{doc_key}:[* TO {value}]"
            return None
        else:
            msg = f"Unsupported operator {operator}"
            raise MarqoDocumentStoreFilterError(msg)

    def get_documents_by_id(self, ids: List[str]) -> List[Document]:
        """
        Retrieve documents by their IDs.

        Args:
            ids (List[str]): List of document IDs to fetch.

        Returns:
            List[Document]: List of matching Document objects.
        """
        results = self._index.get_documents(document_ids=ids)["results"]
        results = [r for r in results if r["_found"]]
        return self._get_result_to_documents(results)

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Add documents to the Marqo index.

        Args:
            documents (List[Document]): List of documents to write.
            policy (DuplicatePolicy, optional): Controls behavior on duplicates. Defaults to NONE.

        Raises:
            ValueError: If the input is not a list of Document objects.
            DuplicateDocumentError: If policy is FAIL and duplicates exist.

        Returns:
            int: Number of documents successfully written.
        """
        if (
            not isinstance(documents, Iterable)
            or isinstance(documents, str)
            or any(not isinstance(doc, Document) for doc in documents)
        ):
            err = "Please provide a list of Documents."
            raise ValueError(err)

        if len(documents) == 0:
            return 0

        ok = 200
        ids = [d.id for d in documents]
        existing_docs = self.get_documents_by_id(ids)

        if policy == DuplicatePolicy.FAIL and len(existing_docs) > 0:
            raise DuplicateDocumentError()

        marqo_docs = []
        for doc in documents:
            if doc.content is None:
                logger.warning(f"Document {doc.id} has no content. This document will be skipped")
                continue
            prepared_doc = self._prepare_document(doc)
            marqo_docs.append(prepared_doc)

        if policy == DuplicatePolicy.SKIP:
            existing_ids = [d.id for d in existing_docs]
            marqo_docs = [d for d in marqo_docs if d["_id"] not in existing_ids]
            if len(marqo_docs) == 0:
                return 0

        response = self._index.add_documents(
            documents=marqo_docs,
            client_batch_size=self._client_batch_size,
            mappings={"content_custom_vector": {"type": "custom_vector"}},
            tensor_fields=["content_custom_vector"],
        )
        return sum(1 for item in response[0]["items"] if item["status"] == ok)

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the Marqo index by ID.

        Args:
            document_ids (List[str]): List of document IDs to delete.
        """
        self._index.delete_documents(ids=document_ids)

    def search(
        self, query: Union[str, List[float]], top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform vector or text search for multiple queries.

        Args:
            query (Union[str, List[float]]): Query text or its vector embedding.
            top_k (int): Number of results to return.
            filters (Optional[Dict[str, Any]]): Optional filters to apply during search.

        Returns:
            List[Document]: List of results for the query.
        """
        content, vector = "", self._dummy_vector
        if isinstance(query, str):
            content = query
        else:
            vector = query
        result = self._index.search(
            q={"customVector": {"content": content, "vector": vector}},
            limit=top_k,
            filter_string=self._convert_filters(filters),
        )
        return self._query_result_to_documents([result])[0]

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this document store into a dictionary.

        Returns:
            Dict[str, Any]: Serialized representation of the store.
        """
        data = default_to_dict(
            self,
            vector_dimension=self._vector_dimension,
            collection_name=self._collection_name,
            url=self._url,
            api_key=self._api_key,
            settings_dict=self._settings_dict,
            client_batch_size=self._client_batch_size,
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarqoDocumentStore":
        """
        Deserialize a document store from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary containing store parameters.

        Returns:
            MarqoDocumentStore: A new store instance.
        """
        return default_from_dict(cls, data)

    def _prepare_document(self, d: Document) -> Dict[str, Any]:
        """
        Convert a Haystack Document into a Marqo-storable dictionary.

        Embeddings are converted to base64 strings, and metadata is flattened
        with ISO-format date fields converted to timestamps.

        Args:
            d (Document): Document to prepare.

        Returns:
            Dict[str, Any]: Document ready for storage in Marqo.
        """
        marqo_doc = {}
        haystack_doc = d.to_dict(flatten=False)
        custom_vector = {"vector": self._dummy_vector, "content": None}

        marqo_doc["_id"] = d.id
        iso_format_date_keys = []
        for key, value in haystack_doc.items():
            if key == "meta":
                for key_, value_ in value.items():
                    if value_ is not None:
                        written_value_ = value_
                        if isinstance(value_, str):
                            try:
                                written_value_ = datetime.fromisoformat(value_).timestamp()
                                iso_format_date_keys.append(key_)
                            except ValueError:
                                pass
                        marqo_doc["__meta_" + key_] = written_value_
            elif key == "content":  # cannot be None
                custom_vector["content"] = value
            elif key == "embedding":
                if value is not None:
                    custom_vector["vector"] = value
                    # Not redundant, didn't find a way to make marqo return the embeddings of the custom vector
                    binary = struct.pack(f"{self._vector_dimension}f", *value)
                    encoded = base64.b64encode(binary).decode()
                    marqo_doc["emb_raw"] = encoded
            elif value is not None:
                marqo_doc[key] = value

        marqo_doc["content_custom_vector"] = custom_vector
        marqo_doc["iso_format_date_keys"] = iso_format_date_keys
        return marqo_doc

    def _get_result_to_documents(self, marqo_documents: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert a list of Marqo documents into Haystack Document objects.

        Args:
            marqo_documents (List[Dict[str, Any]]): Raw Marqo results.

        Returns:
            List[Document]: List of Haystack Documents.
        """
        documents = []
        for marqo_doc in marqo_documents:
            # prepare meta
            haystack_doc: Dict[str, Any] = {}
            meta: Dict[str, Any] = {}
            # stored it independently because custom vector didn't return it
            embedding: Dict[str, Any] = {}
            iso_format_date_keys: List = marqo_doc.pop("iso_format_date_keys", [])

            for key, value in marqo_doc.items():
                if key.startswith("__meta_"):
                    new_k = key.replace("__meta_", "")
                    written_value = value
                    if new_k in iso_format_date_keys:
                        written_value = datetime.fromtimestamp(value).isoformat()  # noqa: DTZ006
                    meta[new_k] = written_value
                elif key == "content_custom_vector":
                    haystack_doc["content"] = value
                elif key == "_id":
                    haystack_doc["id"] = value
                elif key == "emb_raw":
                    embedding[key] = value
                elif key == "_highlights":
                    continue
                elif key == "_score":
                    haystack_doc["score"] = value
                else:
                    haystack_doc[key] = value

            haystack_doc["meta"] = meta
            if len(embedding) > 0:
                decoded = base64.b64decode(embedding["emb_raw"])
                vector = list(struct.unpack(f"{self._vector_dimension}f", decoded))
                haystack_doc["embedding"] = vector
            documents.append(Document().from_dict(haystack_doc))

        return documents

    def _query_result_to_documents(self, result: List[Dict[str, Any]]) -> List[List[Document]]:
        """
        Convert search results from multiple queries into Haystack Documents.

        Args:
            result (List[Dict[str, Any]]): Raw Marqo search results per query.

        Returns:
            List[List[Document]]: Documents grouped per query.
        """
        retrievals = []

        for r in result:
            hits: List[Dict[str, Any]] = r["hits"]
            converted_hits = self._get_result_to_documents(hits)
            retrievals.append(converted_hits)
        return retrievals
