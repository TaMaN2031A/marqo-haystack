from collections import Counter

import marqo
import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import DocumentStore
from haystack.testing.document_store import DocumentStoreBaseTests

from marqo_haystack.document_store import MarqoDocumentStore


class TestDocumentStore(DocumentStoreBaseTests):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        """
        Assert that two lists of Documents are equal.

        This is used in every test, if a Document Store implementation has a different behaviour
        it should override this method. This can happen for example when the Document Store sets
        a score to returned Documents. Since we can't know what the score will be, we can't compare
        the Documents reliably.
        """
        for doc in received:
            doc.score = None  # When we store it, it doesn't have this attribute
        received_dicts = [d.to_dict(flatten=True) if hasattr(d, "to_dict") else d for d in received]
        expected_dicts = [d.to_dict(flatten=True) if hasattr(d, "to_dict") else d for d in expected]

        assert Counter(map(frozenset, received_dicts)) == Counter(map(frozenset, expected_dicts))

    @pytest.fixture
    def document_store(self) -> MarqoDocumentStore:
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        mq = marqo.Client()
        test_index = "test-haystack-document-store"
        mq.delete_index(test_index)
        return MarqoDocumentStore(collection_name=test_index, vector_dimension=768)

    def test_write_documents(self, document_store: DocumentStore):
        """
        Test write_documents() default behaviour.
        """
        doc = Document(content="test doc")
        document_store.write_documents([doc])
        self.assert_documents_are_equal(document_store.filter_documents(), [doc])

    @pytest.mark.skip(reason="Filter on None is not supported.")
    @pytest.mark.unit
    def test_comparison_equal_with_none(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Filter on None is not supported.")
    @pytest.mark.unit
    def test_comparison_not_equal_with_none(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Filter on None is not supported.")
    @pytest.mark.unit
    def test_comparison_greater_than_with_none(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Filter on None is not supported.")
    @pytest.mark.unit
    def test_comparison_greater_than_equal_with_none(self, document_store, filterable_docs):
        pass

    def test_comparison_less_than_with_none(self, document_store, filterable_docs):
        pass

    def test_comparison_less_than_equal_with_none(self, document_store, filterable_docs):
        pass

    @pytest.mark.unit
    def test_from_dict(self):
        dc = {"type": "marqo_haystack.document_store.MarqoDocumentStore", "init_parameters": {"vector_dimension": 768}}
        document_store = MarqoDocumentStore.from_dict(dc)
        assert document_store._vector_dimension == dc["init_parameters"]["vector_dimension"]

    @pytest.mark.unit
    def test_to_dict(self):
        dc = {"type": "marqo_haystack.document_store.MarqoDocumentStore", "init_parameters": {"vector_dimension": 768}}
        document_store = MarqoDocumentStore(vector_dimension=768)
        assert (
            document_store.to_dict()["init_parameters"]["vector_dimension"] == dc["init_parameters"]["vector_dimension"]
        )

    @pytest.mark.unit
    def test_from_dict_all_params(self):
        dc = {
            "type": "marqo_haystack.document_store.MarqoDocumentStore",
            "init_parameters": {
                "vector_dimension": 768,
                "collection_name": "my_collection",
                "url": "http://localhost:8882",
                "api_key": None,
                "settings_dict": None,
                "client_batch_size": 16,
            },
        }

        ds = MarqoDocumentStore.from_dict(dc)

        assert ds._vector_dimension == 768
        assert ds._collection_name == "my_collection"
        assert ds._url == "http://localhost:8882"
        assert ds._api_key is None
        assert ds._settings_dict is None
        assert ds._client_batch_size == 16

    @pytest.mark.unit
    def test_to_dict_all_params(self):
        ds = MarqoDocumentStore(
            vector_dimension=768,
            collection_name="my_collection",
            url="http://localhost:8882",
            api_key=None,
            settings_dict=None,
            client_batch_size=16,
        )

        dc = ds.to_dict()

        params = dc["init_parameters"]

        assert params["vector_dimension"] == 768
        assert params["collection_name"] == "my_collection"
        assert params["url"] == "http://localhost:8882"
        assert params["api_key"] is None
        assert params["settings_dict"] is None
        assert params["client_batch_size"] == 16

    @pytest.mark.unit
    def test_to_from_dict_all_params_combined(self):
        original = MarqoDocumentStore(
            vector_dimension=768,
            collection_name="my_collection",
            url="http://localhost:8882",
            api_key=None,
            settings_dict=None,
            client_batch_size=16,
        )

        serialized = original.to_dict()
        restored = MarqoDocumentStore.from_dict(serialized)

        assert restored._vector_dimension == original._vector_dimension
        assert restored._collection_name == original._collection_name
        assert restored._url == original._url
        assert restored._api_key == original._api_key
        assert restored._settings_dict == original._settings_dict
        assert restored._client_batch_size == original._client_batch_size
