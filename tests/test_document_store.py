from typing import List

import pytest
import marqo

from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseTests

from marqo_haystack.document_store import MarqoDocumentStore


class TestDocumentStore(DocumentStoreBaseTests):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def document_store(self) -> MarqoDocumentStore:
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        mq = marqo.Client()
        test_index = "test-haystack-document-store"
        mq.delete_index(test_index)
        return MarqoDocumentStore(collection_name=test_index)

    @pytest.mark.skip(reason="Filter on None is not supported.")
    @pytest.mark.unit
    def test_comparison_equal_with_none(self, document_store, filterable_docs):
        pass

    """
    @pytest.mark.unit
    def test_get_existing(self, document_store: MarqoDocumentStore):
        #Deleting an existing document
        doc = Document(content="test doc")
        document_store.write_documents([doc])

        gotten_docs = document_store.get_documents_by_id(ids=[doc.id])
        assert len(gotten_docs) == 1

        document_store.delete_documents([doc.id])

    @pytest.mark.unit
    def test_search_documents(self, document_store: MarqoDocumentStore):
        #Searching documents
        doc = Document(id="mydoc", content="test1 test2")
        document_store.write_documents([doc])

        documents = document_store.search(queries=["test1", "test2"], top_k=10)
        assert len(documents) == 2
        assert len(documents[0]) <= 10
        assert len(documents[1]) <= 10
        document_store.delete_documents([doc.id])

    @pytest.mark.skip(reason="Filter on embedding value is not supported.")
    @pytest.mark.unit
    def test_eq_filter_embedding(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on embedding value is not supported.")
    @pytest.mark.unit
    def test_in_filter_embedding(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on embedding value is not supported.")
    @pytest.mark.unit
    def test_ne_filter_embedding(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on embedding value is not supported.")
    @pytest.mark.unit
    def test_nin_filter_embedding(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on embedding value is not supported.")
    @pytest.mark.unit
    def test_gt_filter_embedding(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on embedding value is not supported.")
    @pytest.mark.unit
    def test_gte_filter_embedding(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on embedding value is not supported.")
    @pytest.mark.unit
    def test_lt_filter_embedding(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on embedding value is not supported.")
    @pytest.mark.unit
    def test_lte_filter_embedding(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on table value is not supported.")
    @pytest.mark.unit
    def test_eq_filter_table(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on table value is not supported.")
    @pytest.mark.unit
    def test_in_filter_table(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on table value is not supported.")
    @pytest.mark.unit
    def test_ne_filter_table(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on table value is not supported.")
    @pytest.mark.unit
    def test_nin_filter_table(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on table value is not supported.")
    @pytest.mark.unit
    def test_gt_filter_table(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on table value is not supported.")
    @pytest.mark.unit
    def test_gte_filter_table(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on table value is not supported.")
    @pytest.mark.unit
    def test_lt_filter_table(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on table value is not supported.")
    @pytest.mark.unit
    def test_lte_filter_table(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Range query on non-numeric value is not supported.")
    @pytest.mark.unit
    def test_gt_filter_non_numeric(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Range query on non-numeric value is not supported.")
    @pytest.mark.unit
    def test_gte_filter_non_numeric(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Range query on non-numeric value is not supported.")
    @pytest.mark.unit
    def test_lt_filter_non_numeric(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Range query on non-numeric value is not supported.")
    @pytest.mark.unit
    def test_lte_filter_non_numeric(self, document_store: MarqoDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Duplicate policy not supported.")
    @pytest.mark.unit
    def test_write_documents_duplicate_fail(self, document_store: MarqoDocumentStore):
        pass

    @pytest.mark.skip(reason="Duplicate policy not supported.")
    @pytest.mark.unit
    def test_write_documents_duplicate_skip(self, document_store: MarqoDocumentStore):
        pass

    @pytest.mark.skip(reason="Duplicate policy not supported.")
    @pytest.mark.unit
    def test_write_documents_duplicate_overwrite(self, document_store: MarqoDocumentStore):
        pass

    @pytest.mark.skip(reason="Filter on array contents is not supported.")
    @pytest.mark.unit
    def test_filter_document_array(self, document_store: DocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Filter on dataframe is not supported.")
    @pytest.mark.unit
    def test_filter_document_dataframe(self, document_store: DocumentStore, filterable_docs: List[Document]):
        pass
    
    """