from rag.corpus import load_chunks
from rag.retriever import reciprocal_rank_fusion, tokenize


def test_corpus_has_page_and_stable_ids():
    first = load_chunks()
    second = load_chunks()
    assert len(first) >= 10
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(c.source and c.page > 0 and c.doc_id for c in first)


def test_hybrid_rrf_uses_both_retrievers():
    ranking = reciprocal_rank_fusion([['a', 'b'], ['b', 'c']], [0.65, 0.35])
    assert ranking[0] == 'b'
    assert set(ranking) == {'a', 'b', 'c'}


def test_tokenize_accented_text():
    assert 'validación' in tokenize('Validación con Pydantic')
