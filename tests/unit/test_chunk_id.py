from src.ingestion.chunking.chunk_id import generate_chunk_id, slugify_section


def test_chunk_id_stable_with_page_and_section():
    first = generate_chunk_id(
        doc_hash="abc123",
        text="Install dependencies",
        chunk_index=4,
        page_range=(3, 4),
        section_path=["Chapter 2", "2.1 Installation"],
    )
    second = generate_chunk_id(
        doc_hash="abc123",
        text="Install dependencies",
        chunk_index=4,
        page_range=(3, 4),
        section_path=["Chapter 2", "2.1 Installation"],
    )

    assert first == second
    assert first == "abc123::p003::secchapter-2-2-1-installation::c0004"


def test_chunk_id_falls_back_to_text_hash_without_page():
    chunk_id = generate_chunk_id(
        doc_hash="abc123",
        text="Same text",
        chunk_index=0,
    )

    assert chunk_id.startswith("abc123::c0000::")
    assert len(chunk_id.split("::")[-1]) == 8


def test_slugify_section_safe_fallback_for_non_ascii():
    slug = slugify_section(["安装", "配置"])

    assert slug
    assert slug.isascii()


def test_parent_child_chunk_ids_do_not_collide():
    parent_id = generate_chunk_id(
        doc_hash="abc123",
        text="Parent",
        chunk_index=0,
        chunk_level="parent",
    )
    child_id = generate_chunk_id(
        doc_hash="abc123",
        text="Parent",
        chunk_index=0,
        chunk_level="child",
    )

    assert parent_id != child_id
