# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'create_chunk_results' function."""

from collections.abc import Callable

from graphrag_chunking.text_chunk import TextChunk


def create_chunk_results(
    chunks: list[str],
    transform: Callable[[str], str] | None = None,
    encode: Callable[[str], list[int]] | None = None,
    starts: list[int] | None = None,
) -> list[TextChunk]:
    """Create chunk results from a list of text chunks.

    When ``starts`` is omitted, index assignments are 0-based and assume chunks
    were contiguous and non-overlapping in the source text. Pass ``starts`` for
    overlapping windows (e.g. TokenChunker) so ``start_char``/``end_char`` match
    the real span of each chunk.
    """
    results = []
    start_char = 0
    for index, chunk in enumerate(chunks):
        if starts is not None:
            start_char = starts[index]
        end_char = start_char + len(chunk) - 1  # 0-based indices
        result = TextChunk(
            original=chunk,
            text=transform(chunk) if transform else chunk,
            index=index,
            start_char=start_char,
            end_char=end_char,
        )
        if encode:
            result.token_count = len(encode(result.text))
        results.append(result)
        if starts is None:
            start_char = end_char + 1
    return results
