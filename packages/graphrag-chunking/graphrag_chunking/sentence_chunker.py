# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'SentenceChunker' class."""

from collections.abc import Callable
from typing import Any

import nltk

from graphrag_chunking.bootstrap_nltk import bootstrap
from graphrag_chunking.chunker import Chunker
from graphrag_chunking.create_chunk_results import create_chunk_results
from graphrag_chunking.text_chunk import TextChunk


class SentenceChunker(Chunker):
    """A chunker that splits text into sentence-based chunks."""

    def __init__(
        self, encode: Callable[[str], list[int]] | None = None, **kwargs: Any
    ) -> None:
        """Create a sentence chunker instance."""
        self._encode = encode
        bootstrap()

    def chunk(
        self, text: str, transform: Callable[[str], str] | None = None
    ) -> list[TextChunk]:
        """Chunk the text into sentence-based chunks."""
        sentences = nltk.sent_tokenize(text.strip())
        results = create_chunk_results(
            sentences, transform=transform, encode=self._encode
        )
        # nltk may trim whitespace; locate each original sentence in the source
        # with a running cursor so repeated sentences and transforms (which
        # change result.text) still get the correct span.
        cursor = 0
        for result in results:
            actual_start = text.find(result.original, cursor)
            if actual_start < 0:
                continue
            result.start_char = actual_start
            result.end_char = actual_start + len(result.original) - 1
            cursor = actual_start + len(result.original)
        return results
