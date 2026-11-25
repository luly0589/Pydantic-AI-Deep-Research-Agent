"""Tests for the DDGS search adapter used by the research workflow."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from app import DuckDuckGoSearchError, duckduckgo_search


class DuckDuckGoSearchTests(unittest.TestCase):
    @patch("app.time.sleep")
    @patch("app.DDGS")
    def test_uses_ddgs_default_backend_and_normalizes_results(self, mock_ddgs, _mock_sleep) -> None:
        mock_ddgs.return_value.text.return_value = [
            {
                "title": "NVIDIA Investor Relations",
                "href": "https://investor.nvidia.com/",
                "body": "Investor information.",
            }
        ]

        sources = duckduckgo_search("NVIDIA stock analysis")

        self.assertEqual(sources[0].title, "NVIDIA Investor Relations")
        mock_ddgs.assert_called_once_with(timeout=20)
        mock_ddgs.return_value.text.assert_called_once_with(
            "NVIDIA stock analysis",
            max_results=5,
            region="us-en",
            safesearch="moderate",
        )

    @patch("app.time.sleep")
    @patch("app.DDGS")
    def test_retries_ddgs_exception_then_raises_clear_error(self, mock_ddgs, mock_sleep) -> None:
        mock_ddgs.return_value.text.side_effect = DDGSException("No results found.")

        with self.assertRaises(DuckDuckGoSearchError):
            duckduckgo_search("NVIDIA stock analysis")

        self.assertEqual(mock_ddgs.return_value.text.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @unittest.skipUnless(os.getenv("RUN_LIVE_SEARCH_TESTS") == "1", "Set RUN_LIVE_SEARCH_TESTS=1 to run live DDGS tests.")
    def test_default_ddgs_search_returns_results(self) -> None:
        results = DDGS(timeout=20).text("NVIDIA stock analysis", max_results=5)

        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
