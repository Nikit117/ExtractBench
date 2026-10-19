from __future__ import annotations

from pathlib import Path

import pytest

from extract_bench.inference import runner as runner_module
from extract_bench.inference.providers.base import ProviderTransientError
from extract_bench.inference.providers.registry import create_provider
from extract_bench.inference.runner import DEFAULT_TIMEOUT_RETRIES, InferenceRunner
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from extract_bench.schemas.product import ProductType


@pytest.mark.parametrize("provider_name", ["claude_code_extract", "codex_code_extract"])
def test_a_coding_agent_document_gets_one_run_and_is_never_restarted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider_name: str
) -> None:
    """Claude Code and Codex documents were restarted from zero after an error (including the provider's own
    20-minute stop), and their stored latency and cost covered only the last run. An error now fails the
    document after its one run."""
    pytest.importorskip("pypdf", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")
    monkeypatch.setattr(runner_module, "INITIAL_BACKOFF_S", 0.0)
    pipeline = PipelineSpec(
        pipeline_name=f"{provider_name}_one_run", provider_name=provider_name, product_type=ProductType.EXTRACT
    )
    provider = create_provider(pipeline)
    calls: list[str] = []

    def overloaded(pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        calls.append(request.example_id)
        raise ProviderTransientError("overloaded")

    monkeypatch.setattr(provider, "run_inference", overloaded)
    source = tmp_path / "doc.pdf"
    source.write_text("dummy-pdf")
    runner = InferenceRunner(
        provider=provider, pipeline=pipeline, output_dir=tmp_path / "output", max_concurrent=1, use_rich=False
    )

    _, _, error_info = runner._process_document(source, "doc", ProductType.EXTRACT)

    assert calls == ["doc"]
    assert error_info is not None and error_info[2] == "ProviderTransientError"
    runner.shutdown()


def test_timed_out_files_are_not_retried_by_default() -> None:
    assert DEFAULT_TIMEOUT_RETRIES == 0
