from datetime import datetime

import pytest

httpx = pytest.importorskip("httpx")

from extract_bench.inference.providers.extract import llamaextract_v2_api as module  # noqa: E402
from extract_bench.schemas.pipeline import PipelineSpec  # noqa: E402
from extract_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult  # noqa: E402
from extract_bench.schemas.product import ProductType  # noqa: E402


@pytest.mark.parametrize(
    "telemetry",
    [
        {"latency": {"parse_ms": 1000}, "metadata_state_transitions": [{"status": "COMPLETED"}]},
        {},
        {"latency": None, "metadata_state_transitions": None},
    ],
)
def test_poll_retains_optional_service_telemetry(monkeypatch, telemetry):
    provider = module.LlamaExtractV2Provider("llamaextract_v2", {"api_key": "offline"})
    calls = []
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    def handler(request):
        calls.append(request)
        assert request.url.params["project_id"] == "project"
        assert set(request.url.params.get_list("expand")) == {
            "metadata",
            "extract_metadata",
            "latency",
            "metadata_state_transitions",
        }
        status = "RUNNING" if len(calls) == 1 else "COMPLETED"
        return httpx.Response(
            200,
            json={
                "status": status,
                "extract_result": {"amount": 12},
                "metadata": {"usage": {"num_pages_extracted": 1}},
                "extract_metadata": {"parse_job_id": "parse-1"},
                **telemetry,
            },
        )

    with httpx.Client(base_url="https://example.test", transport=httpx.MockTransport(handler)) as client:
        result = provider._poll_job(client, "job-1", {"project_id": "project"})
    assert len(calls) == 2
    assert result["latency"] == telemetry.get("latency")
    assert result["metadata_state_transitions"] == telemetry.get("metadata_state_transitions")
    assert result["metadata"]["usage"]["num_pages_extracted"] == 1
    assert result["spawned_parse_job_id"] == "parse-1"
    assert result["data"] == {"amount": 12}
    pipeline = PipelineSpec(
        pipeline_name="fixture", provider_name="llamaextract_v2", product_type=ProductType.EXTRACT, config={}
    )
    raw = RawInferenceResult(
        request=InferenceRequest(example_id="one", source_file_path="one.pdf", product_type=ProductType.EXTRACT),
        pipeline=pipeline,
        pipeline_name="fixture",
        product_type=ProductType.EXTRACT,
        raw_output=result,
        started_at=datetime(2026, 1, 1),
        completed_at=datetime(2026, 1, 1),
        latency_in_ms=0,
    )
    normalized = provider.normalize(raw)
    saved = normalized.model_dump(mode="json")
    assert saved["raw_output"]["latency"] == telemetry.get("latency")
    assert saved["raw_output"]["metadata_state_transitions"] == telemetry.get("metadata_state_transitions")
