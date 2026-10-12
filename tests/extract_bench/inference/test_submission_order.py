import json
import random
from types import SimpleNamespace
from unittest.mock import patch

from extract_bench.inference.cli import InferenceCLI
from extract_bench.inference.runner import InferenceRunner
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.product import ProductType


class RecordingRunner(InferenceRunner):
    observed = []
    admitted = []

    async def _process_test_case_with_semaphore(self, semaphore, test_case, *args):
        self.admitted.append(test_case.test_id)
        return await super()._process_test_case_with_semaphore(semaphore, test_case, *args)

    def _process_test_case(self, test_case, product_type):
        self.observed.append(test_case.test_id)
        return None, None, None


def run_mock_cli(tmp_path, name, seed, cases, concurrent=1, **kwargs):
    spec = PipelineSpec(pipeline_name=name, provider_name="fake", product_type=ProductType.EXTRACT, config={})
    RecordingRunner.observed = []
    RecordingRunner.admitted = []
    with (
        patch("extract_bench.inference.cli.get_pipeline", return_value=spec),
        patch("extract_bench.inference.cli.load_test_cases", return_value=cases),
        patch("extract_bench.inference.cli._detect_product_type", return_value=ProductType.EXTRACT),
        patch("extract_bench.inference.cli.create_provider", return_value=object()),
        patch("extract_bench.inference.cli.InferenceRunner", RecordingRunner),
    ):
        result = InferenceCLI().run(
            name,
            input_dir=tmp_path,
            output_dir=tmp_path,
            shuffle_seed=seed,
            max_concurrent=concurrent,
            no_rich=True,
            force=True,
            force_exit_on_completion=False,
            **kwargs,
        )
    assert result == 0
    return list(RecordingRunner.observed), json.loads((tmp_path / name / "_submission_order.json").read_text())


def test_shuffle_same_across_pipelines_and_input_order(tmp_path):
    cases = [SimpleNamespace(test_id=f"doc-{i:02}", file_path=tmp_path / f"{i}.pdf") for i in range(30)]
    first, manifest = run_mock_cli(tmp_path, "llama", 0, cases)
    second, other_manifest = run_mock_cli(tmp_path, "acu", 0, list(reversed(cases)))
    expected = sorted(c.test_id for c in cases)
    random.Random(0).shuffle(expected)
    assert first == second == expected
    assert manifest == other_manifest == {"shuffle_seed": 0, "example_ids": expected}
    third, _ = run_mock_cli(tmp_path, "other-seed", 1, cases)
    assert third != first
    assert sorted(third) == sorted(first)


def test_async_admission_preserves_shuffle(tmp_path):
    cases = [SimpleNamespace(test_id=f"doc-{i:02}", file_path=tmp_path / f"{i}.pdf") for i in range(30)]
    observed, manifest = run_mock_cli(tmp_path, "async", 0, cases, concurrent=2)
    assert RecordingRunner.admitted == manifest["example_ids"]
    assert sorted(observed) == sorted(manifest["example_ids"])


def test_no_seed_preserves_discovery_order(tmp_path):
    cases = [SimpleNamespace(test_id=f"doc-{i}", file_path=tmp_path / f"{i}.pdf") for i in [3, 1, 2]]
    observed, manifest = run_mock_cli(tmp_path, "unshuffled", None, cases)
    assert observed == ["doc-3", "doc-1", "doc-2"]
    assert manifest["shuffle_seed"] is None


def test_top_level_forwards_shuffle(tmp_path):
    from extract_bench.cli import BenchCLI
    from extract_bench.pipeline.cli import PipelineCLI

    with patch.object(PipelineCLI, "run", return_value=0) as run:
        assert BenchCLI().run("fake", input_dir=tmp_path, shuffle_seed=17) == 0
    assert run.call_args.kwargs["shuffle_seed"] == 17


def test_pipeline_forwards_shuffle(tmp_path):
    from extract_bench.pipeline.cli import PipelineCLI

    # Stop at the inference boundary; no provider or evaluation runs.
    with patch.object(InferenceCLI, "run", return_value=1) as run:
        assert PipelineCLI().run("fake", input_dir=tmp_path, shuffle_seed=17) == 1
    assert run.call_args.kwargs["shuffle_seed"] == 17
