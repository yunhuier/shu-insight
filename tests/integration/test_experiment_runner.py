from pathlib import Path

from shu_insight.evaluation.dataset import load_frozen_dataset
from shu_insight.evaluation.runner import EXPERIMENTS, ExperimentRunner


def test_runner_emits_every_named_experiment_and_json_report(tmp_path: Path) -> None:
    dataset = load_frozen_dataset(
        Path("tests/fixtures/annotations/mini_frozen_set.jsonl"),
        allow_mini_fixture=True,
    )
    output = tmp_path / "report.json"

    report = ExperimentRunner(dataset).run()
    report.write_json(output)

    assert {result.name for result in report.experiments} == {
        experiment.name for experiment in EXPERIMENTS
    }
    assert output.is_file()
    assert all(
        0.0 <= value <= 1.0
        for result in report.experiments
        for value in result.metrics.values()
    )
    assert report.dataset_sha256 == dataset.sha256
