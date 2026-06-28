"""Tests for YAML serialization with round-trip compatibility."""

import pytest
import yaml
from pathlib import Path
from armature.core.matrix import Matrix
from armature.core.dependencies import DependencyCell, TemporalDependency, ExistentialDependency
from armature.serialization import YAMLCodec


class TestYAMLCodecBasics:
    """Basic save and load functionality tests."""

    def test_save_creates_file(self, tmp_path):
        m = Matrix(activities=["A", "B"])
        path = tmp_path / "test.yaml"
        YAMLCodec.save(m, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_load_reads_file(self, tmp_path):
        m = Matrix(activities=["A", "B"], description="Test matrix")
        path = tmp_path / "test.yaml"
        YAMLCodec.save(m, path)
        loaded = YAMLCodec.load(path)
        assert isinstance(loaded, Matrix)
        assert loaded.activities == ["A", "B"]
        assert loaded.description == "Test matrix"

    def test_enums_serialize_as_strings(self, tmp_path):
        m = Matrix(activities=["A", "B"])
        m.set_cell("A", "B", DependencyCell(
            temporal=TemporalDependency.DIRECT,
            existential=ExistentialDependency.IMPLICATION,
        ))
        path = tmp_path / "test.yaml"
        YAMLCodec.save(m, path)
        yaml_text = path.read_text()
        assert "direct" in yaml_text
        assert "implication" in yaml_text
        assert "TemporalDependency" not in yaml_text
        assert "ExistentialDependency" not in yaml_text

    def test_metadata_fields_in_yaml(self, tmp_path):
        m = Matrix(
            activities=["A"],
            description="Test description",
            source="test.xes",
            created_at="2026-01-22T00:00:00Z",
            classification="sequential",
        )
        path = tmp_path / "test.yaml"
        YAMLCodec.save(m, path)
        data = yaml.safe_load(path.read_text())
        assert data["metadata"]["format_version"] == "2.0"
        assert data["metadata"]["description"] == "Test description"
        assert data["metadata"]["source"] == "test.xes"
        assert data["metadata"]["created_at"] == "2026-01-22T00:00:00Z"
        assert data["metadata"]["classification"] == "sequential"


class TestRoundTripEquivalence:
    """Round-trip equivalence tests (save → load produces identical matrix)."""

    def test_empty_matrix_round_trip(self, tmp_path):
        original = Matrix()
        path = tmp_path / "empty.yaml"
        YAMLCodec.save(original, path)
        loaded = YAMLCodec.load(path)
        assert loaded == original
        assert loaded.format_version == "2.0"
        assert loaded.activities == []
        assert loaded.dependencies == {}

    def test_matrix_with_activities_round_trip(self, tmp_path):
        original = Matrix(activities=["Submit", "Review", "Approve"])
        path = tmp_path / "activities.yaml"
        YAMLCodec.save(original, path)
        loaded = YAMLCodec.load(path)
        assert loaded == original
        assert loaded.activities == ["Submit", "Review", "Approve"]
        assert loaded.activities[0] == "Submit"
        assert loaded.activities[1] == "Review"
        assert loaded.activities[2] == "Approve"

    def test_matrix_with_dependencies_round_trip(self, tmp_path):
        original = Matrix(activities=["A", "B", "C"])
        original.set_cell("A", "B", DependencyCell(
            temporal=TemporalDependency.DIRECT,
            existential=ExistentialDependency.IMPLICATION,
        ))
        original.set_cell("B", "C", DependencyCell(
            temporal=TemporalDependency.TRUE_EVENTUAL,
            existential=ExistentialDependency.INDEPENDENCE,
        ))
        original.set_cell("A", "C", DependencyCell(
            temporal=TemporalDependency.NO_ORDERING,
            existential=ExistentialDependency.NEGATED_EQUIVALENCE,
        ))
        path = tmp_path / "dependencies.yaml"
        YAMLCodec.save(original, path)
        loaded = YAMLCodec.load(path)
        assert loaded == original
        assert loaded.get_cell("A", "B").temporal == TemporalDependency.DIRECT
        assert loaded.get_cell("A", "B").existential == ExistentialDependency.IMPLICATION
        assert loaded.get_cell("B", "C").temporal == TemporalDependency.TRUE_EVENTUAL
        assert loaded.get_cell("B", "C").existential == ExistentialDependency.INDEPENDENCE
        assert loaded.get_cell("A", "C").temporal == TemporalDependency.NO_ORDERING
        assert loaded.get_cell("A", "C").existential == ExistentialDependency.NEGATED_EQUIVALENCE

    def test_matrix_with_full_metadata_round_trip(self, tmp_path):
        original = Matrix(
            activities=["X", "Y"],
            description="Full metadata test",
            source="test_log.xes",
            created_at="2026-01-22T10:30:00Z",
            classification="parallel",
        )
        original.set_cell("X", "Y", DependencyCell(
            temporal=TemporalDependency.DIRECT,
            existential=ExistentialDependency.IMPLICATION,
        ))
        path = tmp_path / "full_metadata.yaml"
        YAMLCodec.save(original, path)
        loaded = YAMLCodec.load(path)
        assert loaded == original
        assert loaded.format_version == "2.0"
        assert loaded.description == "Full metadata test"
        assert loaded.source == "test_log.xes"
        assert loaded.created_at == "2026-01-22T10:30:00Z"
        assert loaded.classification == "parallel"

    def test_all_temporal_types_round_trip(self, tmp_path):
        acts = ["A", "B", "C", "D", "E", "F", "G", "H"]
        original = Matrix(activities=acts)
        temporal_types = [
            TemporalDependency.DIRECT,
            TemporalDependency.DIRECT_BACKWARD,
            TemporalDependency.TRUE_EVENTUAL,
            TemporalDependency.TRUE_EVENTUAL_BACKWARD,
            TemporalDependency.EVENTUAL,
            TemporalDependency.EVENTUAL_BACKWARD,
            TemporalDependency.INDEPENDENCE,
            TemporalDependency.NO_ORDERING,
        ]
        for i, t in enumerate(temporal_types):
            original.set_cell(acts[i], acts[(i + 1) % len(acts)], DependencyCell(temporal=t))
        path = tmp_path / "temporal_types.yaml"
        YAMLCodec.save(original, path)
        loaded = YAMLCodec.load(path)
        assert loaded == original

    def test_all_existential_types_round_trip(self, tmp_path):
        acts = ["A", "B", "C", "D", "E", "F", "G"]
        original = Matrix(activities=acts)
        existential_types = [
            ExistentialDependency.IMPLICATION,
            ExistentialDependency.IMPLICATION_BACKWARD,
            ExistentialDependency.EQUIVALENCE,
            ExistentialDependency.NEGATED_EQUIVALENCE,
            ExistentialDependency.OR,
            ExistentialDependency.NAND,
            ExistentialDependency.INDEPENDENCE,
        ]
        for i, e in enumerate(existential_types):
            original.set_cell(acts[i], acts[(i + 1) % len(acts)], DependencyCell(existential=e))
        path = tmp_path / "existential_types.yaml"
        YAMLCodec.save(original, path)
        loaded = YAMLCodec.load(path)
        assert loaded == original


class TestYAMLFormatStructure:
    """Tests for YAML format structure and readability."""

    def test_yaml_has_required_keys(self, tmp_path):
        m = Matrix(activities=["A", "B"])
        m.set_cell("A", "B", DependencyCell())
        path = tmp_path / "structure.yaml"
        YAMLCodec.save(m, path)
        data = yaml.safe_load(path.read_text())
        assert "metadata" in data
        assert "activities" in data
        assert "dependencies" in data

    def test_format_version_in_metadata(self, tmp_path):
        m = Matrix(activities=["A"])
        path = tmp_path / "version.yaml"
        YAMLCodec.save(m, path)
        data = yaml.safe_load(path.read_text())
        assert data["metadata"]["format_version"] == "2.0"

    def test_format_version_in_yaml_text(self, tmp_path):
        m = Matrix(activities=["A"])
        path = tmp_path / "version.yaml"
        YAMLCodec.save(m, path)
        yaml_text = path.read_text()
        assert "format_version:" in yaml_text
        assert "2.0" in yaml_text or "'2.0'" in yaml_text

    def test_dependencies_is_flat_list(self, tmp_path):
        m = Matrix(activities=["A", "B", "C"])
        m.set_cell("A", "B", DependencyCell(
            temporal=TemporalDependency.DIRECT,
            existential=ExistentialDependency.IMPLICATION,
        ))
        path = tmp_path / "flat.yaml"
        YAMLCodec.save(m, path)
        data = yaml.safe_load(path.read_text())
        assert isinstance(data["dependencies"], list)
        assert len(data["dependencies"]) == 1
        dep = data["dependencies"][0]
        assert dep["from"] == "A"
        assert dep["to"] == "B"
        assert dep["temporal"]["type"] == "direct"
        assert dep["temporal"]["direction"] == "forward"
        assert dep["existential"]["type"] == "implication"
        assert dep["existential"]["direction"] == "forward"

    def test_dependency_has_symbol_fields(self, tmp_path):
        m = Matrix(activities=["A", "B"])
        m.set_cell("A", "B", DependencyCell(
            temporal=TemporalDependency.DIRECT,
            existential=ExistentialDependency.IMPLICATION,
        ))
        path = tmp_path / "symbols.yaml"
        YAMLCodec.save(m, path)
        data = yaml.safe_load(path.read_text())
        dep = data["dependencies"][0]
        assert dep["temporal"]["symbol"] == "≺_d"
        assert dep["existential"]["symbol"] == "⇒"

    def test_sparse_format_only_nondefault_cells(self, tmp_path):
        m = Matrix(activities=["A", "B", "C"])
        m.set_cell("A", "B", DependencyCell(
            temporal=TemporalDependency.DIRECT,
            existential=ExistentialDependency.IMPLICATION,
        ))
        path = tmp_path / "sparse.yaml"
        YAMLCodec.save(m, path)
        data = yaml.safe_load(path.read_text())
        # Only A->B should be in the flat list
        assert len(data["dependencies"]) == 1
        assert data["dependencies"][0]["from"] == "A"
        assert data["dependencies"][0]["to"] == "B"

    def test_backward_direction_encoded(self, tmp_path):
        m = Matrix(activities=["A", "B"])
        m.set_cell("A", "B", DependencyCell(
            temporal=TemporalDependency.DIRECT_BACKWARD,
            existential=ExistentialDependency.IMPLICATION_BACKWARD,
        ))
        path = tmp_path / "backward.yaml"
        YAMLCodec.save(m, path)
        data = yaml.safe_load(path.read_text())
        dep = data["dependencies"][0]
        assert dep["temporal"]["type"] == "direct"
        assert dep["temporal"]["direction"] == "backward"
        assert dep["existential"]["type"] == "implication"
        assert dep["existential"]["direction"] == "backward"


class TestErrorHandling:
    """Tests for error handling in load/save operations."""

    def test_load_nonexistent_file_raises_error(self):
        with pytest.raises(FileNotFoundError):
            YAMLCodec.load("nonexistent.yaml")

    def test_load_invalid_yaml_raises_error(self, tmp_path):
        path = tmp_path / "invalid.yaml"
        path.write_text("invalid: yaml: content: [unclosed")
        with pytest.raises(yaml.YAMLError):
            YAMLCodec.load(path)

    def test_load_dependency_unknown_activity_raises_error(self, tmp_path):
        """Dependency referencing activity not in activities list must fail."""
        path = tmp_path / "bad_schema.yaml"
        bad = {
            "metadata": {"format_version": "2.0"},
            "activities": ["A"],
            "dependencies": [{
                "from": "A", "to": "UNKNOWN",
                "temporal": {"type": "direct", "symbol": "≺_d", "direction": "forward"},
                "existential": {"type": "implication", "symbol": "⇒", "direction": "forward"},
            }],
        }
        path.write_text(yaml.dump(bad, allow_unicode=True))
        with pytest.raises(Exception):
            YAMLCodec.load(path)


class TestPerformance:
    """Performance tests with larger matrices."""

    def test_larger_matrix_round_trip(self, tmp_path):
        activities = [f"Activity_{i}" for i in range(20)]
        original = Matrix(activities=activities)
        for i in range(19):
            original.set_cell(
                f"Activity_{i}",
                f"Activity_{i+1}",
                DependencyCell(
                    temporal=TemporalDependency.DIRECT,
                    existential=ExistentialDependency.IMPLICATION,
                ),
            )
        path = tmp_path / "large.yaml"
        YAMLCodec.save(original, path)
        loaded = YAMLCodec.load(path)
        assert loaded == original
        assert len(loaded.activities) == 20
        for i in range(19):
            cell = loaded.get_cell(f"Activity_{i}", f"Activity_{i+1}")
            assert cell.temporal == TemporalDependency.DIRECT
            assert cell.existential == ExistentialDependency.IMPLICATION
