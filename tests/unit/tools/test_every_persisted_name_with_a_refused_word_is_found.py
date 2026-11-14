"""The persisted names scan (tools/persisted_names_scan.py): every string literal, data key and container key that
carries a refused word is found, and nothing else; a container's weights are never opened."""
import io
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
import persisted_names_scan as S  # noqa: E402
from public_vocabulary import TOOL, VERB  # noqa: E402


def test_a_python_literal_is_found_and_a_comment_is_not(tmp_path):
    src = tmp_path / "m.py"
    src.write_text(f'"""The {VERB}d module."""\n'
                   f'# a {VERB} in a comment is no literal\n'
                   f'X = spec.get("{VERB}_value")\n'
                   f'ENV = "NBX_{VERB.upper()}_NAN"\n'
                   f'OK = "a {VERB}back is no word of ours"\n', encoding="utf-8")
    rows = S.source([str(tmp_path)])
    assert [(r[1], r[2], r[3]) for r in rows] == [
        (1, "docstring", VERB + "d"), (3, "string", VERB + "_value"), (4, "string", f"NBX_{VERB.upper()}_NAN")]


def test_a_data_key_is_told_from_a_data_value(tmp_path):
    (tmp_path / "t.jsonl").write_text(json.dumps({f"{VERB}_size": 3, "by": TOOL, "n": f"un{VERB}able"}) + "\n",
                                      encoding="utf-8")
    (tmp_path / "c.yml").write_text(f"{VERB}_seq_len: 8\nnote: {TOOL.title()} wrote it  # {VERB} here is a comment\n",
                                    encoding="utf-8")
    rows = sorted((Path(r[0]).name, r[1], r[2], r[3]) for r in S.source([str(tmp_path)]))
    assert rows == [("c.yml", 1, "key", f"{VERB}_seq_len"), ("c.yml", 2, "value", TOOL.title()),
                    ("t.jsonl", 1, "key", f"{VERB}_size"), ("t.jsonl", 1, "value", TOOL)]


def test_a_member_read_in_chunks_counts_what_a_whole_parse_counts():
    doc = {"ops": [{"d": {"type": "symbol", VERB: i, f"{VERB}_value": 7, "note": f"{VERB}d at {i}"}}
                   for i in range(40)], "fine": [f"{VERB}back", f"con{VERB[:-1]}ted"]}
    raw = json.dumps(doc).encode()
    for chunk in (1, 5, 64, 1 << 20):
        keys, values = S.member_words(io.BytesIO(raw), chunk=chunk)
        assert keys == {VERB: 40, f"{VERB}_value": 40}, chunk
        assert sum(values.values()) == 40 and len(values) == 40, chunk


def test_a_container_is_read_by_its_json_members_only(tmp_path):
    nbx = tmp_path / "model.nbx"
    with zipfile.ZipFile(nbx, "w") as z:
        z.writestr("manifest.json", json.dumps({f"{VERB}_resolution": 1024, "name": "m"}))
        z.writestr("components/a/graph.json", json.dumps({"s": {"x": {f"{VERB}_value": 4}}}))
        z.writestr("components/a/weights/shard_000.safetensors", f'{{"{VERB}": 1}}')
    [found] = list(S.containers([str(tmp_path)]))
    assert found["json_members"] == 2 and found["member_names"] == []
    assert found["members"] == {
        "manifest.json": {"keys": {f"{VERB}_resolution": 1}, "values": {}, "distinct_values": 0},
        "components/a/graph.json": {"keys": {f"{VERB}_value": 1}, "values": {}, "distinct_values": 0}}
