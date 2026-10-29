"""The device-file template a new backend starts from (docs/nbxtensor/device_file_template.yml, BACKENDS.md section 4)
names every key a library reader reads (docs/nbxtensor/device_keys.txt): a key the census gains is a key the template
gains, so a new device is never written without one the kernels will ask."""
from pathlib import Path

import yaml

DOCS = Path(__file__).resolve().parents[3] / "docs" / "nbxtensor"


def test_every_census_key_is_in_the_template():
    template = yaml.safe_load((DOCS / "device_file_template.yml").read_text())
    missing = []
    for key in (DOCS / "device_keys.txt").read_text().split():
        top, _, sub = key.partition(".")
        node = template.get(top, KeyError) if isinstance(template, dict) else KeyError
        if node is KeyError or (sub and (not isinstance(node, dict) or sub.split(".")[0] not in node)):
            missing.append(key)
    assert not missing, f"keys the library reads, absent from the device-file template: {missing}"


def test_the_template_states_no_value():
    template = yaml.safe_load((DOCS / "device_file_template.yml").read_text())
    values = [v for v in template.values() if not isinstance(v, dict)] + \
             [v for sec in template.values() if isinstance(sec, dict) for v in sec.values()]
    assert all(v is None for v in values), "a template carries no value: each comes from the device's own source"
