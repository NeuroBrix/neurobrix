"""Every daemon envelope names the protocol and the engine that produced it,
and the identity a client needs before its first request — engine, protocol,
endpoint, operations — is one record, read the same by `status` and by
`neurobrix info --json` (Studio requests 1 and 7).

Injection: `PROTOCOL_VERSION` dropped from `make_response` made the first
test RED; restored, green.
"""
from neurobrix.serving.protocol import PROTOCOL_VERSION, make_response, endpoint
from neurobrix.serving.engine import daemon_identity


def test_every_envelope_names_protocol_and_engine():
    ok = make_response({"x": 1}); err = make_response(error="no")
    for env in (ok, err):
        assert env["protocol"] == PROTOCOL_VERSION and isinstance(env["engine"], str)
    assert ok["result"] == {"x": 1} and err["error"] == "no"


def test_the_identity_record_is_one_record_for_status_and_info():
    ident = daemon_identity()
    assert ident["protocol"] == PROTOCOL_VERSION and ident["endpoint"]["kind"] in ("unix", "tcp")
    assert "status" in ident["operations"] and "generate" in ident["operations"]
    from neurobrix.cli.commands.info import info_record
    rec = info_record(None)
    assert rec["protocol"] == PROTOCOL_VERSION and rec["endpoint"] == endpoint() and rec["operations"] == ident["operations"]
