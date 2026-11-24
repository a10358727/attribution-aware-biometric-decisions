"""[Public file; not part of the research code.] Skips the synthetic end-to-end tests that need private resources.

The test files themselves are byte-identical to the research versions. The tests listed here exercise a full driver
(gates, authorisation checks, frozen-record cross-checks) and need files that are not released: the research matcher
package (`diar`), the frozen E061 outcome record and the experiment protocol documents. They passed in the research
repository before each test read (see the experiment records). Every other test runs here.
"""
import pytest

PRIVATE = {
    "test_run_e064_synthetic.py::test_end_to_end_gates_pass_against_e061_e063_code": "needs the frozen E061 OUTCOME.json",
    "test_run_e064_synthetic.py::test_g0_failure_stops_before_bootstrap": "needs the frozen E061 OUTCOME.json",
    "test_run_e067_synthetic.py::test_stage_v_synthetic_end_to_end": "needs the E067 protocol document (authorisation check)",
    "test_run_e071_synthetic.py::test_validation_then_test_end_to_end": "needs the research matcher package (diar)",
    "test_run_e071_synthetic.py::test_gate_gx_stops_before_dumps": "needs the research matcher package (diar)",
    "test_run_e071_synthetic.py::test_gate_gd_stops_validation_without_tau_file": "needs the research matcher package (diar)",
    "test_run_e071_synthetic.py::test_gx_gc_pass_at_real_tolerances_against_frozen_e065": "needs the research matcher package (diar)",
}


def pytest_collection_modifyitems(config, items):
    for it in items:
        key = f"{it.fspath.basename}::{it.name}"
        if key in PRIVATE:
            it.add_marker(pytest.mark.skip(reason=f"private resource not released: {PRIVATE[key]}"))
