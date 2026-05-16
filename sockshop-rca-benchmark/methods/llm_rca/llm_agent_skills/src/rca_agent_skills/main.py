from __future__ import annotations

from pathlib import Path
from copy import deepcopy

from rca_agent_skills.common.io_utils import read_json, read_yaml
from rca_agent_skills.common.logging_utils import configure_logging
from rca_agent_skills.common.models import TimeWindow
from rca_agent_skills.orchestrator_agent.agent import RCAOrchestratorAgent
from rca_agent_skills.orchestrator_agent.schemas import APIInputs, CSVInputs, RCARequest


def _load_config_bundle(project_root: Path) -> dict:
    configs = project_root / "configs"
    return {
        "settings": read_yaml(configs / "settings.yaml"),
        "metric_kpis": read_yaml(configs / "metric_kpis.yaml"),
        "fault_types": read_yaml(configs / "fault_types.yaml"),
        "rootcause_reasoning_rules": read_yaml(
            configs / "rootcause_reasoning_rules.yaml"
        ),
        "baseline_windows": (
            read_yaml(configs / "baseline_windows.yaml")
            if (configs / "baseline_windows.yaml").exists()
            else {}
        ),
        "prometheus_queries": read_yaml(
            configs / "query_templates" / "prometheus_queries.yaml"
        ),
        "loki_queries": read_yaml(configs / "query_templates" / "loki_queries.yaml"),
        "jaeger_queries": read_yaml(
            configs / "query_templates" / "jaeger_query_templates.yaml"
        ),
    }


def build_request(payload: dict, project_root: Path | None = None) -> RCARequest:
    root = project_root or Path(__file__).resolve().parents[2]
    config_bundle = _load_config_bundle(root)
    config_bundle["settings"] = deepcopy(config_bundle["settings"])
    csv_inputs = payload.get("csv_inputs")
    api_inputs = payload.get("api_inputs")

    if csv_inputs:
        csv_inputs = {
            key: (
                str((root / value).resolve())
                if value and not Path(value).expanduser().is_absolute()
                else value
            )
            for key, value in csv_inputs.items()
        }
    if api_inputs:
        api_inputs = dict(api_inputs)
        api_cfg = config_bundle["settings"].setdefault("api", {})
        if api_inputs.get("prometheus_url"):
            api_cfg.setdefault("prometheus", {})["base_url"] = api_inputs[
                "prometheus_url"
            ]
        if api_inputs.get("loki_url"):
            api_cfg.setdefault("loki", {})["base_url"] = api_inputs["loki_url"]
        if api_inputs.get("jaeger_url"):
            api_cfg.setdefault("jaeger", {})["base_url"] = api_inputs["jaeger_url"]
        if api_inputs.get("namespace"):
            config_bundle["settings"].setdefault("defaults", {})["namespace"] = (
                api_inputs["namespace"]
            )
    topology = payload.get("topology")
    if not topology:
        topology_file = (csv_inputs or {}).get("topology_file") or str(
            root / "configs" / "topology" / "sockshop_topology.json"
        )
        topology = read_json(topology_file)

    return RCARequest(
        incident_id=payload["incident_id"],
        abnormal_window=TimeWindow(**payload["abnormal_window"]),
        baseline_window=TimeWindow(**payload["baseline_window"]),
        backend_mode=payload["backend_mode"],
        topology=topology,
        api_inputs=APIInputs(**api_inputs) if api_inputs else None,
        csv_inputs=CSVInputs(**csv_inputs) if csv_inputs else None,
        namespace=payload.get(
            "namespace",
            config_bundle["settings"]["defaults"].get("namespace", "sock-shop"),
        ),
        config_bundle=config_bundle,
        execution_options=payload.get("execution_options", {}),
    )


def run_rca(payload: dict, project_root: Path | None = None):
    configure_logging()
    root = project_root or Path(__file__).resolve().parents[2]
    request = build_request(payload, root)
    settings = request.config_bundle["settings"]
    agent = RCAOrchestratorAgent(request, settings)
    return agent.run()


def run_request_file(path: str | Path):
    return run_rca(read_json(path), Path(__file__).resolve().parents[2])
