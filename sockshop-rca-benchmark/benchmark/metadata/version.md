# Dataset Version History

## Version 1.0.0  (2026-01-15)
**Status:** Initial Release  
**Description:**  
- First public release of the Sock Shop AIOps Multi-Modal Dataset.  
- Includes full system topology, baseline workload traces, normal-run metrics/logs/traces, and 10 fault-injection experiments.
- Covers faults: CPU hog, memory hog, network delay, network loss, pod kill, DB slowdown, DB down, service crash, queue blocking, and cache miss storms.

**Included Modules**
- metadata/
- normal_run/
- fault_run/ 

---

## Version 1.1.0  (2026-02-01)
**Changes:**  
- Added 5 new fault injection cases:
  - High latency in `payment`
  - Catalogue-DB down
  - Message queue saturation
  - Shipping service thread exhaustion
  - Orders timeout propagation
- Added Kubernetes events export for each run.
- Improved workload generator randomness & behavior diversity.

---

## Version 1.1.1  (2026-02-10)
**Fixes:**  
- Fixed incorrect timestamps in Jaeger trace export.
- Cleaned duplicated log lines from `carts` and `orders` services.
- Normalized JSON schema for fault_metadata.json.

---

## Version 1.2.0 (2026-03-05)
**Changes:**  
- Added Prometheus metric label normalization.
- Added per-run resource usage snapshot (kubectl top nodes/pods).
- Added more detailed service topology with dependency directions.

---

## Versioning Policy
- **MAJOR (X.0.0)** — Breaking dataset structure changes or major schema redesign.
- **MINOR (0.X.0)** — Adding new fault scenarios, new traces, or new data modalities.
- **PATCH (0.0.X)** — Fixing mistakes, cleaning data, small corrections.

---

## Citation
If you use this dataset, please cite:

@dataset{sockshop_aiops_dataset,
title={Sock Shop Multi-Modal Fault Injection Dataset},
author={Fan Zhang, Dittaya Wanvarie},
year={2026},
version={1.0.0},
paper-url={https://...},
dataset-url={https://github.com/Seabook1212}
}