import json
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agentic_harness import MemoryQuery, MemoryRecord, StructuredMemoryService


def main() -> None:
    database_url = "postgresql://postgres:postgres@localhost:5432/agentic_harness"
    service = StructuredMemoryService(
        pathlib.Path(".workflow_memory_structured_verify"),
        database_url=database_url,
    )
    namespace = f"structured_verify_{uuid.uuid4().hex[:12]}"
    service.remember(
        MemoryRecord.create(
            namespace=namespace,
            memory_type="profile",
            content="Enterprise account renewal profile for Acme Corp.",
            source_run_id="structured-run-1",
            source_step_id="seed_structured_memory",
            metadata={"source": "crm", "owner": "sales_ops"},
            structured_payload={
                "customer": {"name": "Acme Corp", "tier": "enterprise", "region": "emea"},
                "renewal": {"month": "2026-09", "risk": "medium"},
            },
        )
    )
    service.remember(
        MemoryRecord.create(
            namespace=namespace,
            memory_type="profile",
            content="SMB account profile for Elm Co.",
            source_run_id="structured-run-2",
            source_step_id="seed_structured_memory",
            metadata={"source": "crm", "owner": "sales_ops"},
            structured_payload={
                "customer": {"name": "Elm Co", "tier": "smb", "region": "na"},
                "renewal": {"month": "2026-11", "risk": "low"},
            },
        )
    )
    results = service.recall(
        MemoryQuery(
            namespace=namespace,
            text="renewal",
            max_results=3,
            memory_types=["profile"],
            metadata_filters={"source": "crm"},
            structured_filters={"customer.tier": "enterprise", "customer.region": "emea"},
        )
    )
    print(
        json.dumps(
            {
                "namespace": namespace,
                "result_count": len(results),
                "top_result": results[0].record.content if results else None,
                "top_payload": results[0].record.structured_payload if results else None,
                "capabilities": service.descriptor.capabilities,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
