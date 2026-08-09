from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from agentic_vol_regime_app.data.forecast_gcs import ForecastGCSPublisher
from agentic_vol_regime_app.data.sector_history_gcs import StorageManifestConflictError, StorageObjectConflictError, StorageObjectMetadata
from agentic_vol_regime_app.forecast_contract import build_forecast_artifact, canonical_json_bytes


@dataclass
class FakeStorage:
    objects: dict
    generation: int = 0
    def bucket_exists(self, bucket): return bucket == "bucket"
    def get_object_metadata(self, bucket, object_name):
        item = self.objects.get((bucket, object_name))
        return StorageObjectMetadata(bucket, object_name, str(item[1]), len(item[0])) if item else None
    def download_bytes(self, bucket, object_name): return self.objects[(bucket, object_name)][0]
    def upload_bytes(self, *, bucket, object_name, data, if_generation_match, content_type):
        key = (bucket, object_name); old = self.objects.get(key)
        if if_generation_match == 0 and old: raise StorageManifestConflictError("exists")
        if if_generation_match not in (None, 0) and (not old or str(old[1]) != str(if_generation_match)): raise StorageManifestConflictError("race")
        self.generation += 1; self.objects[key] = (data, self.generation, content_type)
        return StorageObjectMetadata(bucket, object_name, str(self.generation), len(data))


def artifact(review=False):
    outputs = {"daily_report": {"markdown": "# report"}, "belief_state": {"as_of": "2026-08-08T20:00:00Z", "model_version": "v1", "beliefs": {"A": .9}}, "transition_probabilities": {}, "alert_record": {"requires_human_review": False}, "policy_recommendation": {}, "critic_review": {"requires_human_review": review}, "data_quality": {"is_complete": True}, "feature_record": {"schema_version": "features.v1"}}
    return build_forecast_artifact(named_outputs=outputs, run_id="run", workflow_id="daily_vol_regime_report", agent_id="agent", agent_metadata={}, observation={"as_of": "2026-08-08T20:00:00Z", "source": "test", "symbols": {"SPY": {}}, "schema_version": "observation.v1"})


def test_contract_is_deterministic_and_review_is_ineligible():
    assert artifact()["forecast_id"] == artifact()["forecast_id"]
    flagged = artifact(True)
    assert flagged["decision_eligible"] is False and flagged["review_status"] == "REQUIRED"
    with pytest.raises(ValueError): canonical_json_bytes({"bad": float("nan")})


def test_first_publish_retry_and_collision():
    storage = FakeStorage({}); publisher = ForecastGCSPublisher(storage_client=storage, bucket="bucket")
    result = publisher.publish(artifact=artifact(), report_markdown="# report")
    assert result.verified and len(result.uploaded_objects) == 3
    retry = publisher.publish(artifact=artifact(), report_markdown="# report")
    assert retry.verified and len(retry.reused_objects) == 2
    name = retry.forecast["object_name"]
    storage.objects[("bucket", name)] = (b"wrong", 99, "application/json")
    with pytest.raises(StorageObjectConflictError): publisher.publish(artifact=artifact(), report_markdown="# report")


def test_dry_run_does_not_mutate_storage():
    storage = FakeStorage({})
    result = ForecastGCSPublisher(storage_client=storage, bucket="bucket").publish(artifact=artifact(), report_markdown="# report", dry_run=True)
    assert result.status == "dry_run" and storage.objects == {}
