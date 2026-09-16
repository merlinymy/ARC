#!/usr/bin/env python
"""Fetch a figure through the real API route and check the bytes.

Runs the FastAPI app in-process (``TestClient``), so it exercises the actual
route, the actual record lookup and the actual ``FileResponse`` **without
touching the backend running on 8001**.  ``settings.processed_data_dir`` is
pointed at the scratch directory the extraction check wrote, so nothing reads or
writes the production library.

Then round-trips one caption chunk's payload through a **throwaway** Qdrant
collection -- dummy vectors, no embedding call -- to confirm the new figure
fields survive a write and a read, and deletes the collection.

    python scripts/w2b_figure_api_check.py --records data/w2b_figures/records
"""

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

THROWAWAY = "w2b_figure_payload_check"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/w2b_figures/records")
    ap.add_argument("--qdrant", default="http://localhost:6333")
    ap.add_argument("--skip-qdrant", action="store_true")
    args = ap.parse_args()

    records_dir = Path(args.records).resolve()

    # Point the app at the scratch records *before* anything resolves the path.
    from config import settings
    settings.processed_data_dir = records_dir

    from preprocessing import paper_record as pr
    from preprocessing.chunker import PaperChunker

    assert pr.records_dir() == records_dir, pr.records_dir()

    paths = sorted(records_dir.glob("*.json"))
    if not paths:
        sys.exit(f"no records in {records_dir}")

    from fastapi.testclient import TestClient
    import api.main as main_module

    client = TestClient(main_module.app)
    failures = []

    for path in paths:
        record = json.loads(path.read_text())
        paper_id = record["paper_id"]

        listing = client.get(f"/papers/{paper_id}/figures")
        if listing.status_code != 200:
            failures.append(f"{paper_id}: list -> {listing.status_code}")
            continue
        data = listing.json()
        print(f"\n{record['file_name'][:56]}")
        print(f"  GET /papers/{paper_id}/figures -> 200, "
              f"{data['total']} objects, images_available={data['images_available']}, "
              f"schema_version={data['schema_version']}")

        checked = 0
        for figure in data["figures"]:
            if not figure["has_image"]:
                continue
            url = f"/papers/{paper_id}/figures/{figure['figure_id']}/image"
            if figure["image_url"] != url:
                failures.append(f"{paper_id}: image_url {figure['image_url']} != {url}")
            response = client.get(url)
            if response.status_code != 200:
                failures.append(f"{paper_id}/{figure['figure_id']}: {response.status_code}")
                continue
            body = response.content

            # Is it a real image, and is it *this* figure's crop?
            from io import BytesIO
            from PIL import Image
            with Image.open(BytesIO(body)) as img:
                fmt, size = img.format, img.size
            entry = pr.figure_object(record, figure["figure_id"])
            on_disk = pr.figure_image(record, entry)
            same = hashlib.sha256(body).hexdigest() == \
                hashlib.sha256(on_disk.read_bytes()).hexdigest()
            if not same:
                failures.append(f"{paper_id}/{figure['figure_id']}: bytes differ from disk")
            if checked < 2:
                print(f"  GET .../{figure['figure_id']}/image -> 200 "
                      f"{response.headers.get('content-type')} {len(body)} bytes, "
                      f"{fmt} {size[0]}x{size[1]}, "
                      f"sha256 matches the file the record names: {same}")
                print(f"      label={figure['label']!r} page={figure['page']} "
                      f"bbox={figure['bbox']} bbox_is_image={figure['bbox_is_image']}")
                print(f"      caption={figure['caption'][:96]!r}")
            checked += 1

        # A figure id that does not exist, and one with no crop
        missing = client.get(f"/papers/{paper_id}/figures/figure_9999/image")
        if missing.status_code != 404:
            failures.append(f"{paper_id}: bogus figure id -> {missing.status_code}")
        print(f"  GET .../figure_9999/image -> {missing.status_code} "
              f"({missing.json().get('detail', '')[:60]})")

    # Schema-1 record: the endpoint must 404 the image, not serve a stale path.
    legacy = Path("data/w1_sample/records")
    if legacy.is_dir():
        settings.processed_data_dir = legacy.resolve()
        old = sorted(legacy.glob("*.json"))[0]
        old_record = json.loads(old.read_text())
        pid = old_record["paper_id"]
        listing = client.get(f"/papers/{pid}/figures").json()
        with_images = sum(1 for f in listing["figures"] if f["has_image"])
        first = next((f for f in listing["figures"] if f["kind"] == "figure"), None)
        image = client.get(f"/papers/{pid}/figures/{first['figure_id']}/image") \
            if first else None
        print(f"\nschema-1 record {pid} ({old_record['file_name'][:40]}):")
        print(f"  {listing['total']} objects listed, has_image on {with_images}, "
              f"images_available={listing['images_available']}, "
              f"schema_version={listing['schema_version']}")
        if image is not None:
            print(f"  GET .../{first['figure_id']}/image -> {image.status_code} "
                  f"({image.json().get('detail', '')[:88]})")
            if image.status_code != 404:
                failures.append("schema-1 record served an image it does not have")
        settings.processed_data_dir = records_dir

    # --- payload round-trip through a throwaway collection -----------------
    if not args.skip_qdrant:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (Distance, PointStruct, VectorParams)

        record = json.loads(paths[0].read_text())
        chunks = PaperChunker().chunk_record(record)
        caption = next((c for c in chunks
                        if c.chunk_type.value == "caption" and c.figure_bbox), None)
        body = next((c for c in chunks if c.figure_refs), None)
        if caption is None:
            print("\n(no caption chunk with figure coordinates to round-trip)")
        else:
            client_q = QdrantClient(url=args.qdrant)
            existing = {c.name for c in client_q.get_collections().collections}
            assert THROWAWAY not in existing, f"{THROWAWAY} already exists"
            assert "research_papers" in existing, "expected the live collection to exist"
            client_q.create_collection(
                THROWAWAY,
                vectors_config=VectorParams(size=4, distance=Distance.COSINE),
            )
            try:
                points = [PointStruct(id=str(uuid.uuid4()), vector=[0.1, 0.2, 0.3, 0.4],
                                      payload=c.to_payload())
                          for c in (caption, body) if c is not None]
                client_q.upsert(THROWAWAY, points=points)
                got = client_q.retrieve(THROWAWAY, ids=[p.id for p in points],
                                        with_payload=True)
                print(f"\nthrowaway collection {THROWAWAY!r}: "
                      f"wrote {len(points)}, read {len(got)}")
                for point in got:
                    payload = point.payload
                    keys = ("chunk_type", "figure_id", "figure_kind", "figure_label",
                            "figure_page", "figure_bbox", "figure_image",
                            "has_figure_image", "page_start", "page_end",
                            "char_start", "char_end")
                    print("  " + json.dumps({k: payload.get(k) for k in keys}))
                    if payload.get("figure_refs"):
                        print(f"    figure_refs: "
                              f"{json.dumps(payload['figure_refs'][:3])}")
                    if payload["chunk_type"] == "caption":
                        src = caption
                        for field in ("figure_id", "figure_page", "figure_bbox",
                                      "figure_image", "figure_label"):
                            if payload.get(field) != getattr(src, field):
                                failures.append(
                                    f"payload {field} changed in Qdrant: "
                                    f"{payload.get(field)!r} != {getattr(src, field)!r}")
                        # `text` must still be the verbatim slice after the trip
                        doc = pr.full_text(record)
                        if doc[payload["char_start"]:payload["char_end"]] != payload["text"]:
                            failures.append("round-tripped text is not the record slice")
                        else:
                            print("    full_text(record)[char_start:char_end] == text  OK")
            finally:
                client_q.delete_collection(THROWAWAY)
                after = {c.name for c in client_q.get_collections().collections}
                print(f"  deleted; research_papers intact: "
                      f"{'research_papers' in after}, "
                      f"throwaway gone: {THROWAWAY not in after}")

    print("\n=== " + ("ALL CHECKS PASSED" if not failures
                      else f"{len(failures)} FAILURES") + " ===")
    for failure in failures:
        print("  " + failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
