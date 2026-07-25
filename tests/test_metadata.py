import json

from recsys.metadata import load_metadata, metadata_coverage, public_item_record


def test_metadata_is_filtered_and_builds_content(tmp_path) -> None:
    path = tmp_path / "metadata.jsonl"
    rows = [
        {
            "parent_asin": "a",
            "title": "Space Quest",
            "categories": ["PC", "Adventure"],
            "features": ["Story rich"],
            "images": [{"large": "https://example.com/a.jpg"}],
        },
        {"parent_asin": "b", "title": "Ignore Me"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))

    metadata = load_metadata(path, {"a"})

    assert metadata["parent_asin"].tolist() == ["a"]
    assert "Space Quest" in metadata.iloc[0]["content_text"]
    assert "Adventure" in metadata.iloc[0]["content_text"]
    assert metadata_coverage(metadata, {"a", "missing"}) == 0.5
    record = public_item_record(metadata.iloc[0].to_dict())
    assert record["image_url"] == "https://example.com/a.jpg"
