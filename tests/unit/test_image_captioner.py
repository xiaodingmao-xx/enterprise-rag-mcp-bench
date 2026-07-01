from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image

from src.core.types import Chunk
from src.ingestion.transform.image_captioner import ImageCaptioner


def _settings():
    return SimpleNamespace(vision_llm=SimpleNamespace(enabled=True))


def test_image_captioner_skips_tiny_extracted_images():
    llm = MagicMock()
    captioner = ImageCaptioner(settings=_settings(), llm=llm)
    chunk = Chunk(
        id="chunk-1",
        text="Thin line image: [IMAGE: thin_line]",
        metadata={
            "source_path": "doc.pdf",
            "images": [
                {
                    "id": "thin_line",
                    "path": "missing-tiny-image.jpeg",
                    "position": {"width": 1854, "height": 2},
                }
            ],
        },
    )

    result = captioner.transform([chunk])

    llm.chat_with_image.assert_not_called()
    assert result[0].text == "Thin line image: [IMAGE: thin_line]"
    assert "image_captions" not in result[0].metadata


def test_image_captioner_captions_normal_sized_images(tmp_path):
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (64, 48), color="white").save(image_path)

    llm = MagicMock()
    llm.chat_with_image.return_value = SimpleNamespace(content="A small chart.")
    captioner = ImageCaptioner(settings=_settings(), llm=llm)
    chunk = Chunk(
        id="chunk-1",
        text="Chart: [IMAGE: chart_1]",
        metadata={
            "source_path": "doc.pdf",
            "images": [
                {
                    "id": "chart_1",
                    "path": str(image_path),
                    "position": {"width": 64, "height": 48},
                }
            ],
        },
    )

    result = captioner.transform([chunk])

    llm.chat_with_image.assert_called_once()
    assert "(Description: A small chart.)" in result[0].text
    assert result[0].metadata["image_captions"] == [
        {"id": "chart_1", "caption": "A small chart."}
    ]
