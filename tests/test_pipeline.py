import json
import tempfile
import unittest
from pathlib import Path

import torch

from pathai.data import prefix, prepare
from pathai.encode_reports import contiguous_chunks
from pathai.model import Multimodal, report_loss
from pathai.gpt import ToyAesclea, ToyConfig
from pathai.report_quality import normalize_text, quality


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def test_original_gpt_and_transfer(self):
        original = ToyAesclea(ToyConfig(vocab_size=20))
        model = Multimodal(48, original.state_dict())
        self.assertIsInstance(model.gpt, ToyAesclea)
        torch.testing.assert_close(
            model.gpt.tok_emb.weight[:20], original.tok_emb.weight
        )
        torch.testing.assert_close(
            model.gpt.blocks[0].attn.qkv.weight, original.blocks[0].attn.qkv.weight
        )

    def test_modalities_and_backward(self):
        model = Multimodal(48, ToyAesclea(ToyConfig(vocab_size=20)).state_dict())
        t = torch.tensor([[1, 2, 3], [2, 4, 0]])
        image = torch.randn(2, 768)
        for mode in ["events", "image", "both"]:
            prediction, report = model(
                t if mode != "image" else None,
                image if mode != "events" else None,
                mode,
            )
            self.assertEqual(prediction.shape, (2,))
            self.assertEqual(report.shape, (2, 512))
        prediction, report = model(t, image, "both")
        loss = prediction.square().mean() + report_loss(
            report, torch.randn(2, 512), torch.tensor([True, False])
        )
        loss.backward()
        self.assertGreater(model.gpt.tok_emb.weight.grad.abs().sum(), 0)
        self.assertGreater(model.image_projection.weight.grad.abs().sum(), 0)
        model.zero_grad()
        model.event_loss(t, image).backward()
        self.assertGreater(model.image_projection.weight.grad.abs().sum(), 0)
        self.assertGreater(model.gpt.blocks[0].attn.qkv.weight.grad.abs().sum(), 0)
        with self.assertRaises(ValueError):
            model(None, None, "both")
        torch.testing.assert_close(
            model(t, image, "events")[0],
            model(torch.nn.functional.pad(t, (0, 2)), image, "events")[0],
        )

    def test_prepare_with_explicit_data_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "input"
            for source, filename, pid, events in [
                ("mayo", "train.jsonl", "train", ["DIAG_C34", "DEATH"]),
                ("mayo", "val.jsonl", "val", ["DIAG_C34", "DEATH"]),
                (
                    "tcga",
                    "tcga.jsonl",
                    "patient",
                    ["DIAG_C34", "PROC_RESECTION", "DEATH"],
                ),
            ]:
                (data / source).mkdir(parents=True, exist_ok=True)
                (data / source / filename).write_text(
                    json.dumps({"patient_id": pid, "events": events}) + "\n"
                )
            (data / "tcga/reports").mkdir()
            (data / "tcga/slides").mkdir()
            report = data / "tcga/reports/patient.txt"
            original = "Lung\tcarcinoma specimen diagnosis. " * 8
            report.write_text(original)
            (data / "tcga/slides/patient.svs").touch()
            rows, vocab = prepare(root / "processed", data)
            patient = next(row for row in rows if row["source"] == "tcga")
            self.assertEqual(patient["input_events"], ["DIAG_C34", "PROC_RESECTION"])
            self.assertEqual(patient["label"], 1)
            self.assertTrue(patient["report_trainable"])
            self.assertIn("PROC_RESECTION", vocab)
            self.assertEqual(report.read_text(), original)
            self.assertNotIn("\t", Path(patient["clean_report_path"]).read_text())

    def test_input_and_reports(self):
        self.assertEqual(
            prefix(["STAGE_T2", "PROC_RESECTION", "DEATH"]),
            ["STAGE_T2", "PROC_RESECTION"],
        )
        with self.assertRaises(ValueError):
            prefix(["DEATH", "PROC_RESECTION"])
        self.assertEqual(normalize_text(" abc\t def\r\n"), "abc def\n")
        self.assertEqual(quality("?? 1 x 2")["status"], "review_required")
        ids = list(range(217))
        self.assertEqual(sum(contiguous_chunks(ids), []), ids)


if __name__ == "__main__":
    unittest.main()
