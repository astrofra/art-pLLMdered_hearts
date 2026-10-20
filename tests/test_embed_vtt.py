import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import embed_vtt
import faketerm


class ContextualPassageTests(unittest.TestCase):
    def test_short_tail_is_joined_without_repeating_overlap(self):
        cues = [{"start": "0:00:00.000", "end": "0:00:10.000", "text": "first "*25},
                {"start": "0:00:10.000", "end": "0:00:15.000", "text": "second "*10},
                {"start": "0:00:15.000", "end": "0:00:20.000", "text": "last "*5}]
        passages = embed_vtt.contextual_passages(cues, 30)
        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0]["text"].split(), ("first "*25+"second "*10+"last "*5).split())
        self.assertEqual(passages[0]["start"], "0:00:00.000")
        self.assertEqual(passages[0]["end"], "0:00:20.000")

    def test_production_catalogue_retains_source_text_and_video_metadata(self):
        legacy = json.loads((ROOT / "assets/abriggs-itw-embeddings.json").read_text(encoding="utf-8"))
        grouped = json.loads((ROOT / "assets/abriggs-itw-passages.json").read_text(encoding="utf-8"))
        self.assertEqual(len(grouped), 25)
        self.assertEqual(sum(len(item["passages"]) for item in grouped), 132)
        for old, new in zip(legacy, grouped):
            for key in ["filename", "sequence_title", "duration_sec"]:
                self.assertEqual(old[key], new[key])
            path = ROOT / "godot-viewer/video" / (Path(new["filename"]).stem+".txt")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), new["subtitle_sha256"])
            cues = embed_vtt.subtitle_cues(path)
            source = " ".join(c["text"] for c in cues)
            coverage = [False]*len(source)
            for passage in new["passages"]:
                begin = source.find(passage["text"])
                self.assertGreaterEqual(begin, 0)
                coverage[begin:begin+len(passage["text"])]=[True]*len(passage["text"])
                self.assertGreaterEqual(len(passage["text"].split()), 30)
                self.assertEqual(len(passage["embedding"]), 768)
            self.assertTrue(all(covered or char.isspace() for covered, char in zip(coverage,source)), new["filename"])
        self.assertEqual(len(faketerm.load_video_embeddings(str(ROOT / "assets/abriggs-itw-passages.json"))), 25)


if __name__ == "__main__":
    unittest.main()
