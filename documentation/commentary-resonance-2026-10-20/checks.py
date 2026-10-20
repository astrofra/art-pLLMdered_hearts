"""Check source coverage, timestamp provenance and grouped maximum matching offline."""

import json
import math
import re
import unittest

import analyse
import experiment


def seconds(value):
    hours,minutes,rest=value.replace(",",".").split(":")
    return 3600*int(hours)+60*int(minutes)+float(rest)


class PassageChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.videos,_=experiment.source_data()
        cls.cache=experiment.Cache()

    def test_sentence_windows_cover_every_transcript_character(self):
        for video in self.videos:
            source=" ".join(c["text"] for c in experiment.subtitle_cues(video))
            coverage=[False]*len(source);previous=0
            for row in experiment.sentence_windows(video):
                begin=source.find(row["text"],previous)
                self.assertGreaterEqual(begin,0,video["filename"])
                end=begin+len(row["text"]);coverage[begin:end]=[True]*(end-begin);previous=begin
                self.assertLess(seconds(row["start"]),seconds(row["end"]))
            self.assertTrue(all(covered or char.isspace() for covered,char in zip(coverage,source)),video["filename"])

    def test_all_indexes_keep_all_25_clip_groups_and_valid_vectors(self):
        data=json.loads((experiment.HERE/"granularity-inputs.json").read_text(encoding="utf-8"))
        expected={v["filename"] for v in self.videos}
        for kind in {d["kind"] for d in data["documents"]}:
            self.assertEqual({d["filename"] for d in data["documents"] if d["kind"]==kind},expected)
        for doc in data["documents"]:
            vector=self.cache.data["records"][doc["embeddings"]["raw"]]["embedding"]
            self.assertEqual(len(vector),768)
            self.assertTrue(all(math.isfinite(x) for x in vector))
            self.assertAlmostEqual(math.sqrt(sum(x*x for x in vector)),1,places=5)

    def test_one_best_passage_wins_without_average_or_duplicate_clips(self):
        class Cache:
            data={"records":{"a":{"embedding":[1,0]},"b":{"embedding":[-1,0]},"c":{"embedding":[0.8,0.6]}}}
        docs=[{"filename":clip,"title":clip,"id":key,"start":"0:00:00.000","end":"0:00:01.000",
               "text":key,"embeddings":{"raw":key}} for clip,key in [("A","a"),("A","b"),("B","c")]]
        result=analyse.ranks(analyse.unit([1,0]),docs,"raw",Cache())
        self.assertEqual([r["filename"] for r in result],["A","B"])
        self.assertEqual(result[0]["unit_id"],"a")
        self.assertAlmostEqual(result[0]["score"],1)

    def test_paired_screen_is_complete_and_synthetic_control_is_labelled(self):
        screen=json.loads((experiment.HERE/"screen.json").read_text(encoding="utf-8"))["results"]
        self.assertEqual(len(screen),120)
        for condition in {r["condition"] for r in screen}:
            actual={(r["scene"],r["seed"]) for r in screen if r["condition"]==condition}
            self.assertEqual(actual,{(n,s) for n in experiment.SCENES for s in [42,43]})
        for row in screen:
            self.assertEqual(row["done_reason"],"stop")
            self.assertTrue(row["commentary"].strip())
            self.assertLess(row["prompt_tokens"]+row["output_tokens"],4096)
            self.assertNotRegex(row["history"],r"\x1b")
        inputs=json.loads((experiment.HERE/"retrieval-inputs.json").read_text(encoding="utf-8"))
        vague=next(q for q in inputs["queries"] if q["id"]=="vague_reason")
        self.assertIn("synthetic",vague["control_type"])


if __name__=="__main__":unittest.main()
