import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

SCRIPT = Path(__file__).resolve().parents[1] / "documentation/presentation-update-2026-10-19/record_session.py"
SPEC = importlib.util.spec_from_file_location("record_session", SCRIPT)
record_session = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(record_session)


class SessionRecordingTests(unittest.TestCase):
    def test_pagination_is_not_counted_as_an_executed_game_command(self):
        sent = Mock()
        game = SimpleNamespace(sendline=sent)
        runtime = SimpleNamespace(_start_game_process=Mock(return_value=game))
        for name in ["read_game_output", "display_game_output", "comment_on_game", "_ollama_request",
                     "select_best_video", "maybe_emit_video_request"]:
            setattr(runtime, name, Mock())
        with tempfile.TemporaryDirectory() as temporary:
            recorder = record_session.Recorder(runtime, temporary)
            recorder.install()
            try:
                returned = runtime._start_game_process()
                returned.sendline("")
                returned.sendline("STAND UP")
                self.assertEqual(recorder.command_count, 1)
            finally:
                recorder.close()
            rows = [json.loads(x) for x in (Path(temporary) / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([x["event"] for x in rows], ["game_start", "pager_ack", "command"])
            self.assertEqual([call.args for call in sent.call_args_list], [("",), ("STAND UP",)])

    def test_response_keeps_visible_text_and_metrics_without_internal_reasoning(self):
        response = Mock()
        response.model_dump.return_value = {"message": {"content": "Visible.", "thinking": "Private analysis"},
                                           "eval_count": 42}
        data = record_session.json_response(response)
        self.assertEqual(data["message"], {"content": "Visible.", "thinking_characters": 16})
        self.assertEqual(data["eval_count"], 42)
        self.assertEqual(response.model_dump.return_value["message"]["thinking"], "Private analysis")

    def test_ranking_records_recency_before_original_selection_mutates_it(self):
        selected = {"filename": "second.ogv", "embedding": [1.0], "norm": 1.0}
        catalog = [{"filename": "first.ogv", "embedding": [1.0], "norm": 1.0}, selected]
        runtime = SimpleNamespace()
        names = ["_start_game_process", "read_game_output", "display_game_output", "comment_on_game",
                 "_ollama_request", "maybe_emit_video_request"]
        for name in names:
            setattr(runtime, name, Mock())
        def select(vector, norm, entries, recent, last):
            recent.clear()
            return selected
        runtime.select_best_video = select
        runtime._cosine_similarity = lambda *unused: 1.0
        with tempfile.TemporaryDirectory() as temporary:
            recorder = record_session.Recorder(runtime, temporary)
            recorder.install()
            recent = ["first.ogv"]
            try:
                self.assertIs(runtime.select_best_video([1.0], 1.0, catalog, recent, "first.ogv"), selected)
                self.assertEqual(recent, [])
            finally:
                recorder.close()
            event = json.loads((Path(temporary) / "events.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(event["recent_before"], ["first.ogv"])
            self.assertEqual(event["recent_after"], [])
            self.assertTrue(event["ranking"][0]["is_recent"])
            self.assertEqual(event["selected"], "second.ogv")
            self.assertIs(runtime.select_best_video, select)


if __name__ == "__main__":
    unittest.main()
