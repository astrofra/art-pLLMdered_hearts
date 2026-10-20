import subprocess
import json
from contextlib import ExitStack
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import faketerm
import pexpect


class GameOutputTests(unittest.TestCase):
    def test_waits_for_full_prompt_and_acknowledges_split_markers(self):
        class Game:
            chunks = iter([
                "", "\n>SHOOT THE PIRATE\nOpening\n[Press RETURN or ENTER to be",
                "gin.]", "", "Story\n***MO", "RE***", "Cabin\n", ">",
            ])
            replies = []

            def read_nonblocking(self, **kwargs):
                return next(self.chunks)

            def sendline(self, text):
                self.replies.append(text)

        game = Game()
        output = faketerm.read_game_output(game)
        self.assertIn("Story", output)
        self.assertTrue(output.endswith(">"))
        self.assertNotIn("***MORE***", output)
        self.assertEqual(game.replies, ["", ""])

    def test_dead_game_does_not_spin_forever(self):
        class Game:
            def read_nonblocking(self, **kwargs):
                raise pexpect.EOF("Stopped")

        with self.assertRaisesRegex(RuntimeError, "Frotz stopped"):
            faketerm.read_game_output(Game())

    def test_terminal_escapes_are_removed_and_story_is_preserved(self):
        output = faketerm.clean_output(
            "\x1b]0;Title\x07\x1b[?25l\x1b[31mA pirate appears.\x1b[0m\x00\n>"
        )
        self.assertEqual(output, "A pirate appears.\n>")

    def test_shutdown_waits_for_child_process(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        faketerm._stop_process(process)
        self.assertIsNotNone(process.poll())
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)


class VideoSelectionTests(unittest.TestCase):
    def setUp(self):
        self.log_patch = patch.object(faketerm, "VIDEO_MATCH_LOG_PATH", None)
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)

    def test_passages_use_maximum_and_keep_clip_recency(self):
        catalog = [
            {"filename": "first.ogv", "embedding": [0.0, 1.0], "norm": 1.0,
             "passages": [{"id": "match", "text": "A specific recollection", "embedding": [1.0, 0.0], "norm": 1.0},
                          {"id": "other", "embedding": [-1.0, 0.0], "norm": 1.0}]},
            {"filename": "second.ogv", "embedding": [0.8, 0.6], "norm": 1.0},
        ]
        with patch.object(faketerm, "VIDEO_MATCH_MODE", "passages"):
            entry = faketerm.select_best_video([1.0, 0.0], 1.0, catalog, [], None)
            self.assertEqual(entry["filename"], "first.ogv")
            self.assertEqual(entry["match_score"], 1.0)
            self.assertEqual(entry["matched_passage"]["id"], "match")
            self.assertNotIn("embedding", entry["matched_passage"])
            self.assertNotIn("match_score", catalog[0])
            entry = faketerm.select_best_video([1.0, 0.0], 1.0, catalog, ["first.ogv"], None)
            self.assertEqual(entry["filename"], "second.ogv")
        with patch.object(faketerm, "VIDEO_MATCH_MODE", "clip"):
            self.assertEqual(faketerm.select_best_video([1.0, 0.0], 1.0, catalog, [], None)["filename"], "second.ogv")

    def test_failed_embedding_preserves_recency(self):
        recent = ["first.ogv"]
        self.assertIsNone(faketerm.select_best_video(None, 0, [{"filename": "first.ogv"}], recent, None))
        self.assertEqual(recent, ["first.ogv"])

    def test_loader_keeps_one_entry_per_clip_and_ignores_invalid_passages(self):
        data = [{"filename": "clip.ogv", "embedding": [1, 0], "duration_sec": 30,
                 "passages": [{"embedding": [0, 1], "text": "Valid"}, {"embedding": [1]},
                              {"embedding": [float("nan"), 0]}, {"embedding": ["bad", 0]}]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"catalogue.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            catalog = faketerm.load_video_embeddings(str(path))
        self.assertEqual(len(catalog), 1)
        self.assertEqual(len(catalog[0]["passages"]), 1)
        self.assertEqual(catalog[0]["duration_sec"], 30)

    def test_match_log_records_only_successfully_requested_video(self):
        entry = {"filename": "clip.ogv", "duration_sec": 30, "match_score": 0.4,
                 "matched_commentary": "A game observation.",
                 "matched_passage": {"text": "A contextual recollection.", "start": "0:00:12.000", "end": "0:00:42.000"}}
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/"matches.jsonl"
            with patch.object(faketerm, "VIDEO_MATCH_LOG_PATH", str(log)):
                with patch.object(faketerm, "write_llm_video_request", return_value=False):
                    faketerm.maybe_emit_video_request(None, entry, [], None, 0, 25)
                self.assertFalse(log.exists())
                with patch.object(faketerm, "write_llm_video_request", return_value=True):
                    faketerm.maybe_emit_video_request(None, entry, [], None, 0, 25)
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["commentary"], "A game observation.")
        self.assertEqual(rows[0]["passage"], entry["matched_passage"])

    def test_equal_scores_are_stable_and_skip_recent_clips(self):
        catalog = [
            {"filename": name, "embedding": [1.0, 0.0], "norm": 1.0}
            for name in ("first.ogv", "second.ogv", "third.ogv")
        ]
        entry = faketerm.select_best_video([1.0, 0.0], 1.0, catalog, ["first.ogv"], "third.ogv")
        self.assertEqual(entry["filename"], "second.ogv")

    def test_different_embedding_dimensions_do_not_select_arbitrary_clip(self):
        catalog = [{"filename": "clip.ogv", "embedding": [1.0, 0.0], "norm": 1.0}]
        self.assertIsNone(faketerm.select_best_video([1.0], 1.0, catalog, [], None))

    def test_cooldown_keeps_request_pending(self):
        entry = {"filename": "clip.ogv", "duration_sec": 30.0}
        with patch.object(faketerm, "write_llm_video_request") as write:
            result = faketerm.maybe_emit_video_request(None, entry, [], None, time.monotonic() + 60, 25)
        self.assertEqual(result[0], entry)
        write.assert_not_called()

    def test_request_is_complete_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(faketerm, "LLM_OUT_DIR", directory):
                self.assertTrue(faketerm.write_llm_video_request("clip.ogv"))
            requests = list(Path(directory).iterdir())
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0].suffix, ".txt")
            self.assertEqual(requests[0].read_text(encoding="utf-8"), "clip.ogv\n")

    def test_write_failure_does_not_start_cooldown(self):
        entry = {"filename": "clip.ogv", "duration_sec": 30.0}
        recent = []
        with patch.object(faketerm, "write_llm_video_request", return_value=False):
            result = faketerm.maybe_emit_video_request(None, entry, recent, None, 0.0, 25)
        self.assertEqual(result, (entry, 0.0, None))
        self.assertEqual(recent, [])


class OllamaTests(unittest.TestCase):
    def test_commentary_can_disable_thinking(self):
        response = SimpleNamespace(message=SimpleNamespace(content="A complete comment."))
        with patch.object(faketerm, "LLM_THINK", "0"):
            with patch.object(faketerm, "_ollama_request", return_value=response) as request:
                self.assertEqual(faketerm.generate_commentary("A game scene"), "A complete comment.")
        self.assertIs(request.call_args.kwargs["think"], False)

    def test_missing_thinking_setting_uses_model_default(self):
        response = SimpleNamespace(message=SimpleNamespace(content="A complete comment."))
        with patch.object(faketerm, "LLM_THINK", None):
            with patch.object(faketerm, "_ollama_request", return_value=response) as request:
                faketerm.generate_commentary("A game scene")
        self.assertNotIn("think", request.call_args.kwargs)

    def test_embeddings_use_configured_gpu_placement(self):
        with patch.object(faketerm, "_ollama_request", return_value={"embeddings": [[1.0, 0.0]]}) as request:
            self.assertEqual(faketerm.embed_commentary_text("A game scene"), ([1.0, 0.0], 1.0))
        self.assertEqual(request.call_args.args[0], "embed")
        self.assertIs(request.call_args.kwargs["truncate"], False)
        self.assertEqual(request.call_args.kwargs["options"], {"num_gpu": faketerm.VIDEO_EMBED_NUM_GPU, "num_ctx": 2048})

    def test_slow_request_times_out(self):
        class Client:
            def __init__(self, **kwargs):
                pass

            def list(self):
                time.sleep(0.2)

        with patch.object(faketerm.ollama, "Client", Client):
            with self.assertRaisesRegex(TimeoutError, "exceeded"):
                faketerm._ollama_request("list", 0.02)

    def test_failed_commentary_allows_game_to_continue(self):
        with patch.object(faketerm, "_ollama_request", side_effect=ConnectionError("Offline")):
            self.assertEqual(faketerm.generate_commentary("A game scene"), "")


class CommentaryTimingTests(unittest.TestCase):
    def run_game(self, outputs, comments, commands, steps=None, once=True, minimum=3, embedding_results=None,
                 request_results=None):
        now = [0.0]
        events = []
        game = SimpleNamespace(proc=object(), sendline=Mock())
        game.sendline.side_effect = lambda command: events.append(("command", now[0], command))
        timings = iter(outputs)

        def display(unused):
            now[0], text = next(timings)
            events.append(("display", text))
            return text

        def comment(history, recent_comments=None):
            events.append(("comment", now[0], history, tuple(recent_comments or ())))
            return next(comments)

        def embed(text):
            events.append(("embed", now[0], text))
            return next(embedding_results) if embedding_results is not None else ([1.0], 1.0)

        def write(filename):
            events.append(("request", now[0], filename))
            return next(request_results) if request_results is not None else True

        entry = {"filename": "clip.ogv", "duration_sec": 5.0}
        values = {
            "ENABLE_LLM": True, "ENABLE_RAW_OUTPUT": False, "renderer": None,
            "COMMENTARY_MIN_WORDS": minimum,
            "VIDEO_END_MARGIN_SEC": 1.0, "plundered_hearts_commands": commands,
            "VIDEO_MATCH_LOG_PATH": None,
            "time": SimpleNamespace(monotonic=lambda: now[0]),
        }
        with ExitStack() as stack:
            stack.enter_context(patch.multiple(faketerm, **values))
            stack.enter_context(patch("builtins.print"))
            for name, value in [
                ("_start_game_process", Mock(return_value=game)),
                ("_stop_process", Mock()), ("read_game_output", Mock(return_value="raw")),
                ("display_game_output", display), ("type_to_renderer", Mock()),
                ("comment_on_game", comment),
                ("load_video_embeddings", Mock(return_value=[entry])),
                ("embed_commentary_text", embed),
                ("select_best_video", lambda vector, *args: entry if vector is not None else None),
                ("write_llm_video_request", write),
            ]:
                stack.enter_context(patch.object(faketerm, name, value))
            faketerm.play(steps=steps, once=once)
        return events

    def test_comments_during_video_but_waits_for_fresh_comment_to_start_next(self):
        events = self.run_game(
            [(0, "opening four whole words"), (1, "short"),
             (2, "second complete output"), (3, "third complete output"), (6, "done"), (7, "ending")],
            iter(["First comment.", "Second comment.", "During the video.", "Fresh handover."]),
            ["one", "two", "three", "four", "five"])
        comments = [event for event in events if event[0] == "comment"]
        self.assertEqual([event[1] for event in comments], [0, 2, 3, 6])
        self.assertEqual([event for event in events if event[0] == "embed"],
                         [("embed", 0, "First comment."), ("embed", 6, "Fresh handover.")])
        history = comments[1][2]
        for text in [">> ONE", "short", ">> TWO", "second complete output"]:
            self.assertIn(text, history)
        self.assertNotIn("opening four whole words", history)
        self.assertNotIn(">> THREE", history)
        self.assertLess(events.index(("display", "second complete output")), events.index(comments[1]))

    def test_empty_comment_keeps_the_uncommented_history(self):
        events = self.run_game(
            [(0, "opening four whole words"), (1, "second complete output"), (2, "ending")],
            iter(["", "A real comment."]), ["one", "two"])
        comments = [event for event in events if event[0] == "comment"]
        self.assertIn("opening four whole words", comments[1][2])
        self.assertIn(">> ONE", comments[1][2])
        self.assertIn("second complete output", comments[1][2])

    def test_video_timer_and_history_survive_walkthrough_restarts(self):
        events = self.run_game(
            [(0, "first opening four words"), (1, "first ending text"),
             (2, "second opening text"), (3, "second ending text"),
             (6, "third opening text"), (7, "last ending text")],
            iter(["First comment.", "Second comment.", "Last comment."]), ["one"], steps=3, once=False)
        comments = [event for event in events if event[0] == "comment"]
        self.assertEqual([event[1] for event in comments], [0, 2, 6])
        self.assertEqual([event[3] for event in comments],
                         [(), ("First comment.",), ("First comment.", "Second comment.")])
        self.assertEqual([event for event in events if event[0] == "embed"],
                         [("embed", 0, "First comment."), ("embed", 6, "Last comment.")])
        for text in ["first ending text", "second opening text", "[The game has restarted from the beginning.]"]:
            self.assertIn(text, comments[1][2])
        self.assertIn("second ending text", comments[2][2])

    def test_failed_embedding_requires_another_visible_comment_at_next_prompt(self):
        events = self.run_game(
            [(0, "opening four whole words"), (1, "short"), (2, "ending")],
            iter(["A comment.", "A fresh comment."]), ["one", "two"], embedding_results=iter([(None, 0.0), ([1.0], 1.0)]))
        comments = [event for event in events if event[0] == "comment"]
        self.assertEqual(len(comments), 2)
        self.assertEqual([event for event in events if event[0] == "embed"],
                         [("embed", 0, "A comment."), ("embed", 1, "A fresh comment.")])

    def test_waits_for_150_words_and_a_complete_output(self):
        events = self.run_game(
            [(0, "opening"), (1, "word " * 148), (2, "the next output finishes here"), (3, "ending")],
            iter(["An opening comment.", "A comment during video."]), ["one", "two", "three"], minimum=150)
        comments = [event for event in events if event[0] == "comment"]
        self.assertEqual([event[1] for event in comments], [0, 2])
        self.assertTrue(comments[1][2].endswith("the next output finishes here"))

    def test_idle_video_forces_fresh_comment_below_word_threshold(self):
        events = self.run_game(
            [(0, "opening"), (1, "short"), (6, "the paragraph is complete"), (7, "ending")],
            iter(["First.", "Fresh after video."]), ["one", "two", "three"], minimum=150)
        comments = [event for event in events if event[0] == "comment"]
        self.assertEqual([event[1] for event in comments], [0, 6])
        self.assertLess(len(comments[1][2].split()), 150)
        self.assertEqual([event for event in events if event[0] == "embed"],
                         [("embed", 0, "First."), ("embed", 6, "Fresh after video.")])
        self.assertLess(events.index(("display", "the paragraph is complete")), events.index(comments[1]))
        self.assertLess(events.index(comments[1]), events.index(("embed", 6, "Fresh after video.")))
        self.assertLess(events.index(("embed", 6, "Fresh after video.")), events.index(("request", 6, "clip.ogv")))
        self.assertLess(events.index(("request", 6, "clip.ogv")), events.index(("command", 6, "THREE")))

    def test_failed_request_never_reuses_old_comment_after_more_game_output(self):
        events = self.run_game(
            [(0, "opening"), (1, "next paragraph"), (2, "ending")],
            iter(["First attempt.", "Fresh attempt."]), ["one", "two"], minimum=150,
            request_results=iter([False, True]))
        self.assertEqual([event for event in events if event[0] == "embed"],
                         [("embed", 0, "First attempt."), ("embed", 1, "Fresh attempt.")])

    def test_empty_idle_response_never_uses_a_comment_from_the_video(self):
        events = self.run_game(
            [(0, "opening four whole words"), (1, "during video output"),
             (6, "video just finished"), (7, "new completed paragraph"), (8, "ending")],
            iter(["Opening.", "During.", "", "Fresh now."]), ["one", "two", "three", "four"])
        self.assertEqual([event for event in events if event[0] == "embed"],
                         [("embed", 0, "Opening."), ("embed", 7, "Fresh now.")])
        comments = [event for event in events if event[0] == "comment"]
        self.assertIn("video just finished", comments[-1][2])
        self.assertIn("new completed paragraph", comments[-1][2])

    def test_remembers_only_the_last_two_successful_comments(self):
        events = self.run_game(
            [(index, "a complete game output") for index in range(6)],
            iter(["First.", "Second.", "", "Third.", "Fourth."]),
            ["one", "two", "three", "four", "five"])
        comments = [event for event in events if event[0] == "comment"]
        self.assertEqual([event[3] for event in comments],
                         [(), ("First.",), ("First.", "Second."),
                          ("First.", "Second."), ("Second.", "Third.")])


if __name__ == "__main__":
    unittest.main()
