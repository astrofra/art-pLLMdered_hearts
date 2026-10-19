import subprocess
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
        with patch.object(faketerm, "_ollama_request", return_value={"embedding": [1.0, 0.0]}) as request:
            self.assertEqual(faketerm.embed_commentary_text("A game scene"), ([1.0, 0.0], 1.0))
        self.assertEqual(request.call_args.kwargs["options"], {"num_gpu": faketerm.VIDEO_EMBED_NUM_GPU})

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
    def run_game(self, outputs, comments, commands, steps=None, once=True, minimum=3, embedding_results=None):
        now = [0.0]
        events = []
        game = SimpleNamespace(proc=object(), sendline=Mock())
        timings = iter(outputs)

        def display(unused):
            now[0], text = next(timings)
            events.append(("display", text))
            return text

        def comment(history, command):
            events.append(("comment", now[0], history, command))
            return next(comments)

        def embed(text):
            events.append(("embed", now[0], text))
            return next(embedding_results) if embedding_results is not None else ([1.0], 1.0)

        entry = {"filename": "clip.ogv", "duration_sec": 5.0}
        values = {
            "ENABLE_LLM": True, "ENABLE_RAW_OUTPUT": False, "renderer": None,
            "COMMENTARY_MIN_WORDS": minimum,
            "VIDEO_END_MARGIN_SEC": 1.0, "plundered_hearts_commands": commands,
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
                ("write_llm_video_request", Mock(return_value=True)),
            ]:
                stack.enter_context(patch.object(faketerm, name, value))
            faketerm.play(steps=steps, once=once)
        return events

    def test_comments_during_video_but_embeds_only_the_latest_after_it_finishes(self):
        events = self.run_game(
            [(0, "opening four whole words"), (1, "short"),
             (2, "second complete output"), (3, "third complete output"), (6, "done"), (7, "ending")],
            iter(["First comment.", "Second comment.", "Latest comment."]),
            ["one", "two", "three", "four", "five"])
        comments = [event for event in events if event[0] == "comment"]
        self.assertEqual([event[1] for event in comments], [0, 2, 3])
        self.assertEqual([event for event in events if event[0] == "embed"],
                         [("embed", 0, "First comment."), ("embed", 6, "Latest comment.")])
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
        self.assertEqual([event for event in events if event[0] == "embed"],
                         [("embed", 0, "First comment."), ("embed", 6, "Last comment.")])
        for text in ["first ending text", "second opening text", "[The game has restarted from the beginning.]"]:
            self.assertIn(text, comments[1][2])
        self.assertIn("second ending text", comments[2][2])

    def test_failed_embedding_retries_the_pending_comment_without_another_llm_call(self):
        events = self.run_game(
            [(0, "opening four whole words"), (1, "short"), (2, "ending")],
            iter(["A comment."]), ["one", "two"], embedding_results=iter([(None, 0.0), ([1.0], 1.0)]))
        comments = [event for event in events if event[0] == "comment"]
        self.assertEqual(len(comments), 1)
        self.assertEqual([event for event in events if event[0] == "embed"],
                         [("embed", 0, "A comment."), ("embed", 1, "A comment.")])

    def test_waits_for_150_words_and_a_complete_output(self):
        events = self.run_game(
            [(0, "word " * 150), (1, "the next output finishes here"), (2, "ending")],
            iter(["A comment."]), ["one", "two"], minimum=150)
        comments = [event for event in events if event[0] == "comment"]
        self.assertEqual(len(comments), 1)
        self.assertEqual(comments[0][1], 1)
        self.assertTrue(comments[0][2].endswith("the next output finishes here"))


if __name__ == "__main__":
    unittest.main()
