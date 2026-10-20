extends Control

@onready var video_player: VideoStreamPlayer = $Margin/VideoFrame/VideoPlayer
@onready var video_frame: Control = $Margin/VideoFrame
@onready var margin_container: MarginContainer = $Margin
@onready var subtitle_label: Label = $Subtitles/SubtitlePanel/SubtitleLabel
@onready var subtitle_shadow_label: Label = $Subtitles/SubtitlePanel/SubtitleShadowLabel
@onready var subtitle_panel: Control = $Subtitles/SubtitlePanel

const VIDEO_FOLDER_PATH = "res://video"
const SUBTITLE_EXTENSIONS = ["txt"]
const USE_FRENCH_SUBTITLES = false
const PREFILL_VIDEO_QUEUE = false
const LLM_OUT_RELATIVE_PATH = "../llm_out"
const LLM_OUT_OVERRIDE = ""
const LLM_POLL_INTERVAL = 0.5
const SUBTITLE_FONT_PATH = "res://fonts/RobotoCondensed-Regular.ttf"
const SUBTITLE_FONT_SIZE = 36
const SUBTITLE_SHADOW_OFFSET_RATIO = 0.1
const DEFAULT_VIDEO_ASPECT_RATIO = 720.0 / 720.0
const VIDEO_OFFSET_RATIO = Vector2(0.0, 0.0)
const VIDEO_MARGIN_RATIO = Vector2(0.025, 0.01)
const SUBTITLE_PANEL_HEIGHT_RATIO = 0.2
const SUBTITLE_PANEL_BOTTOM_MARGIN_RATIO = 0.04
const WINDOW_SIZE = Vector2i(480, 1080)
const WINDOW_POSITION = Vector2i(1440, 0)

var subtitles: Array = []
var current_subtitle_index := -1
var video_queue: Array[String] = []
var noise_videos: Array[String] = []
var pending_next_video := ""
var current_video_path := ""
var current_is_noise := false
var rng := RandomNumberGenerator.new()
var llm_out_dir := ""
var llm_poll_elapsed := 0.0
var last_llm_file := ""

func _ready() -> void:
	_apply_window_settings()
	_apply_video_layout()
	if not video_player.finished.is_connected(_on_video_finished):
		video_player.finished.connect(_on_video_finished)
	rng.randomize()
	_scan_video_folder()
	llm_out_dir = _resolve_llm_out_dir()
	last_llm_file = _find_latest_llm_file()
	call_deferred("_update_video_cover")
	subtitle_panel.visible = false
	_apply_subtitle_style()
	if not PREFILL_VIDEO_QUEUE:
		video_queue.clear()
	if PREFILL_VIDEO_QUEUE:
		_play_next_from_queue()
	else:
		_play_random_noise()

func _apply_window_settings() -> void:
	DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_BORDERLESS, true)
	DisplayServer.window_set_size(WINDOW_SIZE)
	DisplayServer.window_set_position(WINDOW_POSITION)
	_apply_video_margins()
	_apply_subtitle_panel_layout()

func _apply_video_margins() -> void:
	var window_size = DisplayServer.window_get_size()
	if window_size.x <= 0 or window_size.y <= 0:
		window_size = WINDOW_SIZE
	var margin_x = int(round(float(window_size.x) * VIDEO_MARGIN_RATIO.x))
	var margin_y = int(round(float(window_size.y) * VIDEO_MARGIN_RATIO.y))
	margin_container.offset_left = margin_x
	margin_container.offset_top = margin_y
	margin_container.offset_right = -margin_x
	margin_container.offset_bottom = -margin_y

func _apply_subtitle_panel_layout() -> void:
	var window_size = DisplayServer.window_get_size()
	if window_size.x <= 0 or window_size.y <= 0:
		window_size = WINDOW_SIZE
	var height = int(round(float(window_size.y) * SUBTITLE_PANEL_HEIGHT_RATIO))
	var bottom_margin = int(round(float(window_size.y) * SUBTITLE_PANEL_BOTTOM_MARGIN_RATIO))
	subtitle_panel.offset_bottom = -bottom_margin
	subtitle_panel.offset_top = subtitle_panel.offset_bottom - height

func _apply_video_layout() -> void:
	video_player.expand = true
	video_player.anchor_left = 0.0
	video_player.anchor_top = 0.0
	video_player.anchor_right = 0.0
	video_player.anchor_bottom = 0.0
	if not video_frame.resized.is_connected(_update_video_cover):
		video_frame.resized.connect(_update_video_cover)
	_update_video_cover()

func _update_video_cover() -> void:
	var frame_size = video_frame.size
	if frame_size.x <= 0.0 or frame_size.y <= 0.0:
		return
	var aspect = _get_video_aspect_ratio()
	if aspect <= 0.0:
		return
	var target_size = Vector2.ZERO
	var frame_aspect = frame_size.x / frame_size.y
	if frame_aspect < aspect:
		target_size.y = frame_size.y
		target_size.x = frame_size.y * aspect
	else:
		target_size.x = frame_size.x
		target_size.y = frame_size.x / aspect
	var offset_ratio = Vector2(
		clamp(VIDEO_OFFSET_RATIO.x, -1.0, 1.0),
		clamp(VIDEO_OFFSET_RATIO.y, -1.0, 1.0)
	)
	var extra = target_size - frame_size
	var offset = Vector2(extra.x * 0.5 * offset_ratio.x, extra.y * 0.5 * offset_ratio.y)
	video_player.size = target_size
	video_player.position = (frame_size - target_size) * 0.5 + offset

func _get_video_aspect_ratio() -> float:
	if video_player.has_method("get_video_texture"):
		var texture = video_player.get_video_texture()
		if texture != null:
			var tex_size = texture.get_size()
			if tex_size.y > 0.0:
				return float(tex_size.x) / float(tex_size.y)
	return DEFAULT_VIDEO_ASPECT_RATIO

func _apply_subtitle_style() -> void:
	var font = load(SUBTITLE_FONT_PATH) as FontFile
	if font != null:
		subtitle_label.add_theme_font_override("font", font)
		subtitle_shadow_label.add_theme_font_override("font", font)
	else:
		push_error("Subtitle font not found: %s" % SUBTITLE_FONT_PATH)
	subtitle_label.add_theme_font_size_override("font_size", SUBTITLE_FONT_SIZE)
	subtitle_shadow_label.add_theme_font_size_override("font_size", SUBTITLE_FONT_SIZE)
	subtitle_label.add_theme_color_override("font_color", Color(1, 1, 0, 1))
	subtitle_shadow_label.add_theme_color_override("font_color", Color(0, 0, 0, 1))
	_apply_subtitle_layout()

func _apply_subtitle_layout() -> void:
	for label in [subtitle_shadow_label, subtitle_label]:
		label.anchor_left = 0.0
		label.anchor_top = 0.0
		label.anchor_right = 1.0
		label.anchor_bottom = 1.0
		label.offset_left = 0.0
		label.offset_top = 0.0
		label.offset_right = 0.0
		label.offset_bottom = 0.0
	var shadow_offset = max(1, int(round(float(SUBTITLE_FONT_SIZE) * SUBTITLE_SHADOW_OFFSET_RATIO)))
	subtitle_shadow_label.offset_left = shadow_offset
	subtitle_shadow_label.offset_top = shadow_offset
	subtitle_shadow_label.offset_right = shadow_offset
	subtitle_shadow_label.offset_bottom = shadow_offset

func _process(delta: float) -> void:
	_poll_llm_out(delta)
	if subtitles.is_empty():
		return
	var current_time = _get_video_time()
	_update_subtitle(current_time)

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_ESCAPE:
			get_tree().quit()
		elif event.keycode == KEY_SPACE:
			_skip_to_next()

func _load_subtitles_for_video(video_path: String) -> Array:
	var base = video_path.get_basename()
	if USE_FRENCH_SUBTITLES:
		base = base + "-fr"
	for extension in SUBTITLE_EXTENSIONS:
		var candidate = base + "." + extension
		if FileAccess.file_exists(candidate):
			return _parse_sbv(candidate)
	return []

func _parse_sbv(path: String) -> Array:
	var cues: Array = []
	var file = FileAccess.open(path, FileAccess.READ)
	if file == null:
		push_error("Failed to open subtitles: %s" % path)
		return cues
	var lines = file.get_as_text().split("\n")
	var idx = 0
	while idx < lines.size():
		var line = lines[idx].strip_edges()
		if line == "":
			idx += 1
			continue
		if line.find(",") == -1:
			idx += 1
			continue
		var parts = line.split(",")
		if parts.size() < 2:
			idx += 1
			continue
		var start = _parse_timecode(parts[0].strip_edges())
		var end = _parse_timecode(parts[1].strip_edges())
		idx += 1
		var text_lines: Array = []
		while idx < lines.size() and lines[idx].strip_edges() != "":
			var cleaned = lines[idx].strip_edges().replace("\u00a0", " ")
			text_lines.append(cleaned)
			idx += 1
		var cue_text = " ".join(text_lines).strip_edges()
		if start >= 0.0 and end >= 0.0 and cue_text != "":
			cues.append({"start": start, "end": end, "text": cue_text})
		idx += 1
	return cues

func _parse_timecode(text: String) -> float:
	var cleaned = text.strip_edges()
	if cleaned == "":
		return -1.0
	var parts = cleaned.split(":")
	var hours = 0
	var minutes = 0
	var seconds = 0.0
	if parts.size() == 3:
		hours = int(parts[0])
		minutes = int(parts[1])
		seconds = float(parts[2])
	elif parts.size() == 2:
		minutes = int(parts[0])
		seconds = float(parts[1])
	else:
		return -1.0
	return hours * 3600.0 + minutes * 60.0 + seconds

func _get_video_time() -> float:
	if video_player.has_method("get_stream_position"):
		return video_player.get_stream_position()
	if video_player.has_method("get_playback_position"):
		return video_player.get_playback_position()
	var pos = video_player.get("stream_position")
	if typeof(pos) == TYPE_FLOAT:
		return pos
	return 0.0

func _find_cue_index(time_sec: float) -> int:
	var low = 0
	var high = subtitles.size() - 1
	while low <= high:
		var mid = int((low + high) / 2)
		var cue = subtitles[mid]
		if time_sec < cue["start"]:
			high = mid - 1
		elif time_sec > cue["end"]:
			low = mid + 1
		else:
			return mid
	return -1

func _update_subtitle(time_sec: float) -> void:
	var idx = _find_cue_index(time_sec)
	if idx == -1:
		if subtitle_label.text != "":
			subtitle_label.text = ""
			subtitle_shadow_label.text = ""
		if subtitle_panel.visible:
			subtitle_panel.visible = false
		current_subtitle_index = -1
		return
	if idx != current_subtitle_index:
		current_subtitle_index = idx
		var cue_text = subtitles[idx]["text"]
		subtitle_label.text = cue_text
		subtitle_shadow_label.text = cue_text
		if not subtitle_panel.visible:
			subtitle_panel.visible = true

func _scan_video_folder() -> void:
	video_queue.clear()
	noise_videos.clear()
	var dir = DirAccess.open(VIDEO_FOLDER_PATH)
	if dir == null:
		push_error("Video folder not found: %s" % VIDEO_FOLDER_PATH)
		return
	dir.list_dir_begin()
	var file = dir.get_next()
	while file != "":
		if not dir.current_is_dir():
			var lower = file.to_lower()
			if lower.ends_with(".ogv"):
				var path = VIDEO_FOLDER_PATH + "/" + file
				if _is_noise_video(lower):
					noise_videos.append(path)
				elif _is_numbered_video(file):
					video_queue.append(path)
		file = dir.get_next()
	dir.list_dir_end()
	video_queue.sort()

func _is_noise_video(filename_lower: String) -> bool:
	return filename_lower.begins_with("noise_")

func _is_numbered_video(filename: String) -> bool:
	if filename.length() == 0:
		return false
	var first = filename[0]
	return first >= "0" and first <= "9"

func _play_video(path: String, is_noise: bool) -> void:
	var stream = load(path)
	if stream == null:
		push_error("Video not found or unsupported: %s" % path)
		return
	current_video_path = path
	current_is_noise = is_noise
	video_player.stream = stream
	video_player.loop = false
	video_player.play()
	current_subtitle_index = -1
	if is_noise:
		_clear_subtitles()
	else:
		subtitles = _load_subtitles_for_video(path)
		_update_subtitle(0.0)

func _play_next_from_queue() -> void:
	if video_queue.is_empty():
		return
	var next_path = video_queue.pop_front()
	if current_video_path == "":
		_play_video(next_path, false)
	else:
		_transition_to_video(next_path)

func _transition_to_video(next_path: String) -> void:
	pending_next_video = next_path
	_play_random_noise()

func _play_random_noise() -> void:
	if noise_videos.is_empty():
		if pending_next_video != "":
			var next_path = pending_next_video
			pending_next_video = ""
			_play_video(next_path, false)
		return
	var index = rng.randi_range(0, noise_videos.size() - 1)
	_play_video(noise_videos[index], true)

func _on_video_finished() -> void:
	if current_is_noise:
		if pending_next_video != "":
			var next_path = pending_next_video
			pending_next_video = ""
			_play_video(next_path, false)
			return
		if not video_queue.is_empty():
			var queued = video_queue.pop_front()
			_play_video(queued, false)
			return
		if current_video_path != "":
			_play_video(current_video_path, true)
		return
	if video_queue.is_empty():
		_play_random_noise()
		return
	pending_next_video = video_queue.pop_front()
	_play_random_noise()

func _skip_to_next() -> void:
	if video_queue.is_empty():
		return
	var next_path = video_queue.pop_front()
	if current_video_path == "":
		_play_video(next_path, false)
		return
	if current_is_noise:
		pending_next_video = ""
		_play_video(next_path, false)
		return
	pending_next_video = next_path
	_play_random_noise()

func _clear_subtitles() -> void:
	subtitles = []
	current_subtitle_index = -1
	subtitle_label.text = ""
	subtitle_shadow_label.text = ""
	subtitle_panel.visible = false

func flush_queue() -> void:
	video_queue.clear()
	pending_next_video = ""

func enqueue_video(filename: String) -> void:
	var path = _resolve_video_path(filename)
	if path == "":
		push_warning("Video not found: %s" % filename)
		return
	video_queue.append(path)

func is_main_video_playing() -> bool:
	return video_player.is_playing() and not current_is_noise

func _resolve_video_path(filename: String) -> String:
	var path = filename
	if not filename.begins_with("res://"):
		path = VIDEO_FOLDER_PATH + "/" + filename
	if FileAccess.file_exists(path):
		return path
	return ""

func _resolve_llm_out_dir() -> String:
	var environment_override = OS.get_environment("LLM_OUT_OVERRIDE")
	if environment_override != "":
		return environment_override.simplify_path()
	if LLM_OUT_OVERRIDE != "":
		return LLM_OUT_OVERRIDE.simplify_path()
	var base_dir = ""
	if OS.has_feature("editor"):
		base_dir = ProjectSettings.globalize_path("res://")
	else:
		base_dir = OS.get_executable_path().get_base_dir()
	return base_dir.path_join(LLM_OUT_RELATIVE_PATH).simplify_path()

func _poll_llm_out(delta: float) -> void:
	llm_poll_elapsed += delta
	if llm_poll_elapsed < LLM_POLL_INTERVAL:
		return
	llm_poll_elapsed = 0.0
	if llm_out_dir == "":
		return
	var latest = _find_latest_llm_file()
	if latest == "" or latest == last_llm_file:
		return
	var next_video = _read_llm_video_request(latest)
	if next_video == "":
		return
	last_llm_file = latest
	enqueue_video(next_video)
	if not video_player.is_playing():
		_play_next_from_queue()

func _find_latest_llm_file() -> String:
	var dir = DirAccess.open(llm_out_dir)
	if dir == null:
		return ""
	dir.list_dir_begin()
	var latest_name = ""
	var name = dir.get_next()
	while name != "":
		if not dir.current_is_dir() and name.ends_with(".txt"):
			if name > latest_name:
				latest_name = name
		name = dir.get_next()
	dir.list_dir_end()
	if latest_name == "":
		return ""
	return llm_out_dir.path_join(latest_name)

func _read_llm_video_request(path: String) -> String:
	var file = FileAccess.open(path, FileAccess.READ)
	if file == null:
		return ""
	var text = file.get_as_text()
	for line in text.split("\n"):
		var cleaned = line.strip_edges()
		if cleaned != "":
			return cleaned
	return ""
