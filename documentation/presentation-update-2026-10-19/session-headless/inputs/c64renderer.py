import os
import sys

import pygame

# Toggle for Windows "always on top" behavior.
ENABLE_ALWAYS_ON_TOP = True
ALWAYS_ON_TOP_REFRESH_MS = 1000

# Commodore 64 style display settings
C64_COLS = 78
C64_ROWS = 40
C64_STATUS_ROWS = 1  # Reserve top row for status/title bar.
C64_CELL_SIZE_H = 8
C64_CELL_SIZE_V = 10
LOGICAL_WIDTH = C64_COLS * C64_CELL_SIZE_H
LOGICAL_HEIGHT = C64_ROWS * C64_CELL_SIZE_V

C64_BLUE = (64, 49, 141)
C64_LIGHT_BLUE = (64 * 3 // 2, 49 * 3 // 2, 141 * 3 // 2)
C64_LIGHT_GRAY = (202, 202, 202)
C64_WHITE = (255, 255, 255)
C64_BLACK = (0, 0, 0)

# Distinct border styling so it stays visible against the screen background.
C64_BORDER_COLOR = C64_BLUE
BORDER_THICKNESS = 64

class C64Renderer:
    def __init__(
        self,
        font_path=None,
        scale=None,
        fps=60,
        fullscreen=False,
        display_index=None,
        always_on_top=None,
        output_scale=1,
        fit_to_display=False,
        window_size=None,
        window_position=None,
        borderless=False,
    ):
        if pygame is None:
            raise ImportError("pygame is required for the C64 renderer. Install pygame to enable it.")

        pygame.init()
        pygame.display.set_caption("Plundered Hearts - C64 view")

        self.fullscreen = bool(fullscreen)
        self.display_index = self._normalize_display_index(display_index)
        self.window_size = self._normalize_window_size(window_size)
        self.window_position = self._normalize_window_position(window_position)
        self.borderless = bool(borderless)
        if always_on_top is None:
            self.always_on_top = ENABLE_ALWAYS_ON_TOP
        else:
            self.always_on_top = bool(always_on_top)
        self.display_size = self._get_display_size(self.display_index)
        self.fps = fps
        try:
            self.output_scale = max(1, int(output_scale))
        except (TypeError, ValueError):
            self.output_scale = 1
        self.fit_to_display = bool(fit_to_display)
        self.scale = self._determine_scale(scale, self.display_size)
        self.total_width = LOGICAL_WIDTH * self.scale * self.output_scale + BORDER_THICKNESS * 2
        self.total_height = LOGICAL_HEIGHT * self.scale * self.output_scale + BORDER_THICKNESS * 2
        if self.fullscreen and self.display_size:
            self.window_width, self.window_height = self.display_size
            if self.fit_to_display:
                self.render_offset_x = 0
                self.render_offset_y = 0
            else:
                self.render_offset_x = max(0, (self.window_width - self.total_width) // 2)
                self.render_offset_y = max(0, (self.window_height - self.total_height) // 2)
            window_flags = pygame.FULLSCREEN
        else:
            if self.window_size:
                self.window_width, self.window_height = self.window_size
                if self.fit_to_display:
                    self.render_offset_x = 0
                    self.render_offset_y = 0
                else:
                    self.render_offset_x = max(0, (self.window_width - self.total_width) // 2)
                    self.render_offset_y = max(0, (self.window_height - self.total_height) // 2)
                self.total_width = self.window_width
                self.total_height = self.window_height
            else:
                self.window_width = self.total_width
                self.window_height = self.total_height
                self.render_offset_x = 0
                self.render_offset_y = 0
            window_flags = 0
        if self.borderless and not self.fullscreen:
            window_flags |= pygame.NOFRAME
        if self.window_position and not self.fullscreen:
            self._set_window_position_env(self.window_position)
        self.window = self._create_window(window_flags)
        if self.always_on_top:
            self._topmost_applied = False
            self._prime_always_on_top()
        else:
            self._next_topmost_check_ms = None
        self.clock = pygame.time.Clock()

        self.logical_surface = pygame.Surface((LOGICAL_WIDTH, LOGICAL_HEIGHT), pygame.SRCALPHA).convert_alpha()

        self.cursor_x = 0
        self.cursor_y = 0  # Content row index; status bar is separate.
        self.status_text = ""
        self.status_bar_bg = C64_LIGHT_BLUE
        self.content_rows = C64_ROWS - C64_STATUS_ROWS
        self.buffer = [[" "] * C64_COLS for _ in range(self.content_rows)]
        self.default_fg = C64_LIGHT_GRAY
        self.default_bg = C64_BLUE
        self.row_fg_colors = [self.default_fg] * self.content_rows
        self.row_bg_colors = [self.default_bg] * self.content_rows
        self.glyphs = self._load_font(font_path)
        self.default_glyph = self._render_pattern(self._fallback_pattern("?"))
        self._glyph_cache = {self.default_fg: self.glyphs}
        self._default_glyph_cache = {self.default_fg: self.default_glyph}

    def _determine_scale(self, forced_scale, display_size=None):
        if forced_scale:
            return max(1, int(forced_scale))
        if display_size:
            usable_w = max(1, display_size[0] - BORDER_THICKNESS * 2)
            usable_h = max(1, display_size[1] - BORDER_THICKNESS * 2)
            max_w = max(1, usable_w // LOGICAL_WIDTH)
            max_h = max(1, usable_h // LOGICAL_HEIGHT)
            return max(1, min(max_w, max_h))
        try:
            info = pygame.display.Info()
            usable_w = max(1, info.current_w - BORDER_THICKNESS * 2)
            usable_h = max(1, info.current_h - BORDER_THICKNESS * 2)
            max_w = max(1, usable_w // LOGICAL_WIDTH)
            max_h = max(1, usable_h // LOGICAL_HEIGHT)
            return max(1, min(max_w, max_h))
        except pygame.error:
            return 2

    def _normalize_display_index(self, display_index):
        if display_index is None:
            return None
        try:
            index = int(display_index)
        except (TypeError, ValueError):
            return None
        if index < 0:
            return None
        try:
            num_displays = pygame.display.get_num_displays()
        except Exception:
            return index
        if index >= num_displays:
            return None
        return index

    def _normalize_window_size(self, window_size):
        if not window_size:
            return None
        try:
            width, height = window_size
            width = int(width)
            height = int(height)
            if width <= 0 or height <= 0:
                return None
            return (width, height)
        except (TypeError, ValueError):
            return None

    def _normalize_window_position(self, window_position):
        if window_position is None:
            return None
        try:
            x, y = window_position
            return (int(x), int(y))
        except (TypeError, ValueError):
            return None

    def _set_window_position_env(self, position):
        try:
            os.environ["SDL_VIDEO_WINDOW_POS"] = f"{position[0]},{position[1]}"
            os.environ.pop("SDL_VIDEO_CENTERED", None)
        except Exception:
            pass

    def _get_display_size(self, display_index):
        if display_index is not None:
            try:
                sizes = pygame.display.get_desktop_sizes()
                if sizes and 0 <= display_index < len(sizes):
                    return sizes[display_index]
            except Exception:
                pass
        try:
            info = pygame.display.Info()
            if info.current_w and info.current_h:
                return (info.current_w, info.current_h)
        except pygame.error:
            return None
        return None

    def _create_window(self, window_flags):
        try:
            if self.display_index is not None:
                return pygame.display.set_mode(
                    (self.window_width, self.window_height),
                    window_flags,
                    display=self.display_index,
                )
        except TypeError:
            pass
        return pygame.display.set_mode((self.window_width, self.window_height), window_flags)

    def _set_always_on_top(self):
        if not sys.platform.startswith("win"):
            return False
        try:
            import ctypes
            from ctypes import wintypes

            wm_info = pygame.display.get_wm_info()
            hwnd = wm_info.get("window")
            if not hwnd:
                return False
            user32 = ctypes.windll.user32
            if not getattr(user32, "_setwindowpos_configured", False):
                user32.SetWindowPos.argtypes = [
                    wintypes.HWND,
                    wintypes.HWND,
                    wintypes.INT,
                    wintypes.INT,
                    wintypes.INT,
                    wintypes.INT,
                    wintypes.UINT,
                ]
                user32.SetWindowPos.restype = wintypes.BOOL
                user32._setwindowpos_configured = True
            HWND_TOPMOST = wintypes.HWND(-1)
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_SHOWWINDOW = 0x0040
            user32.SetWindowPos(
                hwnd,
                HWND_TOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
            )
            return True
        except Exception:
            return False

    def _prime_always_on_top(self):
        try:
            pygame.event.pump()
        except Exception:
            pass
        if self._set_always_on_top():
            self._topmost_applied = True
            self._next_topmost_check_ms = pygame.time.get_ticks() + ALWAYS_ON_TOP_REFRESH_MS
        else:
            self._next_topmost_check_ms = pygame.time.get_ticks()

    def _refresh_always_on_top(self):
        if not self.always_on_top or self._next_topmost_check_ms is None:
            return
        now_ms = pygame.time.get_ticks()
        if not self._topmost_applied:
            if self._set_always_on_top():
                self._topmost_applied = True
            self._next_topmost_check_ms = now_ms + ALWAYS_ON_TOP_REFRESH_MS
            return
        if now_ms < self._next_topmost_check_ms:
            return
        self._set_always_on_top()
        self._next_topmost_check_ms = now_ms + ALWAYS_ON_TOP_REFRESH_MS

    def _load_font(self, font_path):
        # Always use the built-in placeholder font to avoid external dependencies.
        return self._build_placeholder_font()

    def _fallback_pattern(self, char):
        patterns = {
            " ": ["     "] * 7,
            "?": [
                " XXX ",
                "X   X",
                "    X",
                "  XX ",
                "  X  ",
                "     ",
                "  X  ",
            ],
            "!": [
                "  X  ",
                "  X  ",
                "  X  ",
                "  X  ",
                "  X  ",
                "     ",
                "  X  ",
            ],
            ".": [
                "     ",
                "     ",
                "     ",
                "     ",
                "     ",
                " XXX ",
                " XXX ",
            ],
            ",": [
                "     ",
                "     ",
                "     ",
                "     ",
                " XXX ",
                "   X ",
                "  X  ",
            ],
            ":": [
                "     ",
                "  X  ",
                "     ",
                "     ",
                "  X  ",
                "     ",
                "     ",
            ],
            ";": [
                "     ",
                "  X  ",
                "     ",
                "     ",
                "  X  ",
                "  X  ",
                " X   ",
            ],
            "-": [
                "     ",
                "     ",
                "XXXXX",
                "     ",
                "     ",
                "     ",
                "     ",
            ],
            "_": [
                "     ",
                "     ",
                "     ",
                "     ",
                "     ",
                "     ",
                "XXXXX",
            ],
            "'": [
                "  X  ",
                "  X  ",
                "  X  ",
                "     ",
                "     ",
                "     ",
                "     ",
            ],
            "’": [
                "  X  ",
                "  X  ",
                "  X  ",
                "     ",
                "     ",
                "     ",
                "     ",
            ],
            '"': [
                " X X ",
                " X X ",
                " X X ",
                "     ",
                "     ",
                "     ",
                "     ",
            ],
            "(": [
                "   X ",
                "  X  ",
                " X   ",
                " X   ",
                " X   ",
                "  X  ",
                "   X ",
            ],
            ")": [
                " X   ",
                "  X  ",
                "   X ",
                "   X ",
                "   X ",
                "  X  ",
                " X   ",
            ],
            "/": [
                "    X",
                "   X ",
                "  X  ",
                " X   ",
                "X    ",
                "     ",
                "     ",
            ],
            "\\": [
                "X    ",
                " X   ",
                "  X  ",
                "   X ",
                "    X",
                "     ",
                "     ",
            ],
            "+": [
                "     ",
                "  X  ",
                "  X  ",
                "XXXXX",
                "  X  ",
                "  X  ",
                "     ",
            ],
            "*": [
                "X   X",
                " X X ",
                "  X  ",
                "XXXXX",
                "  X  ",
                " X X ",
                "X   X",
            ],
            "<": [
                "    X",
                "   X ",
                "  X  ",
                " X   ",
                "  X  ",
                "   X ",
                "    X",
            ],
            ">": [
                "X    ",
                " X   ",
                "  X  ",
                "   X ",
                "  X  ",
                " X   ",
                "X    ",
            ],
            "[": [
                " XXX ",
                " X   ",
                " X   ",
                " X   ",
                " X   ",
                " X   ",
                " XXX ",
            ],
            "]": [
                " XXX ",
                "   X ",
                "   X ",
                "   X ",
                "   X ",
                "   X ",
                " XXX ",
            ],
            "(": [
                "   X ",
                "  X  ",
                " X   ",
                " X   ",
                " X   ",
                "  X  ",
                "   X ",
            ],
            ")": [
                " X   ",
                "  X  ",
                "   X ",
                "   X ",
                "   X ",
                "  X  ",
                " X   ",
            ],
            "{": [
                "   XX",
                "  X  ",
                "  X  ",
                " XX  ",
                "  X  ",
                "  X  ",
                "   XX",
            ],
            "}": [
                "XX   ",
                "  X  ",
                "  X  ",
                "  XX ",
                "  X  ",
                "  X  ",
                "XX   ",
            ],
        }
        if char in patterns:
            return patterns[char]
        return patterns["?"]

    def _build_placeholder_font(self):
        letters = {
            "A": [
                " XXX ",
                "X   X",
                "X   X",
                "XXXXX",
                "X   X",
                "X   X",
                "X   X",
            ],
            "B": [
                "XXXX ",
                "X   X",
                "X   X",
                "XXXX ",
                "X   X",
                "X   X",
                "XXXX ",
            ],
            "C": [
                " XXX ",
                "X   X",
                "X    ",
                "X    ",
                "X    ",
                "X   X",
                " XXX ",
            ],
            "D": [
                "XXXX ",
                "X   X",
                "X   X",
                "X   X",
                "X   X",
                "X   X",
                "XXXX ",
            ],
            "E": [
                "XXXXX",
                "X    ",
                "X    ",
                "XXXX ",
                "X    ",
                "X    ",
                "XXXXX",
            ],
            "F": [
                "XXXXX",
                "X    ",
                "X    ",
                "XXXX ",
                "X    ",
                "X    ",
                "X    ",
            ],
            "G": [
                " XXX ",
                "X   X",
                "X    ",
                "X XXX",
                "X   X",
                "X   X",
                " XXX ",
            ],
            "H": [
                "X   X",
                "X   X",
                "X   X",
                "XXXXX",
                "X   X",
                "X   X",
                "X   X",
            ],
            "I": [
                " XXX ",
                "  X  ",
                "  X  ",
                "  X  ",
                "  X  ",
                "  X  ",
                " XXX ",
            ],
            "J": [
                "  XXX",
                "   X ",
                "   X ",
                "   X ",
                "   X ",
                "X  X ",
                " XX  ",
            ],
            "K": [
                "X   X",
                "X  X ",
                "X X  ",
                "XX   ",
                "X X  ",
                "X  X ",
                "X   X",
            ],
            "L": [
                "X    ",
                "X    ",
                "X    ",
                "X    ",
                "X    ",
                "X    ",
                "XXXXX",
            ],
            "M": [
                "X   X",
                "XX XX",
                "X X X",
                "X   X",
                "X   X",
                "X   X",
                "X   X",
            ],
            "N": [
                "X   X",
                "XX  X",
                "X X X",
                "X  XX",
                "X   X",
                "X   X",
                "X   X",
            ],
            "O": [
                " XXX ",
                "X   X",
                "X   X",
                "X   X",
                "X   X",
                "X   X",
                " XXX ",
            ],
            "P": [
                "XXXX ",
                "X   X",
                "X   X",
                "XXXX ",
                "X    ",
                "X    ",
                "X    ",
            ],
            "Q": [
                " XXX ",
                "X   X",
                "X   X",
                "X   X",
                "X X X",
                "X  X ",
                " XX X",
            ],
            "R": [
                "XXXX ",
                "X   X",
                "X   X",
                "XXXX ",
                "X X  ",
                "X  X ",
                "X   X",
            ],
            "S": [
                " XXXX",
                "X    ",
                "X    ",
                " XXX ",
                "    X",
                "    X",
                "XXXX ",
            ],
            "T": [
                "XXXXX",
                "  X  ",
                "  X  ",
                "  X  ",
                "  X  ",
                "  X  ",
                "  X  ",
            ],
            "U": [
                "X   X",
                "X   X",
                "X   X",
                "X   X",
                "X   X",
                "X   X",
                " XXX ",
            ],
            "V": [
                "X   X",
                "X   X",
                "X   X",
                "X   X",
                "X   X",
                " X X ",
                "  X  ",
            ],
            "W": [
                "X   X",
                "X   X",
                "X   X",
                "X X X",
                "X X X",
                "XX XX",
                "X   X",
            ],
            "X": [
                "X   X",
                "X   X",
                " X X ",
                "  X  ",
                " X X ",
                "X   X",
                "X   X",
            ],
            "Y": [
                "X   X",
                "X   X",
                " X X ",
                "  X  ",
                "  X  ",
                "  X  ",
                "  X  ",
            ],
            "Z": [
                "XXXXX",
                "    X",
                "   X ",
                "  X  ",
                " X   ",
                "X    ",
                "XXXXX",
            ],
            "0": [
                " XXX ",
                "X   X",
                "X  XX",
                "X X X",
                "XX  X",
                "X   X",
                " XXX ",
            ],
            "1": [
                "  X  ",
                " XX  ",
                "  X  ",
                "  X  ",
                "  X  ",
                "  X  ",
                " XXX ",
            ],
            "2": [
                " XXX ",
                "X   X",
                "    X",
                "   X ",
                "  X  ",
                " X   ",
                "XXXXX",
            ],
            "3": [
                " XXX ",
                "X   X",
                "    X",
                "  XX ",
                "    X",
                "X   X",
                " XXX ",
            ],
            "4": [
                "   X ",
                "  XX ",
                " X X ",
                "X  X ",
                "XXXXX",
                "   X ",
                "   X ",
            ],
            "5": [
                "XXXXX",
                "X    ",
                "X    ",
                "XXXX ",
                "    X",
                "    X",
                "XXXX ",
            ],
            "6": [
                " XXX ",
                "X    ",
                "X    ",
                "XXXX ",
                "X   X",
                "X   X",
                " XXX ",
            ],
            "7": [
                "XXXXX",
                "    X",
                "   X ",
                "  X  ",
                "  X  ",
                "  X  ",
                "  X  ",
            ],
            "8": [
                " XXX ",
                "X   X",
                "X   X",
                " XXX ",
                "X   X",
                "X   X",
                " XXX ",
            ],
            "9": [
                " XXX ",
                "X   X",
                "X   X",
                " XXXX",
                "    X",
                "    X",
                " XXX ",
            ],
        }

        glyphs = {}
        for code in range(256):
            char = chr(code)
            upper_char = char.upper()
            pattern = letters.get(upper_char) or self._fallback_pattern(char) or self._fallback_pattern("?")
            glyphs[code] = self._render_pattern(pattern)
        glyphs[ord("’")] = glyphs.get(ord("'"), self._render_pattern(self._fallback_pattern("'")))
        return glyphs

    def _render_pattern(self, pattern_lines):
        surface = pygame.Surface((C64_CELL_SIZE_H, C64_CELL_SIZE_V), pygame.SRCALPHA).convert_alpha()
        surface.fill((*C64_BLUE, 0))
        top_margin = 1
        left_margin = 1
        for row_idx, line in enumerate(pattern_lines):
            for col_idx, char in enumerate(line):
                if char != " ":
                    surface.set_at((left_margin + col_idx, top_margin + row_idx), C64_LIGHT_GRAY)
        return surface

    def clear(self):
        self.buffer = [[" "] * C64_COLS for _ in range(self.content_rows)]
        self.cursor_x = 0
        self.cursor_y = 0
        self.row_fg_colors = [self.default_fg] * self.content_rows
        self.row_bg_colors = [self.default_bg] * self.content_rows

    def _newline(self):
        self.cursor_x = 0
        self.cursor_y += 1
        if self.cursor_y >= self.content_rows:
            self.buffer.pop(0)
            self.buffer.append([" "] * C64_COLS)
            self.row_fg_colors.pop(0)
            self.row_bg_colors.pop(0)
            self.row_fg_colors.append(self.default_fg)
            self.row_bg_colors.append(self.default_bg)
            self.cursor_y = self.content_rows - 1

    def _glyph_for_char(self, char):
        code = ord(char) if char else ord("?")
        return self.glyphs.get(code, self.default_glyph)

    def _set_row_style(self, row_index, fg_color=None, bg_color=None):
        if fg_color is not None:
            self.row_fg_colors[row_index] = fg_color
        if bg_color is not None:
            self.row_bg_colors[row_index] = bg_color

    def _tint_surface(self, surface, color):
        tinted = pygame.Surface(surface.get_size(), pygame.SRCALPHA).convert_alpha()
        width, height = surface.get_size()
        for y in range(height):
            for x in range(width):
                if surface.get_at((x, y)).a:
                    tinted.set_at((x, y), (*color, 255))
        return tinted

    def _tint_glyphs(self, color):
        tinted = {}
        for code, glyph in self.glyphs.items():
            tinted[code] = self._tint_surface(glyph, color)
        return tinted

    def _get_glyphs_for_color(self, color):
        if color is None:
            return self.glyphs
        key = tuple(color)
        cached = self._glyph_cache.get(key)
        if cached:
            return cached
        tinted = self._tint_glyphs(key)
        self._glyph_cache[key] = tinted
        return tinted

    def _get_default_glyph_for_color(self, color):
        if color is None:
            return self.default_glyph
        key = tuple(color)
        cached = self._default_glyph_cache.get(key)
        if cached:
            return cached
        tinted = self._tint_surface(self.default_glyph, key)
        self._default_glyph_cache[key] = tinted
        return tinted

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

    def write(self, text, fg_color=None, bg_color=None):
        for ch in text:
            if ch == "\n":
                self._newline()
                if fg_color is not None or bg_color is not None:
                    self._set_row_style(self.cursor_y, fg_color, bg_color)
                continue
            if ch == "\f":
                self.clear()
                if fg_color is not None or bg_color is not None:
                    self._set_row_style(self.cursor_y, fg_color, bg_color)
                continue
            if fg_color is not None or bg_color is not None:
                self._set_row_style(self.cursor_y, fg_color, bg_color)
            sanitized = ch
            if not sanitized.isprintable() or sanitized == "\r":
                sanitized = " "
            sanitized = sanitized.upper()
            self.buffer[self.cursor_y][self.cursor_x] = sanitized
            self.cursor_x += 1
            if self.cursor_x >= C64_COLS:
                self._newline()
                if fg_color is not None or bg_color is not None:
                    self._set_row_style(self.cursor_y, fg_color, bg_color)

    def set_status_bar(self, text):
        """Update the persistent status/title bar shown on the top row."""
        self.status_text = (text or "").strip().upper()

    def set_status_bar_color(self, color):
        if color:
            self.status_bar_bg = color

    def _draw_status_bar(self):
        if not self.status_text:
            return
        pygame.draw.rect(
            self.logical_surface,
            self.status_bar_bg,
            pygame.Rect(0, 0, LOGICAL_WIDTH, C64_CELL_SIZE_V),
        )
        truncated = self.status_text[:C64_COLS].ljust(C64_COLS)
        for x, ch in enumerate(truncated):
            glyph = self._glyph_for_char(ch)
            self.logical_surface.blit(glyph, (x * C64_CELL_SIZE_H, 0))

    def _draw_buffer(self):
        self.logical_surface.fill(C64_BLUE)
        self._draw_status_bar()
        for y, row in enumerate(self.buffer):
            bg_color = self.row_bg_colors[y]
            if bg_color != C64_BLUE:
                pygame.draw.rect(
                    self.logical_surface,
                    bg_color,
                    pygame.Rect(
                        0,
                        (y + C64_STATUS_ROWS) * C64_CELL_SIZE_V,
                        LOGICAL_WIDTH,
                        C64_CELL_SIZE_V,
                    ),
                )
            glyphs = self._get_glyphs_for_color(self.row_fg_colors[y])
            for x, ch in enumerate(row):
                code = ord(ch) if ch else ord("?")
                glyph = glyphs.get(code)
                if glyph is None:
                    glyph = self._get_default_glyph_for_color(self.row_fg_colors[y])
                self.logical_surface.blit(
                    glyph, (x * C64_CELL_SIZE_H, (y + C64_STATUS_ROWS) * C64_CELL_SIZE_V)
                )

    def render_frame(self, show_cursor=False):
        self._refresh_always_on_top()
        self._draw_buffer()
        frame = self.logical_surface.copy()
        if show_cursor:
            cursor_color = C64_LIGHT_BLUE
            cursor_rect = (
                self.cursor_x * C64_CELL_SIZE_H,
                (self.cursor_y + C64_STATUS_ROWS) * C64_CELL_SIZE_V,
                C64_CELL_SIZE_H,
                C64_CELL_SIZE_V,
            )
            pygame.draw.rect(frame, cursor_color, cursor_rect)
        scaled_w = LOGICAL_WIDTH * self.scale * self.output_scale
        scaled_h = LOGICAL_HEIGHT * self.scale * self.output_scale
        scaled = pygame.transform.smoothscale(frame, (scaled_w, scaled_h)).convert_alpha()
        if self.fit_to_display:
            target_w = max(1, self.window_width - BORDER_THICKNESS * 2)
            target_h = max(1, self.window_height - BORDER_THICKNESS * 2)
            if scaled_w != target_w or scaled_h != target_h:
                scaled = pygame.transform.smoothscale(scaled, (target_w, target_h)).convert_alpha()
        self.window.fill(C64_BORDER_COLOR)
        dest_x = self.render_offset_x + BORDER_THICKNESS
        dest_y = self.render_offset_y + BORDER_THICKNESS
        self.window.blit(scaled, (dest_x, dest_y))
        if BORDER_THICKNESS > 0:
            border_rect = pygame.Rect(
                self.render_offset_x,
                self.render_offset_y,
                self.total_width,
                self.total_height,
            )
            pygame.draw.rect(self.window, C64_BORDER_COLOR, border_rect, BORDER_THICKNESS)
        pygame.display.flip()
        if self.always_on_top and not self._topmost_applied:
            self._prime_always_on_top()
        self.clock.tick(self.fps)
