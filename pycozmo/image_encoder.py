"""

Cozmo image run-length encoding and decoding.

"""

import sys
from io import StringIO

from PIL import Image

__all__ = [
    "ImageDecoder",
    "ImageEncoder",
    "image_to_str",
    "render",
    "str_to_image",
]


def render(image: bytes) -> None:
    for y in range(32):
        sys.stdout.write(f"{y:3d}")
        for x in range(128):
            if ((x + 1) % 10) == 1:
                sys.stdout.write(" ")
            sys.stdout.write(chr(ord("0") + image[y * 128 + x])) if image[
                y * 128 + x
            ] else sys.stdout.write(".")
        sys.stdout.write("\n")


def image_to_str(image):
    s = StringIO()
    for y in range(32):
        for x in range(128):
            if image[y * 128 + x]:
                s.write(chr(ord("0") + image[y * 128 + x]))
            else:
                s.write(".")
        s.write("\n")
    return s.getvalue()


def str_to_image(sim: str) -> Image:
    s = StringIO(sim)
    im = Image.new("1", (128, 32), color=0)
    px = im.load()
    for y in range(32):
        while True:
            line = s.readline().strip()
            if line:
                break
        x = 0
        for c in line:
            px[x, y] = 1 if c and c != "." else 0
            x += 1
    return im


class ImageDecoder:
    def __init__(self, buffer: bytes):
        self.buffer = buffer
        self.image = bytearray(128 * 32)
        self.x = 0
        self.y = 0
        self.last_draw = False
        self.repeat_column_shift = False
        self.debug = False

    def _draw(self, x, y, color=1):
        assert 0 <= x < 128
        assert 0 <= y < 32
        assert 0 <= color < 256
        if y < 32 and x < 128:
            self.image[y * 128 + x] = color

    def _execute(self, b: int) -> None:
        cmd = (b & 0xC0) >> 6
        cnt = b & 0x3F
        if cmd == 0:  # Skip column
            cnt += 1
            if self.debug:
                print(f"Skip column {cnt}")
            if self.last_draw:
                self.x += 1
            self.x += cnt
            self.y = 0
            self.last_draw = False
            self.repeat_column_shift = False
        elif cmd == 1:  # Repeat column
            cnt += 1
            if self.debug:
                print(f"Repeat column {cnt}")
            if not self.repeat_column_shift:
                self.x += 1
            for _ in range(cnt):
                i = self.x
                for _ in range(32):
                    self.image[i] = self.image[i - 1]
                    i += 128
                self.x += 1
            self.y = 0
            self.last_draw = False
            self.repeat_column_shift = True
        elif cmd == 2:  # Draw / Skip
            draw = cnt & 0x01
            cnt >>= 1
            draw2 = cnt & 0x01
            cnt >>= 1
            cnt += 1
            if draw or draw2:
                if self.debug:
                    print(f"Draw {cnt}")
                for _ in range(cnt):
                    self._draw(self.x, self.y)
                    self.y += 1
            else:
                if self.debug:
                    print(f"Skip {cnt}")
                self.y += cnt
            if self.y > 31:
                self.repeat_column_shift = True
                self.x += 1
                self.y -= 32
            else:
                self.repeat_column_shift = False
            self.last_draw = True
        else:  # Draw extended / Skip extended
            draw = cnt & 0x01
            cnt >>= 1
            draw2 = cnt & 0x01
            cnt >>= 1
            cnt += 1
            cnt += 16
            if draw or draw2:
                if self.debug:
                    print(f"Draw extended {cnt}")
                for _ in range(cnt):
                    self._draw(self.x, self.y)
                    self.y += 1
            else:
                if self.debug:
                    print(f"Skip extended {cnt}")
                self.y += cnt
            if self.y > 31:
                self.repeat_column_shift = True
                self.x += 1
                self.y -= 32
            else:
                self.repeat_column_shift = False
            self.last_draw = False

    def decode(self) -> bytes:
        for i, b in enumerate(self.buffer):
            if self.debug:
                sys.stdout.write(f"{i}: ")
            self._execute(b)
        return self.image


class ImageEncoder:
    def __init__(self, im: Image):
        if im.size[0] != 128 or im.size[1] != 32:
            raise ValueError(
                f"Invalid image dimensions. Only 128x32 images are supported. {im.size[0]}x{im.size[1]} given."
            )
        if im.mode != "1":
            raise ValueError("Invalid pixel format. Only binary images are supported.")
        self.px = im.load()
        self.buffer = bytearray()
        self.last_col = bytearray()
        self.cur_col = bytearray()
        self.skip_cols = 0
        self.repeat_cols = 0
        self.x = 0
        self.y = 0

    def _encode_seq(self, color: int, cnt: int) -> int | None:
        """Encode a sequence of pixels with the same color."""
        cmd = None  # type: Optional[int]
        if color:
            # Draw
            if cnt <= 15:
                cmd = 0x80 + (cnt << 2) + 0x01
            else:
                assert 15 <= cnt <= 31
                cmd = 0xC0 + ((cnt - 16) << 2) + 0x01
        else:
            # Skip
            if cnt <= 15:
                cmd = 0x80 + (cnt << 2)
            elif cnt < 31:
                cmd = 0xC0 + ((cnt - 16) << 2)
            else:
                # Skip column
                assert cnt == 31
                self.skip_cols += 1
        return cmd

    def _count_color(self, color: int) -> int:
        """Count pixels with the same color, down a column."""
        cnt = 0
        if self.y < 32:
            while self.px[self.x, self.y] == color:
                cnt += 1
                self.y += 1
                if self.y > 31:
                    self.x += 1
                    self.y = 0
                    break
        else:
            self.x += 1
            self.y = 0
        return cnt

    def _skip_cols(self) -> None:
        """Handle columns skipping."""
        if self.skip_cols:
            self.repeat_cols = 0
            self.last_col = bytearray()
            # Optimization: Remove skips, right before column skips.
            if self.buffer:
                cmd = self.buffer[-1]
                if (cmd & 0xC3 == 0x80) or (cmd & 0xC3 == 0xC0):
                    self.buffer.pop()
        while self.skip_cols >= 64:
            self.buffer.append(63)
            self.skip_cols -= 64
        if self.skip_cols:
            cmd = self.skip_cols - 1
            self.buffer.append(cmd)
            self.skip_cols = 0

    def _repeat_cols(self) -> None:
        """Handle column repetition."""
        if self.repeat_cols:
            # Optimization: Remove skips, right before column repeats.
            if self.buffer:
                cmd = self.buffer[-1]
                if (cmd & 0xC3 == 0x80) or (cmd & 0xC3 == 0xC0):
                    self.buffer.pop()
        while self.repeat_cols >= 64:
            cmd = 0x40 + 0x3F
            self.buffer.append(cmd)
            self.repeat_cols -= 64
        if self.repeat_cols:
            cmd = 0x40 + self.repeat_cols - 1
            self.buffer.append(cmd)
            self.repeat_cols = 0

    def encode(self) -> bytearray:
        while self.x < 128 and self.y < 32:
            color = self.px[self.x, self.y]
            self.y += 1
            cnt = self._count_color(color)
            cmd = self._encode_seq(color, cnt)
            if cmd is not None:
                if self.y == 0:
                    self._skip_cols()
                    # TODO: Not clear exactly how the 2 draw bits work...
                    if (cmd & 0xC3 == 0x81) or (cmd & 0xC3 == 0xC1):
                        cmd += 1
                self.cur_col.append(cmd)
            # Handle column repetition
            if self.y == 0:
                if not self.skip_cols:
                    if self.cur_col == self.last_col:
                        self.repeat_cols += 1
                    else:
                        self._repeat_cols()
                        self.buffer.extend(self.cur_col)
                        self.last_col = self.cur_col
                else:
                    self._repeat_cols()
                self.cur_col = bytearray()
        if self.y == 0:
            self._skip_cols()
            self._repeat_cols()
        return self.buffer
