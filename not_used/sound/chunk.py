"""Chunk reader for AudioKinetic WWise SoundBank (.bnk) files.

Provides a minimal Chunk class for parsing chunk-based binary formats
used in Wwise SoundBank files.
"""


class Chunk:
    def __init__(self, file, align=True, bigendian=True, inclheader=False):
        self.closed = False
        self.align = align
        self.bigendian = bigendian
        self._file = file
        self._chunk_name = file.read(4)
        if not inclheader:
            data_size = file.read(4)
            self._size = int.from_bytes(data_size, "big" if bigendian else "little")
        else:
            self._size = 0
            self._size -= 8
        self._start = file.tell()
        self._end = self._start + self._size

    def getname(self):
        return self._chunk_name

    def getsize(self):
        return self._size

    def close(self):
        if not self.closed:
            self.closed = True
            self._file.seek(self._end)
            if self.align and (self._size % 2):
                self._file.seek(1, 1)

    def skip(self):
        self._file.seek(self._end)
        if self.align and (self._size % 2):
            self._file.seek(1, 1)

    def read(self, size=-1):
        remaining = self._end - self._file.tell()
        if size < 0 or size > remaining:
            size = remaining
        return self._file.read(size)

    def seek(self, pos, whence=0):
        if whence == 0:
            self._file.seek(self._start + pos)
        elif whence == 1:
            self._file.seek(pos, 1)
        elif whence == 2:
            self._file.seek(self._end + pos)
        else:
            raise ValueError("bad whence")
        cur = self._file.tell()
        if cur < self._start:
            self._file.seek(self._start)
        elif cur > self._end:
            self._file.seek(self._end)

    def tell(self):
        return self._file.tell() - self._start

    def isatty(self):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
