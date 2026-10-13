"""Serialize CSV appends from parallel training workers."""
from contextlib import contextmanager
import os


@contextmanager
def locked_csv_append(path):
    with open(path, 'a', newline='', encoding='utf-8') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            exists = os.fstat(stream.fileno()).st_size > 0
            stream.seek(0, os.SEEK_END)
            yield stream, exists
            stream.flush()
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
