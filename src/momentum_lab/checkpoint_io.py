"""Checkpoint replacement tolerant of temporary Windows file sharing locks."""
import time


def replace_checkpoint(temporary, checkpoint, attempts=21, delay=0.5):
    for attempt in range(attempts):
        try:
            temporary.replace(checkpoint)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
