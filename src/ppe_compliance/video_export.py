"""Single-pass browser-compatible video export with checked encoder completion."""

from pathlib import Path
import subprocess


class H264Writer:
    """Accept contiguous BGR frames; report pipe and finalization failures."""

    def __init__(self, path, fps, size):
        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError('Efficient video export needs imageio-ffmpeg; install the project [ui] or [notebook] extras.') from error

        self.path, self.size = Path(path), tuple(size)
        self.frames = 0
        self.process = None
        self.log_path = self.path.with_suffix('.encoder.log')
        self.log = self.log_path.open('wb')
        try:
            self.process = subprocess.Popen([
                imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', f'{size[0]}x{size[1]}',
                '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                '-preset', 'veryfast', '-crf', '18', '-threads', '2', '-pix_fmt', 'yuv420p',
                '-tune', 'fastdecode', '-bf', '0', '-g', str(max(1, round(fps*2))),
                '-movflags', '+faststart', str(self.path)], stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=self.log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except BaseException:
            self.log.close()
            raise

    def write(self, frame):
        if frame.shape != (self.size[1], self.size[0], 3) or frame.dtype.name != 'uint8':
            raise ValueError('Encoder frame dimensions or pixel type changed')
        try:
            self.process.stdin.write(memoryview(frame if frame.flags.c_contiguous else frame.copy()))
            self.frames += 1
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(f'H.264 encoder stopped; see {self.log_path}') from error

    def release(self):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = process.wait(timeout=60)
            except BaseException:
                process.kill()
                process.wait()
                raise
            if code or (self.frames and (not self.path.is_file() or self.path.stat().st_size == 0)):
                raise RuntimeError(f'H.264 export failed (exit {code}); see {self.log_path}')
        finally:
            self.log.close()
