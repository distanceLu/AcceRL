"""Bounded, best-effort CPU previews. Never imports torch or owns Replay/IPC data.

Each producer owns a small disk ring and one background encoder. The viewer is
read-only and is deliberately independent of training / GPU job lifecycle.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time

DEFAULTS = dict(enabled=True, real_every=20, imagined_every=100, evaluation_every=20,
                frame_stride=2, max_frames=128, clips_per_producer=12,
                bytes_per_producer=64 * 1024**2, live_seconds=4)
_WRITER = None
_CONFIG = None
_COUNTERS = {}


def atomic_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf8')
    os.replace(temp, path)


def config(root):
    path = Path(root) / 'visualization_config.json'
    value = dict(DEFAULTS)
    if path.exists():
        extra = json.loads(path.read_text(encoding='utf8'))
        if set(extra) - set(value):
            raise ValueError('Unknown visualization configuration key')
        value.update(extra)
    if type(value['enabled']) is not bool:
        raise ValueError('enabled must be boolean')
    limits = dict(real_every=(1,100000), imagined_every=(1,100000), evaluation_every=(1,100000),
                  frame_stride=(1,100), max_frames=(2,256), clips_per_producer=(1,100),
                  bytes_per_producer=(1024**2,1024**3), live_seconds=(1,60))
    for key, (lo, hi) in limits.items():
        if type(value[key]) is not int or not lo <= value[key] <= hi:
            raise ValueError('Invalid visualization '+key)
    return value


def cpu_frame(observation, normalized=False):
    """Copy just two RGB views. CUDA tensors are rejected, never copied/synced."""
    import numpy as np
    from PIL import Image
    if isinstance(observation, dict):
        views = [observation[k] for k in ('full_image', 'wrist_image')]
    else:
        if getattr(getattr(observation, 'device', None), 'type', 'cpu') != 'cpu':
            raise ValueError('Preview accepts only existing CPU images')
        if hasattr(observation, 'detach'):
            observation = observation.detach().numpy()
        a = np.asarray(observation)
        if a.ndim != 4 or a.shape[:2] != (2,3):
            raise ValueError('Expected two CHW views')
        views = [a[v].transpose(1,2,0) for v in (0,1)]
    result = []
    for view in views:
        a = np.asarray(view)
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError('Expected RGB image')
        if normalized:
            a = np.rint(np.clip((a.astype(np.float32)+1)*127.5,0,255)).astype(np.uint8)
        if a.dtype != np.uint8:
            raise ValueError('Simulator preview requires uint8 pixels')
        result.append(np.array(Image.fromarray(a).resize((224,224), Image.Resampling.BILINEAR), copy=True))
    return np.concatenate(result, axis=1)


class NullClip:
    active = False
    def add(self, *args, **kwargs): pass
    def finish(self, **kwargs): pass


NULL = NullClip()


class Clip:
    active = True
    def __init__(self, writer, role, worker, episode, metadata):
        self.writer, self.role = writer, role
        self.producer = role + '-' + hashlib.sha256(str(worker).encode()).hexdigest()[:10]
        self.meta = dict(metadata, role=role, worker=str(worker), episode_id=str(episode),
                         run_id=os.getenv('ACCERL_RUN_ID', 'offline'), started_ns=time.time_ns(),
                         views=['full camera', 'wrist camera'], visualization_only=True)
        self.frames, self.samples = [], []
        self.last_live = 0.
        self.closed = False

    def add(self, observation, step, *, normalized=False, force=False, **metadata):
        if self.closed:
            return
        try:
            stride = 1 if self.role == 'imagined' else self.writer.cfg['frame_stride']
            if not force and step % stride:
                return
            if len(self.frames) >= self.writer.cfg['max_frames']:
                self.meta['frame_limit_reached'] = True
                # Retain the true endpoint, even when the preview was capped.
                if not force:
                    return
                self.frames.pop(); self.samples.pop()
            if self.samples and step == self.samples[-1]['step']:
                self.frames.pop(); self.samples.pop()
            frame = cpu_frame(observation, normalized)
            sample = dict(metadata, step=int(step))
            self.frames.append(frame); self.samples.append(sample)
            now = time.monotonic()
            if now - self.last_live >= self.writer.cfg['live_seconds']:
                self.last_live = now
                self.writer.offer(('live', self.producer, frame, dict(self.meta, sample=sample, captured_ns=time.time_ns())))
        except Exception as error:
            self.writer.failed(error)
            self.closed = True
            self.frames.clear(); self.samples.clear()

    def finish(self, **metadata):
        if self.closed:
            return
        self.closed = True
        self.meta.update(metadata, completed_ns=time.time_ns(), samples=self.samples,
                         encoded_frames=len(self.frames), preview_sampling=True)
        if self.frames:
            self.writer.offer(('clip', self.producer, self.frames, self.meta))
        self.frames, self.samples = [], []


class Writer:
    def __init__(self, root, cfg):
        self.root = Path(root) / 'visualization'
        self.root.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.writer_id = str(os.getpid())+'-'+str(time.time_ns())
        self.jobs = queue.Queue(maxsize=2)
        self.errors, self.dropped, self.written = 0, 0, 0
        self.error = None
        self.busy = False
        self.thread = threading.Thread(target=self._run, name='weaver-preview-cpu', daemon=True)
        self.thread.start()

    def failed(self, error):
        self.errors += 1
        self.error = str(error)[:400]

    def offer(self, task):
        try:
            self.jobs.put_nowait(task)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _health(self, directory):
        atomic_json(directory/'health.json', dict(updated_ns=time.time_ns(), pid=os.getpid(), writer_id=self.writer_id,
                    errors=self.errors, dropped=self.dropped, written=self.written, last_error=self.error,
                    config=self.cfg, note='Preview drops/errors never fail training; counters are process-local'))

    def _run(self):
        while True:
            task = self.jobs.get()
            self.busy = True
            directory = self.root/task[1]
            try:
                directory.mkdir(exist_ok=True)
                if task[0] == 'live':
                    from PIL import Image
                    Image.fromarray(task[2]).save(directory/'latest.tmp.jpg', quality=75)
                    os.replace(directory/'latest.tmp.jpg', directory/'latest.jpg')
                    atomic_json(directory/'latest.json', task[3])
                else:
                    self._encode(directory, task[2], task[3])
                    self.written += 1
                self._prune(directory)
            except Exception as error:
                self.failed(error)
            finally:
                try:
                    self._health(directory)
                except OSError:
                    pass
                self.busy = False
                self.jobs.task_done()

    def _encode(self, directory, frames, metadata):
        from PIL import Image
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            raise RuntimeError('Install imageio-ffmpeg or ffmpeg; live JPEG remains available')
        stem = str(metadata['completed_ns'])+'-'+hashlib.sha256(metadata['episode_id'].encode()).hexdigest()[:8]
        video = directory/(stem+'.mp4')
        temporary = directory/(stem+'.tmp.mp4')
        fps = float(metadata.get('preview_fps', 20/self.cfg['frame_stride']))
        if not 0 < fps <= 120:
            raise ValueError('Invalid preview fps')
        args = [ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                '-s','448x224','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','1',
                '-preset','veryfast','-crf','25','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)]
        try:
            # An owned child only; timeout also bounds a blocked pipe write.
            with (directory/(stem+'.encoder.log')).open('wb') as log:
                process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=log, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                timer = threading.Timer(60, lambda: process.kill() if process.poll() is None else None)
                timer.daemon = True; timer.start()
                try:
                    for frame in frames:
                        process.stdin.write(frame.tobytes())
                    process.stdin.close()
                    if process.wait(timeout=65) != 0:
                        raise RuntimeError('CPU preview encoder failed or timed out')
                finally:
                    timer.cancel()
                    if process.poll() is None:
                        process.kill(); process.wait(timeout=5)
                    if process.stdin and not process.stdin.closed:
                        process.stdin.close()
            if temporary.stat().st_size > min(self.cfg['bytes_per_producer'],16*1024**2):
                raise ValueError('Preview exceeds per-clip disk budget')
            os.replace(temporary, video)
            Image.fromarray(frames[0]).save(directory/(stem+'.jpg'), quality=80)
            metadata = dict(metadata, video=video.name, poster=stem+'.jpg', preview_fps=fps,
                            duration_seconds=len(frames)/fps,
                            playback='uniform preview frames; exact action positions are in samples')
            atomic_json(directory/(stem+'.json'), metadata)  # Commit marker written last.
            (directory/(stem+'.encoder.log')).unlink(missing_ok=True)
        except Exception:
            for path in directory.glob(stem+'.*'):
                path.unlink(missing_ok=True)
            raise

    def _prune(self, directory):
        clips = sorted(directory.glob('[0-9]*.json'), key=lambda p:p.name, reverse=True)
        used = 0
        for index, path in enumerate(clips):
            files = list(directory.glob(path.stem+'.*'))
            size = sum(p.stat().st_size for p in files)
            if index >= self.cfg['clips_per_producer'] or used+size > self.cfg['bytes_per_producer']:
                # Remove commit marker first; a concurrent browser may receive 404.
                path.unlink(missing_ok=True)
                for p in files:
                    p.unlink(missing_ok=True)
            else:
                used += size

    def flush(self, timeout=5):
        deadline = time.monotonic()+timeout
        while self.jobs.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(.02)
        return not self.jobs.unfinished_tasks


def begin(role, worker, episode, **metadata):
    global _WRITER, _CONFIG
    root = os.getenv('ACCERL_EXPERIMENT_ROOT')
    if not root:
        return NULL
    try:
        if _CONFIG is None:
            _CONFIG = config(root)
        if not _CONFIG['enabled']:
            return NULL
        key = (role,str(worker)); count = _COUNTERS.get(key,0); _COUNTERS[key] = count+1
        if count % _CONFIG[role+'_every']:
            return NULL
        if _WRITER is None:
            _WRITER = Writer(root, _CONFIG)
            atexit.register(_WRITER.flush, 3)
        if _WRITER.busy or not _WRITER.jobs.empty():
            _WRITER.dropped += 1
            return NULL
        return Clip(_WRITER, role, worker, episode, metadata)
    except Exception as error:
        # A bad preview config cannot terminate a distributed training stage.
        if _CONFIG is None:
            _CONFIG = dict(DEFAULTS, enabled=False)
        print('WEAVER preview unavailable: '+str(error), flush=True)
        return NULL


def env_metadata(env, info):
    """Only report a state ID if the wrapper exposes the actual cyclic index."""
    result = {k:info[k] for k in ('initial_state_id','initial_state_hash','task_action_horizon') if k in info}
    if 'initial_state_id' not in result:
        states, index = getattr(env,'initial_states',None), getattr(env,'episode_idx',None)
        if states is not None and index is not None and len(states):
            result['initial_state_id'] = (int(index)-1) % len(states)
            result['initial_state_id_source'] = 'wrapper cyclic reset counter'
    return result


def catalog(root):
    entries, live, health = [], [], []
    for directory in sorted((Path(root)/'visualization').glob('*')):
        if directory.is_symlink() or not directory.is_dir():
            continue
        for path in directory.glob('*.json'):
            try:
                if path.is_symlink() or path.stat().st_size > 2*1024**2:
                    continue
                row = json.loads(path.read_text(encoding='utf8'))
                row['producer'] = directory.name
                if path.name == 'health.json': health.append(row)
                elif path.name == 'latest.json':
                    row['image_url'] = '/media/'+directory.name+'/latest.jpg'
                    live.append(row)
                elif 'video' in row:
                    row['video_url'] = '/media/'+directory.name+'/'+row['video']
                    row['poster_url'] = '/media/'+directory.name+'/'+row['poster']
                    row['metadata_url'] = '/media/'+directory.name+'/'+path.name
                    entries.append(row)
            except (OSError, ValueError):
                continue
    return dict(root=str(Path(root).resolve()), observed_ns=time.time_ns(),
                clips=sorted(entries,key=lambda x:x['completed_ns'],reverse=True), live=live, health=health)


def serve(root, port=8765):
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    from urllib.parse import unquote, urlsplit
    import mimetypes
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Experiment directory does not exist')
    media = root/'visualization'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == '/api':
                data = json.dumps(catalog(root), ensure_ascii=False).encode()
                self.send_response(200); self.send_header('Content-Type','application/json; charset=utf-8')
                self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(data)))
                self.end_headers(); self.wfile.write(data); return
            if path == '/':
                file = Path(__file__).with_name('visualization_viewer.html')
            elif path.startswith('/media/'):
                parts = path[len('/media/'):].split('/')
                if len(parts)!=2 or any(not re.fullmatch(r'[A-Za-z0-9_.-]+',p) or p in ('.','..') for p in parts):
                    self.send_error(404); return
                file = media.joinpath(*parts)
                if not file.resolve().is_relative_to(media.resolve()) or file.is_symlink():
                    self.send_error(404); return
                if file.suffix not in ('.jpg','.mp4','.json'):
                    self.send_error(404); return
            else:
                self.send_error(404); return
            try:
                with file.open('rb') as f:
                    size = os.fstat(f.fileno()).st_size
                    start, end, partial = 0, size-1, False
                    header = self.headers.get('Range')
                    if header:
                        m = re.fullmatch(r'bytes=(\d+)-(\d*)',header)
                        if not m or int(m[1]) >= size:
                            self.send_error(416); return
                        start, end = int(m[1]), min(int(m[2]) if m[2] else end,end)
                        if end < start: self.send_error(416); return
                        partial = True
                    self.send_response(206 if partial else 200)
                    self.send_header('Content-Type',mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
                    self.send_header('Cache-Control','no-cache'); self.send_header('Accept-Ranges','bytes')
                    self.send_header('X-Content-Type-Options','nosniff')
                    if partial: self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
                    self.send_header('Content-Length',str(end-start+1)); self.end_headers()
                    f.seek(start); remaining = end-start+1
                    while remaining:
                        data = f.read(min(1024*1024,remaining))
                        if not data: break
                        self.wfile.write(data); remaining -= len(data)
            except FileNotFoundError:
                self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'WEAVER viewer http://127.0.0.1:{server.server_port} | {root}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


def main(argv=None):
    p = argparse.ArgumentParser(description='Read-only WEAVER preview viewer; no GPU/model imports')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--port',type=int,default=8765)
    args = p.parse_args(argv)
    serve(args.root,args.port)


if __name__ == '__main__':
    main()
