import socket
import contextlib

def find_free_port() -> int:
    """
    利用 socket 绑定到端口 0 的技巧，由操作系统找到一个当前未被使用的临时端口。
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]