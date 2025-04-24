import sys, ctypes, mmap

if sys.platform == "win32":
    kernel32 = ctypes.windll.kernel32
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualAlloc.argtypes = (
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_ulong,
        ctypes.c_ulong
    )
    MEM_COMMIT               = 0x1000
    MEM_RESERVE              = 0x2000
    PAGE_EXECUTE_READWRITE   = 0x40

def make_asm_func(shellcode: bytes, restype, argtypes):
    size = len(shellcode)

    if sys.platform == "win32":
        ptr = kernel32.VirtualAlloc(
            None,
            size,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_EXECUTE_READWRITE
        )
        if not ptr:
            raise MemoryError("VirtualAlloc failed")
        ctypes.memmove(ptr, shellcode, size)
        addr = ptr

    else:
        # POSIX: mmap an anonymous RWX page
        prot = mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC
        buf = mmap.mmap(-1, size, prot=prot)
        buf.write(shellcode)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))

    return ctypes.CFUNCTYPE(restype, *argtypes)(addr)


_add_shellcode = bytes([
    0x55,                   # push rbp
    0x48, 0x89, 0xE5,       # mov rbp, rsp
    0x89, 0xF8,             # mov eax, edi
    0x01, 0xF0,             # add eax, esi
    0x5D,                   # pop rbp
    0xC3                    # ret
])
add = make_asm_func(_add_shellcode, ctypes.c_int, (ctypes.c_int, ctypes.c_int))

_sub_shellcode = bytes([
    0x55,                   # push rbp
    0x48, 0x89, 0xE5,       # mov rbp, rsp
    0x89, 0xF8,             # mov eax, edi
    0x29, 0xF0,             # sub eax, esi
    0x5D,                   # pop rbp
    0xC3                    # ret
])
sub = make_asm_func(_sub_shellcode, ctypes.c_int, (ctypes.c_int, ctypes.c_int))

_mul_shellcode = bytes([
    0x55,                   # push rbp
    0x48, 0x89, 0xE5,       # mov rbp, rsp
    0x89, 0xF8,             # mov eax, edi
    0x0F, 0xAF, 0xC6,       # imul eax, esi
    0x5D,                   # pop rbp
    0xC3                    # ret
])
mul = make_asm_func(_mul_shellcode, ctypes.c_int, (ctypes.c_int, ctypes.c_int))

_sum_shellcode = bytes([
    0x48,0x31,0xC0,         # xor    rax, rax
    0x48,0x31,0xC9,         # xor    rcx, rcx
    0x48,0x39,0xCA,         # cmp    rcx, rdx
    0x7D,0x0C,              # jge    +12
    0x8B,0x14,0x0F,         # mov    edx, [rdi + rcx*4]
    0x01,0xD0,              # add    eax, edx
    0x48,0xFF,0xC1,         # inc    rcx
    0xEB,0xF4,              # jmp    -12
    0xC3                    # ret
])
sum_array = make_asm_func(_sum_shellcode, ctypes.c_int, (ctypes.POINTER(ctypes.c_int), ctypes.c_int))
