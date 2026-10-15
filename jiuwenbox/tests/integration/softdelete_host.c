/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Build without a longPathAware manifest to exercise MAX_PATH on any host.
 * cl /utf-8 softdelete_host.c /Fe:softdelete_host.exe
 */
#include <windows.h>
#include <stdio.h>
#include <wchar.h>

int wmain(int argc, wchar_t **argv) {
    if (argc < 3) return 9;
    if (wcscmp(argv[1], L"none") && !LoadLibraryW(argv[1])) return 8;
    if (argc == 4) {
        STARTUPINFOA startup = {sizeof(startup)};
        PROCESS_INFORMATION process = {0};
        char command[32768];
        char environment[] = "FOO=bar\0";
        char empty[] = "\0";
        DWORD code = 1;
        if (!WideCharToMultiByte(CP_ACP, 0, argv[3], -1, command, sizeof(command),
                                NULL, NULL)) return 7;
        if (!CreateProcessA(NULL, command, NULL, NULL, FALSE, 0,
                            wcscmp(argv[2], L"ansi-empty") ? environment : empty,
                            NULL, &startup, &process)) return 6;
        if (WaitForSingleObject(process.hProcess, 20000) != WAIT_OBJECT_0) {
            TerminateProcess(process.hProcess, 5);
        } else {
            GetExitCodeProcess(process.hProcess, &code);
        }
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        return (int)code;
    }
    printf("bare_attrs=%lu\n", GetFileAttributesW(argv[2] + 4));
    HANDLE file = CreateFileW(argv[2], DELETE | FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        NULL, OPEN_EXISTING, 0, NULL);
    printf("open=%d\n", file != INVALID_HANDLE_VALUE);
    if (file == INVALID_HANDLE_VALUE) return 1;
    FILE_DISPOSITION_INFO disposition = {TRUE};
    BOOL ok = SetFileInformationByHandle(file, FileDispositionInfo,
                                        &disposition, sizeof(disposition));
    printf("disposition=%d\n", ok);
    BOOL closed = CloseHandle(file);
    printf("close=%d error=%lu\n", closed, GetLastError());
    return ok && closed ? 0 : 2;
}
