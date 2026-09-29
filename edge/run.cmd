@echo off
rem run.sh 로 넘기는 얇은 껍데기. PowerShell/cmd 의 기본 진입점이다.
rem
rem   PowerShell 에서 `bash run.sh` 는 쓰지 않는다 - 거기서 bash 는
rem   C:\Windows\System32\bash.exe(WSL 기본 배포판 실행기)로 잡히고,
rem   이 기기의 기본 배포판은 docker-desktop 이라 /bin/bash 조차 없다
rem   (2026-09-22 실측: execvpe /bin/bash failed). 그래서 여기서
rem   Git Bash 를 직접 찾아 준다.
rem
rem   run.cmd                 기본 포트(config server.port)
rem   run.cmd --port 9000
rem   run.cmd stop            남은 엣지를 정리만 한다
rem
rem   Ctrl-C 를 누르면 cmd 가 "Terminate batch job (Y/N)?" 를 한 번 더 묻는다.
rem   그 질문이 뜰 때는 이미 run.sh 의 정리가 끝난 뒤다 - 묻는 건 cmd 사정이고
rem   WSL 쪽 서버와 포트는 그 전에 반환된다.
setlocal
set "SH="
if exist "%ProgramFiles%\Git\bin\bash.exe" set "SH=%ProgramFiles%\Git\bin\bash.exe"
if not defined SH if exist "%ProgramFiles%\Git\usr\bin\bash.exe" set "SH=%ProgramFiles%\Git\usr\bin\bash.exe"
if not defined SH if exist "%ProgramFiles(x86)%\Git\bin\bash.exe" set "SH=%ProgramFiles(x86)%\Git\bin\bash.exe"
if not defined SH if exist "%LocalAppData%\Programs\Git\bin\bash.exe" set "SH=%LocalAppData%\Programs\Git\bin\bash.exe"
if not defined SH (
  echo Git Bash 를 찾지 못했다. Git for Windows 를 설치하거나,
  echo WSL 안에서 직접 ./run.sh 를 실행한다.
  exit /b 1
)
"%SH%" "%~dp0run.sh" %*
exit /b %errorlevel%
