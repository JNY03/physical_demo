# run.sh 로 넘기는 PowerShell 진입점.
#
# 왜 run.cmd 말고 이것이 있나: cmd 는 Ctrl-C 를 받으면
# "Terminate batch job (Y/N)?" 을 묻고, 대답할 때까지 콘솔을 붙잡는다.
# 엣지는 이미 내려갔는데 프롬프트가 안 돌아와서 "Ctrl-C 로 종료가 안 된다"로
# 보인다(2026-09-29 실측: 포트도 풀리고 WSL 쪽 server.py 도 사라진 뒤였는데
# cmd.exe 와 그 부모 PowerShell 이 그대로 남아 있었다).
# PowerShell 에서 Git Bash 를 직접 부르면 그 층이 없다.
#
#   .\run.ps1                 기본 포트(config server.port)
#   .\run.ps1 --port 9000
#   .\run.ps1 stop            띄우지 않고 남은 엣지를 정리만 한다
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# PowerShell 의 `bash` 는 System32(WSL 기본 배포판 실행기)로 잡힌다. 기본
# 배포판이 Ubuntu 가 아니면 /bin/bash 조차 없어 execvpe 로 죽는다. 그래서
# Git Bash 를 경로로 직접 찾는다.
$candidates = @(
  (Join-Path $env:ProgramFiles 'Git\bin\bash.exe'),
  (Join-Path $env:ProgramFiles 'Git\usr\bin\bash.exe'),
  (Join-Path ${env:ProgramFiles(x86)} 'Git\bin\bash.exe'),
  (Join-Path $env:LocalAppData 'Programs\Git\bin\bash.exe')
)
$sh = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

if (-not $sh) {
  Write-Host "Git Bash 를 찾지 못했다. Git for Windows 를 설치하거나," -ForegroundColor Red
  Write-Host "WSL 안에서 직접 ./run.sh 를 실행한다." -ForegroundColor Red
  exit 1
}

& $sh (Join-Path $here 'run.sh') @args
exit $LASTEXITCODE
