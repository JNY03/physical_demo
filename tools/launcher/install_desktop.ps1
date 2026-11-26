# install_desktop.ps1 — 바탕화면에 **이 저장소의 런처 둘**을 깐다 (2026-11-26).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File \\wsl.localhost\<배포판>\<저장소>\tools\launcher\install_desktop.ps1
#
# 깔리는 것은 저장소에 있는 것뿐이다 — 새 런처를 만들지 않는다.
#   데이터 스트림 실행      launcher.ps1      (WinForms 런처. 소스, 기능, 지정 클래스, 오버레이, 배치)
#   데이터 스트림 웹 런처    web_launcher.cmd  (브라우저만 연다. 웹 런처 서버는 WSL · 컨테이너 안에서 따로 돈다:
#                                            python -m ds_resources.weblauncher)
#
# 하는 일
#   1. %USERPROFILE%\DataStreamLauncher\ 에 그 둘을 복사한다 — launcher.ps1 은 Windows 쪽에 있어야 한다
#      (settings.json · profile.json 을 자기 폴더에 쓰고, 비전이 그 profile.json 을 /mnt/c/... 로 읽는다)
#   2. 그 기기 값(배포판 · 저장소 자리)을 launcher.local.json 에 적는다 — 저장소 파일은 고치지 않는다
#   3. 바탕화면에 바로가기 둘을 만든다
#
# 이 저장소는 리눅스 바탕에서 돈다(fcntl 잠금 · /proc · setsid · 셸 스크립트 · nvidia-smi · kubectl).
# 바로가기는 그 바탕을 부르는 얇은 조각이다 — WinForms 런처는 wsl.exe 로 창을 띄우고, 웹 런처 바로가기는 주소만 연다.
# 네이티브 리눅스 · 컨테이너에서는 바로가기 없이 그대로 쓴다: python -m ds_resources.weblauncher (+ supervisor).
#
# 다시 깔아도 안전하다 — 런처 파일만 덮어쓴다. settings.json · profile.json · launcher.local.json 은 그대로 둔다
# (-Force 를 주면 launcher.local.json 도 다시 쓴다).
param(
    [string]$Dest = (Join-Path $env:USERPROFILE "DataStreamLauncher"),
    [string]$Desktop = [Environment]::GetFolderPath("Desktop"),
    [string]$Distro,
    [string]$RepoDir,
    [string]$PythonWsl = "./venv/bin/python",
    [string]$WebUrl = "http://localhost:10080/",
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot

# ---- 이 스크립트가 어디서 왔나 → 배포판 · 저장소 (\\wsl.localhost\Ubuntu\home\me\repo\tools\launcher)
if (-not $Distro -or -not $RepoDir) {
    $m = [regex]::Match($here, '^\\\\wsl(?:\.localhost|\$)\\([^\\]+)\\(.*)\\tools\\launcher$')
    if ($m.Success) {
        if (-not $Distro) { $Distro = $m.Groups[1].Value }
        if (-not $RepoDir) { $RepoDir = "/" + ($m.Groups[2].Value -replace '\\', '/') }
    }
}
if (-not $Distro) { $Distro = (@(& wsl.exe -l -q) -replace "`0", "" | Where-Object { $_.Trim() } | Select-Object -First 1).Trim() }
if (-not $Distro) { throw "WSL 배포판을 찾지 못했습니다 — -Distro 로 알려 주세요." }
if (-not $RepoDir) { throw "저장소 자리를 찾지 못했습니다 — -RepoDir /home/<사용자>/data_stream 처럼 알려 주세요." }
Write-Host ("배포판 {0} · 저장소 {1}" -f $Distro, $RepoDir)

# ---- 저장소가 거기 있나 (바로가기를 만들고 나서 안 되는 것보다 먼저 막는다)
$probe = (& wsl.exe -d $Distro --cd $RepoDir --exec sh -c "test -f ds_resources/cli.py && echo ok") -join ""
if ($probe -notmatch "ok") { throw "WSL $Distro 의 $RepoDir 에 저장소가 없습니다 (ds_resources/cli.py 를 못 찾음)." }

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
foreach ($f in "launcher.ps1", "web_launcher.cmd") {
    Copy-Item (Join-Path $here $f) (Join-Path $Dest $f) -Force
}
# 부품 (tools/launcher/parts/*.ps1) — 런처가 자기 폴더의 parts\ 에서 불러온다. 저장소에서 지운 부품은 여기서도 지운다
$partsSrc = Join-Path $here "parts"
$partsDst = Join-Path $Dest "parts"
if (Test-Path $partsSrc) {
    New-Item -ItemType Directory -Force -Path $partsDst | Out-Null
    Get-ChildItem $partsDst -Filter "*.ps1" -ErrorAction SilentlyContinue |
        Where-Object { -not (Test-Path (Join-Path $partsSrc $_.Name)) } | Remove-Item -Force
    Copy-Item (Join-Path $partsSrc "*.ps1") $partsDst -Force -ErrorAction SilentlyContinue
}

# ---- 그 기기 값 (저장소 파일은 그대로 두고 여기에만 적는다 — launcher.ps1 의 Pick-Setting 이 읽는다)
$local = Join-Path $Dest "launcher.local.json"
if ($Force -or -not (Test-Path $local)) {
    [ordered]@{
        distro     = $Distro
        repoDir    = $RepoDir
        pythonWsl  = $PythonWsl
        profileWsl = "/mnt/" + $Dest.Substring(0, 1).ToLower() + ($Dest.Substring(2) -replace '\\', '/') + "/profile.json"
    } | ConvertTo-Json | Set-Content -Path $local -Encoding UTF8
}

# ---- 바탕화면 바로가기 둘
$sh = New-Object -ComObject WScript.Shell
function New-Shortcut($name, $target, $arguments, $icon, $desc) {
    $lnk = $sh.CreateShortcut((Join-Path $Desktop ($name + ".lnk")))
    $lnk.TargetPath = $target
    $lnk.Arguments = $arguments
    $lnk.WorkingDirectory = $Dest
    $lnk.Description = $desc
    if ($icon) { $lnk.IconLocation = $icon }
    $lnk.Save()
    Write-Host ("바로가기: {0}" -f (Join-Path $Desktop ($name + ".lnk")))
}
$ps = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$cmd = Join-Path $env:SystemRoot "System32\cmd.exe"
$imageres = Join-Path $env:SystemRoot "System32\imageres.dll"
New-Shortcut "데이터 스트림 실행" $ps `
    ("-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"{0}`"" -f (Join-Path $Dest "launcher.ps1")) `
    "$imageres,109" "소스, 기능, 지정 클래스, 오버레이, 배치를 차례로 고르고 WSL 에서 띄운다"
New-Shortcut "데이터 스트림 웹 런처" $cmd `
    ("/c `"{0}`" {1}" -f (Join-Path $Dest "web_launcher.cmd"), $WebUrl) `
    "$imageres,165" "웹 런처를 브라우저로 연다 (서버는 python -m ds_resources.weblauncher 로 따로 띄운다)"
Write-Host ("깔린 자리: {0}" -f $Dest)
