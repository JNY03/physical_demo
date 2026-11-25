# 데이터 스트림 런처 — 소스 · 비전 모델 · 지정 클래스(대상 · 근거 · 거리) · 오버레이를 고르고 WSL 에서 띄운다.
# 바탕화면 바로가기에서 실행: powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File launcher.ps1
#
# 2026-11-23 · 다시 짰다(탐지대상_런처_계획_261123.md). 이전 판은 launcher.ps1.bak_261123.
#   * 탭 셋: 스트림 · 지정 클래스 · 오버레이. 지정 클래스 탭은 '지정 클래스 탐지' 모델을 골랐을 때만 켜진다.
#   * 선택지는 vision/cue_catalog.py --json 이 준다(쓸 수 없는 근거는 아예 안 나온다). 실패하면 catalog_cache.json.
#   * **런처는 실행 뒤에도 열려 있다.** 선택을 바꾸면 1초 뒤 바로 반영한다:
#       대상 · 근거 · 거리 · 오버레이 → profile.json 만 고쳐 쓴다(vision_infer.py 가 지켜보다 실행 중에 바꾼다)
#       비전 모델 · 켜기       → 비전 창만 다시 띄운다
#       소스 · 로봇 캠          → 서버(와 비전)를 다시 띄운다
#       슬라이스               → 슬라이스 창만
#   * 창마다 tools/ds_run.sh 이름표를 단다. ✕ 로 닫으면 확인한 뒤 tools/ds_stop.sh 로 전부 끈다.
#     런처가 비정상으로 죽어도 창은 남고, 다시 열면 tools/ds_status.sh 로 '실행 중' 을 이어받는다.
#   * 화면에는 선택지만 둔다(괄호 설명 · 안내 문장 없음). 설명은 data_stream_가이드.md.
# 이전 판의 기록(2026-11-11 ~ 11-15): 360 슬라이스 · a2 · pi11~13 · drone2 소스, 비전 HF_HUB_OFFLINE=1,
#   river 는 남은 GPU 메모리로 기기를 나눠 여러 창(프로세스 하나 실측 최대 ~3.0 GB).
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$VisionDelay  = 5
$SliceDelay   = 3
$SliceV       = -30
$GpuSpareMB   = 300
$Sources      = @("cam360", "drone", "drone2", "robot1", "ep-001", "realsense", "a2", "pi11", "pi12", "pi13")
$Here         = $PSScriptRoot
$SettingsFile = Join-Path $Here "settings.json"
$ProfileFile  = Join-Path $Here "profile.json"
$CatalogCache = Join-Path $Here "catalog_cache.json"
#: 기기마다 다른 WSL 자리 (Docker 계획 D1, 2026-11-25) — 환경변수 DS_WSL_* > launcher.local.json > 기본값.
#:   launcher.local.json (이 폴더, 런처가 고쳐 쓰지 않는다. settings.json 은 마지막 선택을 통째로 다시 쓰므로 따로 둔다)
#:   { "distro": "Ubuntu-26.04", "repoDir": "/home/<사용자>/data_stream", "profileWsl": "/mnt/c/Users/<사용자>/DataStreamLauncher/profile.json" }
#:   기본값: 배포판 Ubuntu-26.04 · 저장소 = WSL 의 $HOME/data_stream · profile = 이 폴더의 profile.json 을 /mnt/<드라이브>/… 로
$LocalFile    = Join-Path $Here "launcher.local.json"
$Local        = $null
if (Test-Path $LocalFile) { try { $Local = Get-Content $LocalFile -Raw -Encoding UTF8 | ConvertFrom-Json } catch { $Local = $null } }
function Pick-Setting([string]$envName, [string]$key, [scriptblock]$default) {
    $v = [Environment]::GetEnvironmentVariable($envName)
    if ($v) { return $v }
    if ($null -ne $Local -and $Local.$key) { return [string]$Local.$key }
    return (& $default)
}
$Distro       = Pick-Setting "DS_WSL_DISTRO" "distro" { "Ubuntu-26.04" }
$RepoDir      = Pick-Setting "DS_WSL_REPO" "repoDir" { ((& wsl.exe -d $Distro --exec sh -c 'printf %s "$HOME"') -join "") + "/data_stream" }
$ProfileWsl   = Pick-Setting "DS_WSL_PROFILE" "profileWsl" { "/mnt/" + $ProfileFile.Substring(0, 1).ToLower() + ($ProfileFile.Substring(2) -replace '\\', '/') }
# 「배치 보기」 전용 — 비전이 지켜보는 profile.json 은 「실행」(Start-All · Apply-Changes) 때만 바뀐다 (M11 R4)
$PreviewFile  = Join-Path $Here "profile_preview.json"
$PreviewWsl   = $ProfileWsl -replace "profile\.json$", "profile_preview.json"
#: 비전 GPU 메모리 어림(MB)은 vision/model_registry.yaml 하나에서 온다 (카탈로그 gpu_mb_base · gpu_mb_option · gpu_mb_targeted)

function Invoke-Wsl([string[]]$argv) {
    $out = & wsl.exe -d $Distro --cd $RepoDir --exec @argv 2>$null
    return ($out -join "`n")
}

# ---- 카탈로그
$cat = $null
try {
    $txt = Invoke-Wsl @("./venv/bin/python", "vision/cue_catalog.py", "--json")
    $cat = $txt | ConvertFrom-Json
    [System.IO.File]::WriteAllText($CatalogCache, $txt, (New-Object System.Text.UTF8Encoding($false)))
} catch { $cat = $null }
if ($null -eq $cat -and (Test-Path $CatalogCache)) {
    try { $cat = Get-Content $CatalogCache -Raw -Encoding UTF8 | ConvertFrom-Json } catch { $cat = $null }
}
if ($null -eq $cat) {
    [void][System.Windows.Forms.MessageBox]::Show("선택지를 못 읽었습니다.", "데이터 스트림 실행")
    exit 1
}
$Models = @($cat.models | ForEach-Object { $_.key })
$ModelShown = @{}; foreach ($m in $cat.models) { $ModelShown[$m.key] = $m.shown }
$TargetDb = @{}; foreach ($t in $cat.targets) { $TargetDb[$t.name] = $t }

# ---- 마지막 선택
$s = [pscustomobject]@{ sources = $Sources; robotCam = 1; vision = $false; models = @("yoloe"); orchestrate = $true;
                        slice = $false; sliceU = ""; targets = $null; direct = @(); distance = $cat.distance_default;
                        cueMode = $null; cues = $null; overlay = $null; features = $null }
if (Test-Path $SettingsFile) {
    try {
        $saved = Get-Content $SettingsFile -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($k in "sources", "robotCam", "vision", "orchestrate", "models", "slice", "sliceU", "targets", "direct", "distance",
                       "cueMode", "cues", "overlay", "features") {
            if ($null -ne $saved.$k) { $s.$k = $saved.$k }
        }
    } catch { }
}
function To-Hash($o) { $h = @{}; if ($null -ne $o) { foreach ($p in $o.PSObject.Properties) { $h[$p.Name] = $p.Value } }; return $h }
$script:Direct   = [System.Collections.ArrayList]@()     # 직접 입력 대상 {name, size, colour}
foreach ($d in @($s.direct)) { if ($d -and $d.name) { [void]$script:Direct.Add(@{ name = [string]$d.name; size = [string]$d.size; colour = [string]$d.colour }) } }
$script:CueMode  = To-Hash $s.cueMode                    # 대상 → "default" | "custom"
$script:Cues     = @{}                                   # 대상 → 고른 근거 키 목록
foreach ($kv in (To-Hash $s.cues).GetEnumerator()) { $script:Cues[$kv.Key] = @($kv.Value) }
$script:Overlay  = @{}                                   # 모델 → 사용자가 바꾼 오버레이(바꾼 모델만)
foreach ($kv in (To-Hash $s.overlay).GetEnumerator()) { $script:Overlay[$kv.Key] = To-Hash $kv.Value }
$script:TargetsTouched = ($null -ne $s.targets)
$script:Running  = $false
$script:Applied  = $null                                 # 반영된 실행 구성(비교용)
$script:Building = $false

# ---- 창
$font = New-Object System.Drawing.Font("Malgun Gothic", 10)
$form = New-Object System.Windows.Forms.Form
$form.Text = "데이터 스트림 실행"
$form.Font = $font
$form.StartPosition = "CenterScreen"
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false
$form.ClientSize = New-Object System.Drawing.Size(640, 640)

$tabs = New-Object System.Windows.Forms.TabControl
$tabs.Location = New-Object System.Drawing.Point(8, 8)
$tabs.Size = New-Object System.Drawing.Size(624, 572)
$form.Controls.Add($tabs)
function New-Tab($text) { $p = New-Object System.Windows.Forms.TabPage; $p.Text = $text; $tabs.TabPages.Add($p); $p }
$tStream = New-Tab "스트림"
$tTarget = New-Tab "지정 클래스"
$tOver   = New-Tab "오버레이"

function New-Group($parent, $text, $x, $y, $w, $h) {
    $g = New-Object System.Windows.Forms.GroupBox
    $g.Text = $text
    $g.Location = New-Object System.Drawing.Point($x, $y)
    $g.Size = New-Object System.Drawing.Size($w, $h)
    $parent.Controls.Add($g)
    $g
}
function New-Check($parent, $text, $x, $y, $checked) {
    $c = New-Object System.Windows.Forms.CheckBox
    $c.Text = $text
    $c.AutoSize = $true
    $c.Location = New-Object System.Drawing.Point($x, $y)
    $c.Checked = [bool]$checked
    $parent.Controls.Add($c)
    $c
}
function New-Label($parent, $text, $x, $y) {
    $l = New-Object System.Windows.Forms.Label
    $l.Text = $text
    $l.AutoSize = $true
    $l.Location = New-Object System.Drawing.Point($x, $y)
    $parent.Controls.Add($l)
    $l
}

# ================================================================ 스트림 탭
$gStream = New-Group $tStream "데이터 스트림" 8 8 596 170
[void](New-Label $gStream "소스" 14 30)
$srcChecks = @{}
$i = 0
foreach ($n in $Sources) {
    $x = 80 + ($i % 6) * 84
    $y = 28 + [Math]::Floor($i / 6) * 26
    $srcChecks[$n] = New-Check $gStream $n $x $y ($s.sources -contains $n)
    $i++
}
[void](New-Label $gStream "로봇 캠" 14 96)
$cam = New-Object System.Windows.Forms.ComboBox
$cam.DropDownStyle = "DropDownList"
$cam.Location = New-Object System.Drawing.Point(80, 92)
$cam.Width = 60
1..5 | ForEach-Object { [void]$cam.Items.Add($_) }
$cam.SelectedItem = [int]$s.robotCam
if ($cam.SelectedIndex -lt 0) { $cam.SelectedIndex = 0 }
$gStream.Controls.Add($cam)
[void](New-Label $gStream "슬라이스" 14 134)
$sliceOn = New-Check $gStream "켜기" 80 132 $s.slice
[void](New-Label $gStream "u" 160 134)
$sliceU = New-Object System.Windows.Forms.TextBox
$sliceU.Location = New-Object System.Drawing.Point(178, 130)
$sliceU.Width = 55
$sliceU.Text = [string]$s.sliceU
$gStream.Controls.Add($sliceU)
[void](New-Label $gStream ("v {0}" -f $SliceV) 246 134)

$gVision = New-Group $tStream "비전 추론" 8 186 596 146
$visOn = New-Check $gVision "켜기" 14 28 $s.vision
# 자원 배분 — 켜면 비전을 오케스트레이터(python -m ds_resources.cli run)로 띄운다: 자원에 맞춰 배치 · GPU 여유 확인 ·
# fps 상한 · k3s 가 있으면 파이까지. 끄면 예전처럼 창을 직접 나눠 띄운다 (2026-11-24)
$orchOn = New-Check $gVision "자원 배분" 80 28 $(if ($null -ne $s.orchestrate) { [bool]$s.orchestrate } else { $true })
$planBtn = New-Object System.Windows.Forms.Button
$planBtn.Text = "배치 보기"
$planBtn.Location = New-Object System.Drawing.Point(208, 24)
$planBtn.Width = 90
$gVision.Controls.Add($planBtn)
[void](New-Label $gVision "모델" 14 62)
$modelChecks = @{}
$x = 80
$i = 0
foreach ($n in $Models) {
    $mx = 80 + ($i % 4) * 128
    $my = 60 + [Math]::Floor($i / 4) * 26
    if ($n -eq "targeted") { $mx = 80 + ($i % 4) * 128 }
    $modelChecks[$n] = New-Check $gVision $ModelShown[$n] $mx $my ($s.models -contains $n)
    $i++
}
# 선택지마다 「단서 → 모델」 — vision/model_registry.yaml (카탈로그 options)
$modelTip = New-Object System.Windows.Forms.ToolTip
$modelTip.AutoPopDelay = 30000
function Option-Tip($m) {
    $lines = @("단서 → 모델")
    foreach ($r in @($cat.options.$m)) {
        $ms = @($r.models | ForEach-Object { if ($_.available) { $_.key } else { "{0} (없음)" -f $_.key } })
        $lines += ("{0}: {1}" -f $r.label, $(if ($ms.Count) { $ms -join ", " } else { "규칙 (모델 없음)" }))
    }
    $add = [int]$cat.gpu_mb_option.$m
    if ($m -eq "targeted") { $lines += "GPU 어림: 고른 단서의 모델 합" } else { $lines += ("GPU 어림: +{0} MB" -f $add) }
    return ($lines -join "`n")
}
foreach ($n in $Models) { if ($null -ne $cat.options) { $modelTip.SetToolTip($modelChecks[$n], (Option-Tip $n)) } }
# 기능 — 모델과 따로 켜는 것(카탈로그 features). 지금은 객체 재식별
[void](New-Label $gVision "기능" 14 116)
$featChecks = @{}
$fx = 80
foreach ($f in @($cat.features)) {
    $on = $false
    if ($null -ne $s.features -and $null -ne $s.features.($f.key)) { $on = [bool]$s.features.($f.key) }
    $featChecks[$f.key] = New-Check $gVision $f.shown $fx 114 $on
    if (-not $f.available) { $featChecks[$f.key].Checked = $false; $featChecks[$f.key].Enabled = $false }
    $fx += 128
}

# ================================================================ 지정 클래스 탭
$gT = New-Group $tTarget "대상" 8 8 596 118
$targetFlow = New-Object System.Windows.Forms.FlowLayoutPanel
$targetFlow.Location = New-Object System.Drawing.Point(10, 22)
$targetFlow.Size = New-Object System.Drawing.Size(576, 58)
$targetFlow.AutoScroll = $true
$gT.Controls.Add($targetFlow)
$directBox = New-Object System.Windows.Forms.TextBox
$directBox.Location = New-Object System.Drawing.Point(12, 84)
$directBox.Width = 200
$gT.Controls.Add($directBox)
$directAdd = New-Object System.Windows.Forms.Button
$directAdd.Text = "추가"
$directAdd.Location = New-Object System.Drawing.Point(218, 82)
$directAdd.Size = New-Object System.Drawing.Size(60, 26)
$gT.Controls.Add($directAdd)
$distOn = New-Check $gT "거리 측정" 300 85 $s.distance
if (-not $cat.distance_available) { $distOn.Checked = $false; $distOn.Enabled = $false }

$cuePanel = New-Object System.Windows.Forms.Panel
$cuePanel.Location = New-Object System.Drawing.Point(8, 132)
$cuePanel.Size = New-Object System.Drawing.Size(600, 408)
$cuePanel.AutoScroll = $true
$tTarget.Controls.Add($cuePanel)

$script:TargetChecks = @{}
$script:PendingCheck = $null
if ($script:TargetsTouched) { $script:InitialTargets = @($s.targets) }
else { $script:InitialTargets = @($cat.targets | Where-Object { $_.default } | ForEach-Object { $_.name }) }

function Target-Names {
    $n = @($cat.targets | ForEach-Object { $_.name })
    foreach ($d in $script:Direct) { if ($n -notcontains $d.name) { $n += $d.name } }
    return $n
}
function Is-Direct($name) { return ($null -ne ($script:Direct | Where-Object { $_.name -eq $name } | Select-Object -First 1)) }
function Get-Direct($name) { return ($script:Direct | Where-Object { $_.name -eq $name } | Select-Object -First 1) }

function Cue-List($name) {
    # 이 대상에서 쓸 수 있는 근거 [{key,label,default,needs_distance}]
    if (-not (Is-Direct $name)) { return @($TargetDb[$name].cues) }
    $d = Get-Direct $name
    $out = @()
    foreach ($c in $cat.direct.always) { $out += [pscustomobject]@{ key = $c.key; label = $c.label; default = $true; needs_distance = $c.needs_distance } }
    if ($d.size) { foreach ($c in $cat.direct.with_size) { $out += [pscustomobject]@{ key = $c.key; label = $c.label; default = $true; needs_distance = $c.needs_distance } } }
    if ($d.colour) { foreach ($c in $cat.direct.with_colour) { $out += [pscustomobject]@{ key = $c.key; label = $c.label; default = $true; needs_distance = $c.needs_distance } } }
    return $out
}

function Build-TargetChecks {
    # 지금 체크된 것을 지키며 다시 그린다(처음에는 마지막 선택). $script:PendingCheck = 방금 추가한 직접 입력
    $keep = if ($script:TargetChecks.Count -gt 0) { @(Checked-Targets) } else { @($script:InitialTargets) }
    if ($script:PendingCheck) { $keep += $script:PendingCheck; $script:PendingCheck = $null }
    $script:Building = $true
    $targetFlow.Controls.Clear()
    $script:TargetChecks = @{}
    foreach ($n in (Target-Names)) {
        $c = New-Object System.Windows.Forms.CheckBox
        $c.Text = $n
        $c.AutoSize = $true
        $c.Checked = ($keep -contains $n)
        $c.Add_CheckedChanged({ On-TargetCheck $this })
        $targetFlow.Controls.Add($c)
        $script:TargetChecks[$n] = $c
    }
    $script:Building = $false
}
function On-TargetCheck($cb) {
    if ($script:Building) { return }
    $script:TargetsTouched = $true
    $name = $cb.Text
    if (-not $cb.Checked -and (Is-Direct $name)) {        # 직접 입력은 체크를 풀면 목록에서 사라진다
        [void]$script:Direct.Remove((Get-Direct $name))
        Build-TargetChecks
    }
    Build-CueBoxes; Build-Overlay; Changed
}

function Checked-Targets { return @(Target-Names | Where-Object { $script:TargetChecks[$_] -and $script:TargetChecks[$_].Checked }) }

function Build-CueBoxes {
    $script:Building = $true
    $cuePanel.SuspendLayout()
    $cuePanel.Controls.Clear()
    $y = 4
    foreach ($name in (Checked-Targets)) {
        $list = @(Cue-List $name)
        $direct = Is-Direct $name
        $rows = [Math]::Ceiling(($list.Count) / 4)
        $h = 54 + 26 * $rows + ($(if ($direct) { 30 } else { 0 }))
        $g = New-Object System.Windows.Forms.GroupBox
        $g.Text = $name
        $g.Location = New-Object System.Drawing.Point(4, $y)
        $g.Size = New-Object System.Drawing.Size(568, $h)
        $cuePanel.Controls.Add($g)
        $mode = $script:CueMode[$name]; if (-not $mode) { $mode = "default" }
        foreach ($rm in @(@("default", "기본값", 12), @("custom", "직접 고르기", 100))) {
            $r = New-Object System.Windows.Forms.RadioButton
            $r.Text = $rm[1]; $r.AutoSize = $true; $r.Location = New-Object System.Drawing.Point($rm[2], 22)
            $r.Checked = ($mode -eq $rm[0]); $r.Tag = @{ target = $name; mode = $rm[0] }
            $r.Add_CheckedChanged({ On-CueMode $this })
            $g.Controls.Add($r)
        }
        $chosen = $script:Cues[$name]
        $i = 0
        foreach ($c in $list) {
            $cx = 12 + ($i % 4) * 138
            $cy = 50 + [Math]::Floor($i / 4) * 26
            $on = [bool]$c.default
            if ($mode -eq "custom" -and $null -ne $chosen) { $on = ($chosen -contains $c.key) }
            $cb = New-Check $g $c.label $cx $cy $on
            $cb.Tag = @{ target = $name; key = $c.key; cue = $true }
            if ($mode -eq "custom" -and ($on -ne [bool]$c.default)) { $cb.Font = New-Object System.Drawing.Font($font, [System.Drawing.FontStyle]::Bold) }
            $cb.Enabled = ($mode -eq "custom") -and -not ($c.needs_distance -and -not $distOn.Checked)
            if ($c.key -eq "class") { $cb.Enabled = $false; $cb.Checked = $true }      # 후보를 내는 근거 — 끌 수 없다
            $cb.Add_CheckedChanged({ On-CueCheck $this })
            $i++
        }
        if ($direct) {
            $d = Get-Direct $name
            $ly = 50 + 26 * $rows + 2
            [void](New-Label $g "크기" 12 ($ly + 3))
            $sz = New-Object System.Windows.Forms.TextBox
            $sz.Location = New-Object System.Drawing.Point(52, $ly); $sz.Width = 110; $sz.Text = $d.size; $sz.Tag = $name
            $sz.Add_Leave({ On-DirectSize $this })
            $g.Controls.Add($sz)
            [void](New-Label $g "색" 176 ($ly + 3))
            $col = New-Object System.Windows.Forms.ComboBox
            $col.DropDownStyle = "DropDownList"; $col.Location = New-Object System.Drawing.Point(200, $ly); $col.Width = 100
            [void]$col.Items.Add("")
            foreach ($cn in $cat.colours) { [void]$col.Items.Add($cn) }
            $col.SelectedItem = $d.colour; $col.Tag = $name
            $col.Add_SelectedIndexChanged({ On-DirectColour $this })
            $g.Controls.Add($col)
            $reg = New-Object System.Windows.Forms.Button
            $reg.Text = "등록으로 저장"; $reg.Location = New-Object System.Drawing.Point(420, ($ly - 2)); $reg.Size = New-Object System.Drawing.Size(130, 26)
            $reg.Tag = $name
            $reg.Add_Click({ On-Register $this })
            $g.Controls.Add($reg)
        }
        $y += $h + 6
    }
    $cuePanel.ResumeLayout()
    $script:Building = $false
}
function Cue-Selection($target) {
    $sel = @()
    foreach ($g in $cuePanel.Controls) {
        foreach ($c in $g.Controls) {
            if ($c -is [System.Windows.Forms.CheckBox] -and $c.Tag -and $c.Tag.cue -and $c.Tag.target -eq $target -and $c.Checked) { $sel += $c.Tag.key }
        }
    }
    return $sel
}
function On-CueCheck($cb) {
    if ($script:Building) { return }
    $script:Cues[$cb.Tag.target] = @(Cue-Selection $cb.Tag.target)
    $script:TargetsTouched = $true
    Build-CueBoxes; Changed
}
function On-CueMode($r) {
    if ($script:Building -or -not $r.Checked) { return }
    $t = $r.Tag.target
    if ($r.Tag.mode -eq "custom" -and $null -eq $script:Cues[$t]) { $script:Cues[$t] = @(Cue-Selection $t) }
    $script:CueMode[$t] = $r.Tag.mode
    $script:TargetsTouched = $true
    Build-CueBoxes; Changed
}
function On-DirectSize($tb) {
    $d = Get-Direct $tb.Tag
    $v = $tb.Text.Trim().ToLower()
    if ($null -eq $d -or $v -eq $d.size) { return }
    if ($v -and $v -notmatch '^\d+(\.\d+)?x\d+(\.\d+)?x\d+(\.\d+)?$') {
        [void][System.Windows.Forms.MessageBox]::Show("크기는 가로x세로x높이 mm (예: 300x300x700)", "데이터 스트림 실행"); return
    }
    $d.size = $v; Build-CueBoxes; Changed
}
function On-DirectColour($cb) {
    if ($script:Building) { return }
    $d = Get-Direct $cb.Tag
    if ($null -eq $d) { return }
    $d.colour = [string]$cb.SelectedItem; Build-CueBoxes; Changed
}
function On-Register($b) {
    $d = Get-Direct $b.Tag
    $argv = @("./venv/bin/python", "vision/cue_catalog.py", "--save-target", $d.name)
    if ($d.size) { $argv += @("--size", $d.size) }
    if ($d.colour) { $argv += @("--colours", $d.colour) }
    try { $r = (Invoke-Wsl $argv) | ConvertFrom-Json } catch { $r = [pscustomobject]@{ ok = $false; error = "저장 실패" } }
    if ($r.ok) { [void][System.Windows.Forms.MessageBox]::Show("저장했습니다. 다음 실행부터 목록에 나옵니다.", "데이터 스트림 실행") }
    else { [void][System.Windows.Forms.MessageBox]::Show([string]$r.error, "데이터 스트림 실행") }
}

function Add-Direct($text) {
    $n = $text.Trim().ToLower()
    if ($n -notmatch '^[a-z][a-z0-9 _\-]{0,40}$') {
        [void][System.Windows.Forms.MessageBox]::Show("영문 이름으로 적으세요 (예: traffic cone).", "데이터 스트림 실행"); return
    }
    $directBox.Text = ""
    if ((Target-Names) -contains $n) { $script:TargetChecks[$n].Checked = $true; return }
    [void]$script:Direct.Add(@{ name = $n; size = ""; colour = "" })
    $script:PendingCheck = $n
    $script:TargetsTouched = $true
    Build-TargetChecks; Build-CueBoxes; Build-Overlay; Changed
}
$directAdd.Add_Click({ Add-Direct $directBox.Text })
$distOn.Add_CheckedChanged({ if ($script:Building) { return }; $script:TargetsTouched = $true; Build-CueBoxes; Build-Overlay; Changed })

# ================================================================ 오버레이 탭
$overPanel = New-Object System.Windows.Forms.Panel
$overPanel.Location = New-Object System.Drawing.Point(4, 4)
$overPanel.Size = New-Object System.Drawing.Size(606, 536)
$overPanel.AutoScroll = $true
$tOver.Controls.Add($overPanel)
$LabelItems = @(@("name", "클래스명"), @("conf", "신뢰도"), @("id", "트랙 번호"), @("dist", "거리"), @("gps", "GPS"))

function Chosen-Models { return @($Models | Where-Object { $modelChecks[$_].Checked }) }

function Overlay-Value($m, $key) {
    $o = $script:Overlay[$m]
    if ($o -and $o.ContainsKey($key)) { return $o[$key] }
    return $cat.overlay_defaults.$m.$key
}
function Overlay-Allowed($m, $key) {
    # 고정 가능표는 카탈로그, 선택에 걸린 셋은 여기서: yoloe 거리 = 깊이 모델 함께 · targeted 거리 = 거리 측정 · 관문 = demo_box
    $base = $cat.overlay.$m.$key
    if ($base -ne $true) { return $false }
    $chosen = Chosen-Models
    if ($m -eq "yoloe" -and ($key -eq "dist" -or $key -eq "gps")) { return [bool]($chosen | Where-Object { $_ -eq "unidepth" -or $_ -eq "moge2_aerial" }) }
    if ($m -eq "targeted" -and ($key -eq "dist" -or $key -eq "gps")) { return [bool]$distOn.Checked }
    return $true
}
function Label-Has($m, $tok) {
    $lab = @(Overlay-Value $m "label")
    if ($tok -eq "name") { return ($lab -contains "name" -or $lab -contains "lead") }
    if ($tok -eq "dist") { return ($lab -contains "dist" -or $lab -contains "range" -or $lab -contains "depth") }
    return ($lab -contains $tok)
}

function Mark-Overlay($m, $k, $v) {
    if (-not $script:Overlay.ContainsKey($m)) { $script:Overlay[$m] = @{} }
    $script:Overlay[$m][$k] = $v
}
function Build-Overlay {
    $script:Building = $true
    $overPanel.SuspendLayout()
    $overPanel.Controls.Clear()
    $y = 4
    foreach ($m in (Chosen-Models)) {
        $hasCards = ($cat.overlay.$m.cards -eq $true)
        $h = 112 + ($(if ($hasCards) { 30 } else { 0 }))
        $g = New-Object System.Windows.Forms.GroupBox
        $g.Text = $ModelShown[$m]
        $g.Location = New-Object System.Drawing.Point(4, $y)
        $g.Size = New-Object System.Drawing.Size(578, $h)
        $overPanel.Controls.Add($g)
        $i = 0
        foreach ($it in $LabelItems) {
            if ($cat.overlay.$m.($it[0]) -ne $true) { continue }
            $cb = New-Check $g $it[1] (12 + $i * 92) 24 (Label-Has $m $it[0])
            $cb.Tag = @{ model = $m; kind = "label"; tok = $it[0] }
            $cb.Enabled = (Overlay-Allowed $m $it[0])
            if (-not $cb.Enabled) { $cb.Checked = $false }
            $cb.Add_CheckedChanged({ On-OverLabel $this })
            $i++
        }
        $x = 12
        $fy = 54
        foreach ($k in @(@("count", "탐지 개수"), @("roi", "탐지 영역"), @("show_non_suspect", "판정 아닌 검출"), @("show_rejected", "관문에 떨어진 검출"), @("title", "제목 띠"), @("pose", "드론 위치 띠"))) {
            if ($cat.overlay.$m.($k[0]) -ne $true) { continue }
            if ($k[0] -eq "show_rejected" -and -not ((Checked-Targets) -contains "demo_box")) { continue }
            if ($x -gt 450) { $x = 12; $fy += 26 }
            $cb = New-Check $g $k[1] $x $fy ([bool](Overlay-Value $m $k[0]))
            $cb.Tag = @{ model = $m; kind = "flag"; key = $k[0] }
            $cb.Add_CheckedChanged({ if ($script:Building) { return }; Mark-Overlay $this.Tag.model $this.Tag.key $this.Checked; Changed })
            $x += 130
        }
        if ($hasCards) {
            [void](New-Label $g "카드" 12 114)
            $cx = 60
            $cur = Overlay-Value $m "cards"
            foreach ($cm in $cat.card_modes) {
                $r = New-Object System.Windows.Forms.RadioButton
                $r.Text = $cm.shown; $r.AutoSize = $true; $r.Location = New-Object System.Drawing.Point($cx, 112)
                $r.Checked = ($cur -eq $cm.key); $r.Tag = @{ model = $m; mode = $cm.key }
                $r.Add_CheckedChanged({ if ($script:Building -or -not $this.Checked) { return }; Mark-Overlay $this.Tag.model "cards" $this.Tag.mode; Changed })
                $g.Controls.Add($r)
                $cx += 140
            }
        }
        $y += $h + 6
    }
    $overPanel.ResumeLayout()
    $script:Building = $false
}
function On-OverLabel($cb) {
    if ($script:Building) { return }
    $m = $cb.Tag.model
    $toks = @()
    foreach ($c in $cb.Parent.Controls) {
        if ($c -is [System.Windows.Forms.CheckBox] -and $c.Tag -and $c.Tag.kind -eq "label" -and $c.Checked) {
            $t = $c.Tag.tok
            if ($t -eq "name" -and $m -eq "river") { $t = "lead" }
            $toks += $t
        }
    }
    Mark-Overlay $m "label" $toks
    Changed
}

# ================================================================ 아래
$status = New-Object System.Windows.Forms.Label
$status.AutoSize = $false
$status.Location = New-Object System.Drawing.Point(12, 596)
$status.Size = New-Object System.Drawing.Size(500, 34)
$form.Controls.Add($status)
$run = New-Object System.Windows.Forms.Button
$run.Text = "실행"
$run.Size = New-Object System.Drawing.Size(100, 34)
$run.Location = New-Object System.Drawing.Point(530, 592)
$form.Controls.Add($run)
$form.AcceptButton = $run

function Sync-Enabled {
    $tOn = $visOn.Checked -and $modelChecks["targeted"].Checked
    foreach ($c in $tTarget.Controls) { $c.Enabled = $tOn }
    foreach ($c in $modelChecks.Values) { $c.Enabled = $visOn.Checked }
    foreach ($k in $featChecks.Keys) { $featChecks[$k].Enabled = $visOn.Checked -and [bool](@($cat.features) | Where-Object { $_.key -eq $k -and $_.available }) }
    $overPanel.Enabled = $visOn.Checked
    $orchOn.Enabled = $visOn.Checked
    $planBtn.Enabled = $visOn.Checked
    $sliceU.Enabled = $sliceOn.Checked
}

# ---- 구성 · 프로필
function Current-Config {
    $on = @($Sources | Where-Object { $srcChecks[$_].Checked })
    $onAll = @($on) + @(if ($sliceOn.Checked) { "cam360_slice" })
    return [pscustomobject]@{
        server = (($on -join " ") + "|" + $cam.SelectedItem + "|" + $sliceOn.Checked)
        slice  = ($sliceOn.Checked.ToString() + "|" + $sliceU.Text.Trim())
        vision = ($visOn.Checked.ToString() + "|" + ((Chosen-Models) -join " ") + "|" + ($onAll -join " ") + "|" + (Vision-Groups-Key) + "|" + $orchOn.Checked)
        on = $on; onAll = $onAll
    }
}
function Vision-Groups-Key { if ($modelChecks["targeted"].Checked) { return ((Checked-Targets) -join ",") + ($distOn.Checked) } return "" }

function Build-Profile {
    $pm = @{}
    $tm = @{}
    if ($modelChecks["targeted"].Checked -and $script:TargetsTouched) {
        $ts = @()
        foreach ($n in (Checked-Targets)) {
            $t = [ordered]@{ name = $n }
            if (Is-Direct $n) {
                $d = Get-Direct $n
                if ($d.size) { $t.size_mm = @($d.size -split "x" | ForEach-Object { [double]$_ }) }
                if ($d.colour) { $t.colours = @($d.colour) }
            }
            if ($script:CueMode[$n] -eq "custom" -and $null -ne $script:Cues[$n]) {
                # 지금 쓸 수 있는 근거만 — 거리 측정을 끈 뒤 남은 '크기 거부' 같은 것은 뺀다(비전이 거절한다)
                $ok = @(Cue-List $n | Where-Object { -not ($_.needs_distance -and -not $distOn.Checked) } | ForEach-Object { $_.key })
                $t.cues = @($script:Cues[$n] | Where-Object { $ok -contains $_ })
            }
            $ts += $t
        }
        $tm.targets = $ts
        $tm.distance = [bool]$distOn.Checked
    }
    foreach ($m in (Chosen-Models)) {
        $o = $script:Overlay[$m]
        if (-not $o -or $o.Count -eq 0) { continue }
        $ov = @{}
        foreach ($k in $o.Keys) {
            if ($k -eq "label") {
                $toks = @($o[$k] | Where-Object { Overlay-Allowed $m ($(if ($_ -eq "lead") { "name" } else { $_ })) })
                $ov.label = @(, $toks)
            } elseif ($k -eq "show_rejected" -and -not ((Checked-Targets) -contains "demo_box")) {
            } else { $ov[$k] = $o[$k] }
        }
        if ($m -eq "targeted") { $tm.overlay = $ov }
        else { $pm[$m] = @{ overlay = $ov } }
    }
    if ($tm.Count -gt 0) { $pm["targeted"] = $tm }
    $prof = @{ profile_version = [int]$script:ProfileVer; models = $pm }   # 버전은 Write-Profile 이 올린다
    $feat = @{}
    foreach ($k in $featChecks.Keys) { if ($featChecks[$k].Checked) { $feat[$k] = $true } }
    if ($feat.Count -gt 0) { $prof.features = $feat }
    return $prof
}
$script:ProfileVer = [int]((Get-Date).ToString("HHmmss"))

# 실행 프로필 — 「실행」(Start-All) · 실행 중 반영(Apply-Changes) 만 부른다. 버전은 여기서만 오른다
function Write-Profile {
    $script:ProfileVer = 1 + [int]$script:ProfileVer
    $json = (Build-Profile) | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($ProfileFile, $json, (New-Object System.Text.UTF8Encoding($false)))
}
# 「배치 보기」 프로필 — profile_preview.json 에만 쓴다(버전을 올리지 않는다). 돌고 있는 비전은 이것을 보지 않는다
function Write-PreviewProfile {
    $json = (Build-Profile) | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($PreviewFile, $json, (New-Object System.Text.UTF8Encoding($false)))
}

function Save-Settings {
    $on = @($Sources | Where-Object { $srcChecks[$_].Checked })
    $direct = @($script:Direct | ForEach-Object { [pscustomobject]$_ })
    [pscustomobject]@{ sources = $on; robotCam = [int]$cam.SelectedItem; vision = $visOn.Checked; orchestrate = $orchOn.Checked;
                       models = @(Chosen-Models);
                       slice = $sliceOn.Checked; sliceU = $sliceU.Text.Trim();
                       targets = $(if ($script:TargetsTouched) { @(Checked-Targets) } else { $null });
                       direct = $direct; distance = $distOn.Checked; cueMode = $script:CueMode; cues = $script:Cues;
                       overlay = $script:Overlay;
                       features = $(if ($featChecks.Count) { $h = @{}; foreach ($k in $featChecks.Keys) { $h[$k] = $featChecks[$k].Checked }; $h } else { $null }) } |
        ConvertTo-Json -Depth 8 | Set-Content $SettingsFile -Encoding UTF8
}

# ---- WSL 창 (tools/ds_run.sh 이름표)
function Start-WslWindow($name, $title, $cmd, $envs = "") {
    # 환경변수는 export 로 — 'env A=1 명령' 으로 넘기면 env 가 명령으로 바뀌어 이름표의 명령줄과 달라진다(ds_stop 이 못 끈다)
    $exp = if ($envs) { "export $envs; " } else { "" }
    $inner = "printf '\033]0;$title\007'; trap : INT; ${exp}tools/ds_run.sh $name $cmd"
    Start-Process -FilePath "wsl.exe" -ArgumentList "-d $Distro --cd $RepoDir --exec bash -c `"$inner`""
}
function Stop-Windows([string[]]$names) {
    if (-not $names -or $names.Count -eq 0) { return }
    Start-Process -FilePath "wsl.exe" -ArgumentList ("-d $Distro --cd $RepoDir --exec tools/ds_stop.sh " + ($names -join " ")) -WindowStyle Hidden -Wait
}
function Running-Names {
    $txt = Invoke-Wsl @("tools/ds_status.sh")
    return @($txt -split "`n" | Where-Object { $_ } | ForEach-Object { ($_ -split "`t")[0] })
}

function Set-Status($text) { $status.Text = $text; [System.Windows.Forms.Application]::DoEvents() }

function Check-Inputs($cfg) {
    if ($cfg.on.Count -eq 0) { return "소스를 하나 이상 고르세요." }
    if ($visOn.Checked -and (Chosen-Models).Count -eq 0) { return "비전 모델을 하나 이상 고르세요." }
    if ($visOn.Checked -and $modelChecks["targeted"].Checked -and (Checked-Targets).Count -eq 0) { return "지정 클래스 탐지에는 대상이 하나 이상 필요합니다." }
    if ($sliceOn.Checked -and ($cfg.on -notcontains "cam360")) { return "360 슬라이스는 cam360 소스도 켜야 합니다." }
    $u = $sliceU.Text.Trim()
    if ($sliceOn.Checked -and $u -ne "") {
        $num = [double]0
        if (-not [double]::TryParse($u, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$num)) { return "슬라이스 u 는 숫자로 적으세요." }
    }
    return $null
}

function Start-Server($cfg) {
    Start-WslWindow "ds_server" "데이터 스트림" ("./venv/bin/python server_stream_multi_source.py --on {0} --robot-cam {1}" -f ($cfg.onAll -join " "), $cam.SelectedItem)
}
function Start-Slice {
    $a = ""
    $u = $sliceU.Text.Trim()
    if ($u -ne "") { $a += " --u $u" }
    $a += " --v $SliceV"
    Start-WslWindow "ds_slice" "360 슬라이스" ("./venv/bin/python e2p/slice360_live.py{0}" -f $a)
}
function Targeted-MB {   # targeted 단서 모델 합 (프로세스 기본값 제외) — model_registry.yaml
    $g = $cat.gpu_mb_targeted
    $mb = [int]$g.yoloe_text + [int]$g.yoloe_free + [int]$g.clip + [int]$g.yolo_world
    $ts = Checked-Targets
    if ($ts -contains "demo_door") { $mb += [int]$g.grounding_dino }
    if ($ts -contains "demo_box") { $mb += [int]$g.trained }
    if ($distOn.Checked) { $mb += [int]$g.depth }
    return $mb
}
function Orch-Args($cfg, $profWsl = $ProfileWsl) {
    return ("--sources {0} --models {1} --profile {2}" -f ($cfg.onAll -join " "), ((Chosen-Models) -join " "), $profWsl)
}
function Show-Plan {
    $cfg = Current-Config
    if ($cfg.onAll.Count -eq 0 -or (Chosen-Models).Count -eq 0) { [void][System.Windows.Forms.MessageBox]::Show("소스와 비전 모델을 고르세요.", "배치 보기"); return }
    Write-PreviewProfile
    Set-Status "배치 계산 중"
    $argv = @("./venv/bin/python", "-m", "ds_resources.cli", "plan") + ((Orch-Args $cfg $PreviewWsl) -split " " | Where-Object { $_ })
    $txt = Invoke-Wsl $argv
    Set-Status $(if ($script:Running) { "● 실행 중" } else { "" })
    if (-not $txt) { $txt = "배치를 계산하지 못했습니다 (WSL · venv 확인)." }
    [void][System.Windows.Forms.MessageBox]::Show($txt, "배치 보기")
}
function Start-Vision($cfg) {
    $mods = Chosen-Models
    if ($orchOn.Checked) {
        $env = "HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
        Start-WslWindow "ds_vision_orch" "비전 오케스트레이터" ("./venv/bin/python -m ds_resources.cli run " + (Orch-Args $cfg)) $env
        return
    }
    $procMB = [int]$cat.gpu_mb_base           # 프로세스 하나에 한 번 (torch · CUDA)
    foreach ($m in $mods) { if ($m -eq "targeted") { $procMB += (Targeted-MB) } else { $procMB += [int]$cat.gpu_mb_option.$m } }
    $groups = 1
    $heavy = [bool]($mods | Where-Object { $_ -eq "river" -or $_ -eq "targeted" })
    if ($heavy -and $cfg.onAll.Count -gt 1) {
        try {
            $q = (& nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | Select-Object -First 1) -split ","
            $freeMB = [int]$q[0].Trim() - [int]$q[1].Trim() - $GpuSpareMB
            $groups = [Math]::Max(1, [Math]::Min($cfg.onAll.Count, [Math]::Floor($freeMB / [Math]::Max(1, $procMB))))
        } catch { $groups = 1 }
    }
    $buckets = @(for ($g = 0; $g -lt $groups; $g++) { ,@() })
    for ($i = 0; $i -lt $cfg.onAll.Count; $i++) { $buckets[$i % $groups] += $cfg.onAll[$i] }
    $env = "HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    $k = 0
    foreach ($b in $buckets) {
        $k++
        $title = if ($groups -gt 1) { "비전 추론 ({0})" -f ($b -join " ") } else { "비전 추론" }
        Start-WslWindow ("ds_vision_{0}" -f $k) $title ("./venv/bin/python vision/vision_infer.py --devices {0} --models {1} --profile {2}" -f ($b -join " "), ($mods -join " "), $ProfileWsl) $env
        if ($groups -gt 1) { Start-Sleep -Seconds 3 }
    }
}
function Vision-Names { return @(Running-Names | Where-Object { $_ -like "ds_vision_*" }) }

function Start-All {
    $cfg = Current-Config
    $why = Check-Inputs $cfg
    if ($why) { [void][System.Windows.Forms.MessageBox]::Show($why, "데이터 스트림 실행"); return }
    Save-Settings
    Write-Profile
    Remove-Item -Path (Join-Path $Here "status_*.json") -ErrorAction SilentlyContinue   # 지난 실행의 반영 결과
    Set-Status "띄우는 중"
    Start-Server $cfg
    $waited = 0
    if ($sliceOn.Checked) { Start-Sleep -Seconds $SliceDelay; $waited = $SliceDelay; Start-Slice }
    if ($visOn.Checked) { Start-Sleep -Seconds ([Math]::Max(0, $VisionDelay - $waited)); Start-Vision $cfg }
    $script:Applied = $cfg
    $script:Running = $true
    $run.Visible = $false
    Set-Status "● 실행 중"
}

# ---- 실행 중 반영 (마지막 변경 1초 뒤)
$debounce = New-Object System.Windows.Forms.Timer
$debounce.Interval = 1000
$debounce.Add_Tick({ $debounce.Stop(); Apply-Changes })
function Changed {
    if ($script:Building) { return }
    Sync-Enabled
    if (-not $script:Running) { return }
    $debounce.Stop(); $debounce.Start()
}
function Apply-Changes {
    if (-not $script:Running) { return }
    $cfg = Current-Config
    $why = Check-Inputs $cfg
    if ($why) { Set-Status $why; return }
    Save-Settings
    Write-Profile
    $old = $script:Applied
    if ($cfg.server -ne $old.server) {
        Set-Status "서버를 다시 띄우는 중"
        Stop-Windows (@(Vision-Names) + @("ds_slice", "ds_server"))
        Start-Server $cfg
        if ($sliceOn.Checked) { Start-Sleep -Seconds $SliceDelay; Start-Slice }
        if ($visOn.Checked) { Start-Sleep -Seconds $VisionDelay; Start-Vision $cfg }
    } else {
        if ($cfg.slice -ne $old.slice) {
            Stop-Windows @("ds_slice")
            if ($sliceOn.Checked) { Start-Slice }
        }
        if ($cfg.vision -ne $old.vision) {
            # 대상 · 거리만 바뀐 것은 프로필로 반영된다 — 모델 · 기기 · 켜기가 바뀐 것만 창을 다시 띄운다
            $oldv = $old.vision -split "\|"; $newv = $cfg.vision -split "\|"
            if ($oldv[0] -ne $newv[0] -or $oldv[1] -ne $newv[1] -or $oldv[2] -ne $newv[2] -or $oldv[4] -ne $newv[4]) {
                Set-Status "비전을 다시 띄우는 중"
                Stop-Windows (Vision-Names)
                if ($visOn.Checked) { Start-Vision $cfg }
            }
        }
    }
    $script:Applied = $cfg
    Set-Status "● 실행 중"
}

# ---- 비전이 남긴 반영 결과(틀린 프로필이면 이유)를 상태 줄에
$poll = New-Object System.Windows.Forms.Timer
$poll.Interval = 2000
$poll.Add_Tick({
    if (-not $script:Running) { return }
    $f = Get-ChildItem -Path $Here -Filter "status_*.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $f) { return }
    try { $j = Get-Content $f.FullName -Raw -Encoding UTF8 | ConvertFrom-Json } catch { return }
    if ($j.ok) { if ($status.Text -notlike "*중") { $status.Text = "● 실행 중" } }
    else { $status.Text = "● 실행 중 · 반영 안 됨: " + $j.message }
})
$poll.Start()

# ---- 연결
foreach ($c in $srcChecks.Values) { $c.Add_CheckedChanged({ Changed }) }
$cam.Add_SelectedIndexChanged({ Changed })
$sliceOn.Add_CheckedChanged({ Changed })
$sliceU.Add_Leave({ Changed })
$visOn.Add_CheckedChanged({ Build-Overlay; Changed })
foreach ($c in $modelChecks.Values) { $c.Add_CheckedChanged({ Build-Overlay; Changed }) }
foreach ($c in $featChecks.Values) { $c.Add_CheckedChanged({ Changed }) }
$run.Add_Click({ Start-All })
$orchOn.Add_CheckedChanged({ Changed })
$planBtn.Add_Click({ Show-Plan })
$form.Add_FormClosing({
    param($sender, $e)
    if (-not $script:Running) { return }
    $r = [System.Windows.Forms.MessageBox]::Show("데이터 스트림을 끕니다.", "데이터 스트림 실행",
                                                [System.Windows.Forms.MessageBoxButtons]::OKCancel)
    if ($r -ne [System.Windows.Forms.DialogResult]::OK) { $e.Cancel = $true; return }
    Save-Settings
    Set-Status "끄는 중"
    $poll.Stop(); $debounce.Stop()
    Stop-Windows (@(Vision-Names) + @("ds_slice", "ds_server"))
})

Build-TargetChecks
Build-CueBoxes
Build-Overlay
Sync-Enabled

# ---- 이어받기: 지난번 런처가 띄운 창이 아직 돌고 있으면 '실행 중' 으로
$alive = Running-Names
if ($alive.Count -gt 0) {
    $script:Running = $true
    $script:Applied = Current-Config
    $run.Visible = $false
    $status.Text = "● 실행 중"
}

# ---- 자체 시험 (창을 띄우지 않는다): $env:DS_LAUNCHER_SELFTEST 가 있으면 선택을 코드로 바꿔 보고 프로필을 찍는다
function Find-Ctl($panel, $pred) {
    foreach ($g in $panel.Controls) { foreach ($c in $g.Controls) { if (& $pred $c) { return $c } } }
    return $null
}
function Self-Test {
    $r = [ordered]@{}
    $r.models = $Models
    $r.initial_targets = @(Checked-Targets)
    $r.profile_untouched = Build-Profile
    $r.orchestrate = $orchOn.Checked
    $r.orch_args = Orch-Args (Current-Config)
    $visOn.Checked = $true
    foreach ($m in $Models) { $modelChecks[$m].Checked = ($m -eq "targeted" -or $m -eq "yoloe") }
    $r.target_tab_enabled = $tTarget.Controls[0].Enabled
    # 사용자 설정에 기대지 않는다 — 아래 단계가 쓰는 대상을 직접 켠다 (저장된 대상이 person · chair 뿐이면 door 근거
    # 컨트롤이 없어 시험이 깨졌다, 2026-11-24)
    foreach ($n in @("person", "door", "chair", "desk")) { if ($script:TargetChecks[$n]) { $script:TargetChecks[$n].Checked = $true } }
    Add-Direct "traffic cone"
    $d = Get-Direct "traffic cone"; $d.size = "300x300x700"; $d.colour = "orange"; Build-CueBoxes
    $script:TargetChecks["desk"].Checked = $false
    $rc = Find-Ctl $cuePanel { param($c) $c -is [System.Windows.Forms.RadioButton] -and $c.Tag.target -eq "door" -and $c.Tag.mode -eq "custom" }
    $rc.Checked = $true
    $cb = Find-Ctl $cuePanel { param($c) $c -is [System.Windows.Forms.CheckBox] -and $c.Tag.target -eq "door" -and $c.Tag.key -eq "aspect" }
    $cb.Checked = $false
    $r.door_custom = @(Cue-Selection "door")
    $r.cone_cues = @(Cue-List "traffic cone" | ForEach-Object { $_.key })
    $cnt = Find-Ctl $overPanel { param($c) $c.Tag -and $c.Tag.model -eq "targeted" -and $c.Tag.key -eq "count" }
    $cnt.Checked = $true
    $alw = Find-Ctl $overPanel { param($c) $c.Tag -and $c.Tag.model -eq "targeted" -and $c.Tag.mode -eq "always" }
    $alw.Checked = $true
    $yc = Find-Ctl $overPanel { param($c) $c.Tag -and $c.Tag.model -eq "yoloe" -and $c.Tag.key -eq "count" }
    $yc.Checked = $true
    $roiCb = Find-Ctl $overPanel { param($c) $c.Tag -and $c.Tag.model -eq "targeted" -and $c.Tag.key -eq "roi" }
    $r.roi_checkbox_default = $roiCb.Checked
    $roiCb.Checked = $false
    $ydist = Find-Ctl $overPanel { param($c) $c.Tag -and $c.Tag.model -eq "yoloe" -and $c.Tag.tok -eq "dist" }
    $r.yoloe_dist_enabled_without_depth = $ydist.Enabled
    $distOn.Checked = $false
    $tdist = Find-Ctl $overPanel { param($c) $c.Tag -and $c.Tag.model -eq "targeted" -and $c.Tag.tok -eq "dist" }
    $r.targeted_dist_enabled_after_off = $tdist.Enabled
    $sv = Find-Ctl $cuePanel { param($c) $c -is [System.Windows.Forms.CheckBox] -and $c.Tag.target -eq "chair" -and $c.Tag.key -eq "size_veto" }
    $r.size_veto_enabled_after_off = if ($sv) { $sv.Enabled } else { "none" }
    $r.rejected_box_shown = [bool](Find-Ctl $overPanel { param($c) $c.Tag -and $c.Tag.key -eq "show_rejected" })
    $r.overlay_state = $script:Overlay
    $r.count_ctl = [bool]$cnt
    if ($featChecks["reid"]) { $featChecks["reid"].Checked = $true }
    $r.config = Current-Config
    $r.profile = Build-Profile
    $modelChecks["targeted"].Checked = $false
    $r.target_tab_enabled_after_off = $tTarget.Controls[0].Enabled
    # 「배치 보기」 는 profile.json · 프로필 버전에 손대지 않는다 (M11 R4)
    $snap = { if (Test-Path $ProfileFile) { [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($ProfileFile)) } else { "missing" } }
    $before, $verBefore = (& $snap), $script:ProfileVer
    Write-PreviewProfile
    $r.preview_keeps_profile = ((& $snap) -eq $before)
    $r.preview_keeps_version = ($script:ProfileVer -eq $verBefore)
    $r.preview_written = (Test-Path $PreviewFile)
    $r.plan_args = Orch-Args (Current-Config) $PreviewWsl
    $r | ConvertTo-Json -Depth 10
}
# 끝에서 끝까지(창을 실제로 띄운다): 실행 → 실행 중 거리 끄기 → 전부 끄기. 결과를 한 줄씩 찍는다
function E2E-Test {
    # 소스: DS_E2E_SOURCE (기본 pi13 — 영상이 오는 소스여야 비전 결과까지 본다), 자원 배분: DS_E2E_ORCH (기본 켬, 0 이면 끔)
    $src = if ($env:DS_E2E_SOURCE) { $env:DS_E2E_SOURCE } else { "pi13" }
    # 시험이 사용자 설정 · 프로필을 덮어쓰지 않게 — 끝나면 되돌린다 (Start-All 이 Save-Settings 를 부른다, 2026-11-24)
    $keep = @{}
    foreach ($f in @($SettingsFile, $ProfileFile)) { if (Test-Path $f) { $keep[$f] = [System.IO.File]::ReadAllBytes($f) } }
    $orchOn.Checked = ($env:DS_E2E_ORCH -ne "0")
    "source=$src orchestrate=" + $orchOn.Checked
    foreach ($n in $Sources) { $srcChecks[$n].Checked = ($n -eq $src) }
    $sliceOn.Checked = $false
    $visOn.Checked = $true
    foreach ($m in $Models) { $modelChecks[$m].Checked = ($m -eq "targeted") }
    foreach ($n in (Target-Names)) { if ($script:TargetChecks[$n]) { $script:TargetChecks[$n].Checked = ($n -eq "person" -or $n -eq "chair") } }
    $distOn.Checked = $true
    Start-All
    "started: " + ((Running-Names) -join ",")
    Start-Sleep -Seconds 45
    "after 45s: " + ((Running-Names) -join ",")
    if ($orchOn.Checked) { "orchestrator: " + (Invoke-Wsl @("./venv/bin/python", "-m", "ds_resources.cli", "status")) }
    $distOn.Checked = $false
    Apply-Changes
    "applied distance off; status=" + $status.Text
    Start-Sleep -Seconds 15
    $f = Get-ChildItem -Path $Here -Filter "status_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    "vision status: " + (Get-Content $f.FullName -Raw -Encoding UTF8)
    "still running: " + ((Running-Names) -join ",")
    $modelChecks["yoloe"].Checked = $true
    Apply-Changes
    Start-Sleep -Seconds 5
    "after adding yoloe (vision restarted): " + ((Running-Names) -join ",")
    Start-Sleep -Seconds 40
    if ($orchOn.Checked) { "orchestrator after restart: " + (Invoke-Wsl @("./venv/bin/python", "-m", "ds_resources.cli", "status")) }
    Stop-Windows (@(Vision-Names) + @("ds_slice", "ds_server"))
    Start-Sleep -Seconds 2
    "after stop: [" + ((Running-Names) -join ",") + "]"
    foreach ($f in $keep.Keys) { [System.IO.File]::WriteAllBytes($f, $keep[$f]) }
    "settings restored: " + ($keep.Keys.Count)
}
if ($env:DS_LAUNCHER_SELFTEST) { Self-Test; exit 0 }
if ($env:DS_LAUNCHER_E2E) { E2E-Test; exit 0 }
[void]$form.ShowDialog()
