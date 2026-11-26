# 부품 30_sources — 소스 연결을 저절로 다시 보고, 안 닿는 이유를 체크 아래에 적고,
# 꺼진 cam360 옆에 「360 카메라 실행」 단추를 둔다 (2026-11-27)
#
# 본문이 부르는 자리
#   Part-Hook "SourcesAuto"                          — 첫 Load-Sources 뒤. 이유 줄을 만들고 시계를 건다
#   Part-Hook "Cam360Button" @($gStream, $x, $y)     — cam360 이 꺼져 있을 때만 보이는 단추 ($x 가 $null 이면 이유 줄 아래)
#
# 왜 뒤에서 도나: source_probe 는 WSL 을 한 번 띄우는 데만 1초 가까이 걸린다. 창 스레드에서 기다리면 그동안 창이 굳는다.
# 그래서 wsl.exe 를 따로 띄우고(출력은 ReadToEndAsync 로 받는다) 1초마다 다 끝났는지만 본다. 끝나면 결과를
# **창 스레드에서**(Forms 시계의 Tick) 바르니 컨트롤을 건드려도 안전하다.
#
# 실행 중에는: 다시 보기는 그대로 한다(source_probe 는 읽기만 한다 — /health · Redis · TCP · /proc). 다만 **체크된 소스는
# 풀지도 막지도 않는다** — 체크를 풀면 Changed → 「실행 중 반영」이 돌아 잠깐 끊긴 소스 하나 때문에 데이터 서버를 다시 띄운다.
# 글(상태 줄 · 이유 줄 · 풍선말)과 체크 안 된 소스의 막기 · 풀기만 바꾼다. 「연결 재확인」 단추(Load-Sources)는 예전 그대로다.

$SrcAutoEverySec = 10          # 몇 초마다 다시 보나
$SrcAutoTickMs   = 1000        # 시계 눈금 — 이 간격으로 결과가 왔는지 본다
$SrcAutoTimeout  = 0.8         # source_probe --timeout (소스 하나에 쓰는 초)
$SrcAutoKillSec  = 20          # 이보다 오래 걸리면 그 확인은 버린다
$SrcAutoCamDelay = 4           # 「360 카메라 실행」 뒤 몇 초 있다 다시 보나
#: 바탕화면 「360 카메라 실행」과 같은 보낼 곳 (tools/launcher/install_desktop.ps1)
$SrcAutoCam360Uploads = @("http://210.110.250.33:7866/upload_pano", "http://127.0.0.1:10001/upload_pano")

$Str["src_reasons"]     = @{ ko = "안 닿는 이유 — "; en = "Why unreachable — " }
$Str["src_cam360_run"]  = @{ ko = "360 카메라 실행"; en = "Start 360 camera" }
$Str["src_cam360_tip"]  = @{ ko = ("cam360 은 이 PC 에서 보내는 쪽(360camera_work/stream_to_server.py)이 돌아야 고를 수 있습니다. " +
                                   "바탕화면 「360 카메라 실행」과 같은 것을 새 창으로 띄우고, 몇 초 뒤 연결을 다시 봅니다.")
                            en = ("cam360 can be picked only while its sender (360camera_work/stream_to_server.py) runs on this PC. " +
                                  "This opens the same window as the desktop shortcut and rechecks the sources a few seconds later.") }
$Str["src_cam360_started"] = @{ ko = "360 카메라를 띄웠습니다 — 곧 연결을 다시 봅니다"; en = "Started the 360 camera — rechecking shortly" }

$script:SrcAutoTimer  = $null
$script:SrcAutoJob    = $null      # @{ proc; task; started }
$script:SrcAutoDue    = [DateTime]::MinValue
$script:SrcAutoFails  = 0
$script:SrcAutoWhy    = $null      # 이유 줄 (Label)
$script:SrcAutoCamBtn = $null
$script:SrcAutoCamAt  = $null      # $null 이면 이유 줄 아래를 따라간다
$script:SrcAutoCamWant = $false    # 단추를 보여야 하나 (cam360 이 down)
$script:SrcAutoCamHold = [DateTime]::MinValue   # 눌린 뒤 이때까지 단추를 막아 둔다

function SrcAuto-Apply($r) {
    # source_probe 결과를 화면에 바른다 — Load-Sources 의 바르는 부분과 같은 규칙:
    # down 이면 막고 체크를 푼다, 아니면 다시 열어 준다. 모름(unknown)·결과에 없는 소스는 건드리지 않는다.
    # 실행 중에는 체크된 소스는 그대로 둔다(머리말). 막기·체크가 하나라도 바뀌었으면 $true
    if ($null -eq $r -or $null -eq $r.sources) { return $false }
    $probe = @{}
    foreach ($x in @($r.sources)) { $probe[$x.source] = $x }
    $script:SrcProbe = $probe
    $live = @(); $down = @()
    $changed = $false
    $script:Building = $true
    try {
        foreach ($n in @($script:Sources)) {
            $x = $probe[$n]
            $c = $srcChecks[$n]
            if ($null -eq $x -or $null -eq $c) { continue }
            $ok = $x.state -ne "down"
            $keep = $script:Running -and $c.Checked          # 실행 중 체크된 것은 그대로
            if (-not $keep) {
                if ($c.Enabled -ne $ok) { $c.Enabled = $ok; $changed = $true }
                if (-not $ok -and $c.Checked) { $c.Checked = $false; $changed = $true }
            }
            $srcTip.SetToolTip($c, ("{0}: {1}" -f $x.state, $x.why))
            if ($x.state -eq "live") { $live += $n } elseif (-not $ok) { $down += $n }
        }
    } finally { $script:Building = $false }
    $script:SrcLive = @($live)
    $script:SrcDown = @($down)
    Set-SrcState
    SrcAuto-Render
    if ($changed) { Build-Match; Changed }
    return $changed
}

function SrcAuto-ReasonText {
    # 「소스: 이유」 — 안 닿는 것만, 한 줄에 하나
    $rows = @()
    foreach ($n in @($script:SrcDown)) {
        $x = $script:SrcProbe[$n]
        if ($null -ne $x) { $rows += ("{0}: {1}" -f $n, $x.why) }
    }
    if ($rows.Count -eq 0) { return "" }
    return ((L "src_reasons") + "`n" + ($rows -join "`n"))
}

function SrcAuto-Render {
    # 이유 줄과 cam360 단추를 지금 결과에 맞추고, 소스 묶음 아래쪽 자리를 다시 잡는다
    $x = $(if ($null -ne $script:SrcProbe) { $script:SrcProbe["cam360"] } else { $null })
    $script:SrcAutoCamWant = ($null -ne $x -and $x.state -eq "down")
    $cb = $script:SrcAutoCamBtn
    if ($cb) {
        $cb.Visible = $script:SrcAutoCamWant
        if (-not $cb.Enabled -and [DateTime]::UtcNow -ge $script:SrcAutoCamHold) { $cb.Enabled = $true }
    }
    if ($null -eq $script:SrcAutoWhy) { return }
    $t = SrcAuto-ReasonText
    $script:SrcAutoWhy.Text = $t
    $script:SrcAutoWhy.Visible = [bool]$t
    SrcAuto-Layout
}

function SrcAuto-Layout {
    # 이유 줄은 상태 글 · 「소스 추가」 단추 아래에 둔다(Move-SrcFoot 가 그 둘을 옮기므로 매번 거기서 잰다)
    if ($null -eq $script:SrcAutoWhy) { return }
    $top = [Math]::Max($srcState.Bottom, $srcAdd.Bottom) + 4
    $script:SrcAutoWhy.Location = New-Object System.Drawing.Point(14, $top)
    # .Visible 은 창이 뜨기 전에는 늘 false 다 — 글이 있는지로 잰다
    $bottom = $top + $(if ($script:SrcAutoWhy.Text) { $script:SrcAutoWhy.PreferredSize.Height } else { 0 })
    $b = $script:SrcAutoCamBtn
    if ($b -and $null -eq $script:SrcAutoCamAt) {
        $b.Location = New-Object System.Drawing.Point(14, ($bottom + 4))
        if ($script:SrcAutoCamWant) { $bottom = $b.Location.Y + $b.Height }
    }
    $gStream.Height = [Math]::Max($gStream.Height, $bottom + 10)
}

function SrcAuto-Start {
    # source_probe 를 따로 띄운다 — 기다리지 않는다
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "wsl.exe"
    $psi.Arguments = ("-d {0} --cd {1} --exec {2} tools/source_probe.py --json --timeout {3}" -f
                      $Distro, $RepoDir, $PyWsl, ([string]$SrcAutoTimeout))
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = New-Object System.Text.UTF8Encoding($false)
    try {
        $p = [System.Diagnostics.Process]::Start($psi)
        # 출력을 바로 읽어 두어야 관이 차서 멈추지 않는다 (소스가 많으면 몇 KB)
        $script:SrcAutoJob = @{ proc = $p; task = $p.StandardOutput.ReadToEndAsync()
                                err = $p.StandardError.ReadToEndAsync(); started = [DateTime]::UtcNow }
    } catch { $script:SrcAutoJob = $null; $script:SrcAutoFails++ }
}

function SrcAuto-Tick {
    $j = $script:SrcAutoJob
    if ($null -ne $j) {
        if ($j.task.IsCompleted -and $j.proc.HasExited) {
            $script:SrcAutoJob = $null
            $txt = $(try { $j.task.Result } catch { "" })
            try { $j.proc.Dispose() } catch {}
            $r = From-Json $txt
            if ($null -eq $r) { $script:SrcAutoFails++ } else { $script:SrcAutoFails = 0; [void](SrcAuto-Apply $r) }
            $script:SrcAutoDue = [DateTime]::UtcNow.AddSeconds($SrcAutoEverySec)
        } elseif (([DateTime]::UtcNow - $j.started).TotalSeconds -gt $SrcAutoKillSec) {
            try { $j.proc.Kill() } catch {}
            $script:SrcAutoJob = $null
            $script:SrcAutoFails++
            $script:SrcAutoDue = [DateTime]::UtcNow.AddSeconds($SrcAutoEverySec)
        }
        return
    }
    if ([DateTime]::UtcNow -lt $script:SrcAutoDue) { return }
    if ($form.WindowState -eq [System.Windows.Forms.FormWindowState]::Minimized) { return }   # 안 보는 동안은 쉰다
    SrcAuto-Start
}

function SrcAuto-Stop {
    if ($script:SrcAutoTimer) { $script:SrcAutoTimer.Stop() }
    $j = $script:SrcAutoJob
    $script:SrcAutoJob = $null
    if ($j) { try { if (-not $j.proc.HasExited) { $j.proc.Kill() } } catch {} }
}

function SrcAuto-Soon($sec) {
    # 다음 확인을 앞당긴다 (이미 도는 확인이 있으면 그것이 끝난 뒤)
    $t = [DateTime]::UtcNow.AddSeconds($sec)
    if ($t -lt $script:SrcAutoDue -or $null -eq $script:SrcAutoJob) { $script:SrcAutoDue = $t }
}

function Hook-SourcesAuto {
    if ($null -eq $script:SrcAutoWhy) {
        $w = New-Object System.Windows.Forms.Label
        $w.AutoSize = $true
        $w.MaximumSize = New-Object System.Drawing.Size(572, 0)          # 길면 줄을 바꾼다
        $w.ForeColor = [System.Drawing.Color]::FromArgb(150, 60, 0)
        $w.Visible = $false
        $gStream.Controls.Add($w)
        $script:SrcAutoWhy = $w
    }
    SrcAuto-Render                                     # 첫 Load-Sources 결과로 바로 채운다
    if ($env:DS_LAUNCHER_SELFTEST -or $env:DS_LAUNCHER_SHOT) { return }   # 시험에서는 시계를 걸지 않는다(결과가 흔들리지 않게)
    if ($null -eq $script:SrcAutoTimer) {
        $script:SrcAutoTimer = New-Object System.Windows.Forms.Timer
        $script:SrcAutoTimer.Interval = $SrcAutoTickMs
        $script:SrcAutoTimer.Add_Tick({ SrcAuto-Tick })
        $form.Add_FormClosed({ SrcAuto-Stop })
    }
    $script:SrcAutoDue = [DateTime]::UtcNow.AddSeconds($SrcAutoEverySec)
    $script:SrcAutoTimer.Start()
}

function SrcAuto-Cam360Cmd {
    $up = @($SrcAutoCam360Uploads | ForEach-Object { "--upload " + $_ }) -join " "
    return ("{0} 360camera_work/stream_to_server.py {1}" -f $PyWsl, $up)
}

function SrcAuto-Cam360Run {
    # 바탕화면 「360 카메라 실행」과 같은 명령을 제 창으로 띄운다. 이름표 ds_cam360 — 런처를 닫아도 끄지 않는다
    # (Mine-Names 에 들지 않는다: 360 카메라는 바탕화면에서 켠 것처럼 따로 돈다)
    Start-WslWindow "ds_cam360" "360 카메라" (SrcAuto-Cam360Cmd)
    Set-Status (L "src_cam360_started")
    $b = $script:SrcAutoCamBtn
    if ($b) { $b.Enabled = $false }                    # 두 번 눌러 둘 띄우지 않게 — 다음 확인 결과가 올 때까지
    $script:SrcAutoCamHold = [DateTime]::UtcNow.AddSeconds($SrcAutoCamDelay + 12)
    SrcAuto-Soon $SrcAutoCamDelay
    if ($null -eq $script:SrcAutoTimer -or -not $script:SrcAutoTimer.Enabled) {
        # 시계가 없으면(훅을 안 걸었으면) 기다렸다가 한 번 직접 본다
        Start-Sleep -Seconds $SrcAutoCamDelay
        Load-Sources
        SrcAuto-Render
        if ($b) { $b.Enabled = $true }
    }
}

function Hook-Cam360Button($parent, $x, $y) {
    if ($null -ne $script:SrcAutoCamBtn) { return }
    $b = New-Object System.Windows.Forms.Button
    $b.AutoSize = $true
    $b.AutoSizeMode = [System.Windows.Forms.AutoSizeMode]::GrowAndShrink
    $b.Padding = New-Object System.Windows.Forms.Padding(6, 0, 6, 0)
    [void](Reg $b "src_cam360_run")
    if ($null -ne $x -and $null -ne $y) {
        $script:SrcAutoCamAt = @{ x = [int]$x; y = [int]$y }
        $b.Location = New-Object System.Drawing.Point([int]$x, [int]$y)
    }
    $srcTip.SetToolTip($b, (L "src_cam360_tip"))
    $b.Add_Click({ SrcAuto-Cam360Run })
    $parent.Controls.Add($b)
    $b.BringToFront()
    $script:SrcAutoCamBtn = $b
    SrcAuto-Render
}

function PartTest-Sources {
    # 가짜 확인 결과를 발라 막기 · 체크가 규칙대로 바뀌는지 본다 (창을 띄우지 않는다). 끝나면 원래대로 돌린다
    $names = @($script:Sources)
    $others = @($names | Where-Object { $_ -ne "cam360" })
    $a = $others[0]; $b = $others[1]; $c = "cam360"
    $save = @{}
    foreach ($n in $names) { $save[$n] = @($srcChecks[$n].Enabled, $srcChecks[$n].Checked) }
    $saveProbe = $script:SrcProbe; $saveLive = $script:SrcLive; $saveDown = $script:SrcDown; $saveRun = $script:Running
    $out = [ordered]@{}
    try {
        $script:Running = $false
        foreach ($n in @($a, $b, $c)) { $srcChecks[$n].Enabled = $true; $srcChecks[$n].Checked = $true }
        $fake1 = [pscustomobject]@{ sources = @(
            [pscustomobject]@{ source = $a; state = "live"; why = "fake live" },
            [pscustomobject]@{ source = $b; state = "down"; why = "fake down" },
            [pscustomobject]@{ source = $c; state = "down"; why = "보내는 런처가 꺼져 있음" }) }
        $out.changed1 = SrcAuto-Apply $fake1
        $out.after_down = [ordered]@{ a = @($srcChecks[$a].Enabled, $srcChecks[$a].Checked)
                                      b = @($srcChecks[$b].Enabled, $srcChecks[$b].Checked)
                                      cam360 = @($srcChecks[$c].Enabled, $srcChecks[$c].Checked) }
        $out.reasons = $(if ($script:SrcAutoWhy) { $script:SrcAutoWhy.Text } else { SrcAuto-ReasonText })
        $out.state_line = $srcState.Text
        $out.cam_button_wanted_when_down = $script:SrcAutoCamWant
        $fake2 = [pscustomobject]@{ sources = @(
            [pscustomobject]@{ source = $b; state = "ready"; why = "fake back" },
            [pscustomobject]@{ source = $c; state = "ready"; why = "보내는 런처 실행 중" }) }
        $out.changed2 = SrcAuto-Apply $fake2
        # 돌아오면 다시 고를 수 있지만 체크는 사람 몫이다(저절로 다시 켜지 않는다)
        $out.after_back = [ordered]@{ b = @($srcChecks[$b].Enabled, $srcChecks[$b].Checked)
                                      cam360 = @($srcChecks[$c].Enabled, $srcChecks[$c].Checked) }
        $out.cam_button_wanted_when_ready = $script:SrcAutoCamWant
        $out.reasons_when_none = $(if ($script:SrcAutoWhy) { $script:SrcAutoWhy.Text } else { SrcAuto-ReasonText })
        # 실행 중: 체크된 소스는 down 이어도 그대로, 체크 안 된 소스는 막는다
        $script:Running = $true
        $srcChecks[$a].Checked = $true; $srcChecks[$b].Checked = $false
        $fake3 = [pscustomobject]@{ sources = @(
            [pscustomobject]@{ source = $a; state = "down"; why = "blip" },
            [pscustomobject]@{ source = $b; state = "down"; why = "blip" }) }
        [void](SrcAuto-Apply $fake3)
        $out.running_checked_kept = @($srcChecks[$a].Enabled, $srcChecks[$a].Checked)
        $out.running_unchecked_disabled = @($srcChecks[$b].Enabled, $srcChecks[$b].Checked)
        $out.cam360_cmd = SrcAuto-Cam360Cmd
        $out.pass = ($out.after_down.a[0] -and $out.after_down.a[1] -and -not $out.after_down.b[0] -and -not $out.after_down.b[1] -and
                     -not $out.after_down.cam360[0] -and $out.cam_button_wanted_when_down -and
                     $out.after_back.b[0] -and -not $out.after_back.b[1] -and -not $out.cam_button_wanted_when_ready -and
                     $out.running_checked_kept[0] -and $out.running_checked_kept[1] -and
                     -not $out.running_unchecked_disabled[0] -and ($out.reasons -match "cam360: "))
    } finally {
        $script:Running = $saveRun
        $script:Building = $true
        foreach ($n in $names) { $srcChecks[$n].Enabled = $save[$n][0]; $srcChecks[$n].Checked = $save[$n][1] }
        $script:Building = $false
        $script:SrcProbe = $saveProbe; $script:SrcLive = $saveLive; $script:SrcDown = $saveDown
        Set-SrcState
        SrcAuto-Render
    }
    return $out
}
