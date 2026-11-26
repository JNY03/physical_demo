# 부품: 실행 상태 칸 (2026-11-27) — 무엇이 도는지 가시화 웹을 열지 않고 런처 안에서 본다. 「정지」「재시작」도 여기에.
#
# 본문이 부르는 자리 (README 의 Part-Hook 규칙)
#   Part-Hook "StatusPanel" @($부모)   상태 칸을 $부모(TabPage · Panel) 안에 꽉 채워 만든다. 갱신 타이머도 여기서 만든다
#   Part-Hook "StatusStart"            (선택) 「실행」 직후 — 3초를 기다리지 않고 바로 한 번 읽는다
#   Part-Hook "StatusTick"             타이머가 부른다 (본문이 따로 부를 일은 없다)
#   Part-Hook "StopRun" / "RestartRun" 상태 칸의 「정지」「재시작」 단추가 부른다 — 본문 메뉴에서 불러도 된다
#
# 읽는 길: tools/run_status.py (읽기만 · torch 없이). 타이머 안에서는 기다리지 않는다 — wsl.exe 를 숨겨 띄워
#   결과를 이 폴더의 run_status.json 에 쓰게 하고, 다음 틱에 그 파일을 읽는다(창이 멈추지 않게).
#   자체 시험 · 「새로 보기」 처럼 바로 답이 필요한 곳만 본문의 Invoke-Wsl 로 기다려 읽는다.
# 끄기는 새로 짜지 않는다: 본문에 Stop-All 이 있으면 그것을, 없으면 창을 닫을 때와 같은 Stop-Windows 호출을 쓴다.

$Str["status_tab"]      = @{ ko = "실행 상태"; en = "Run status" }
$Str["status_col_src"]  = @{ ko = "소스"; en = "Source" }
$Str["status_col_st"]   = @{ ko = "상태"; en = "State" }
$Str["status_col_in"]   = @{ ko = "입력 fps"; en = "In fps" }
$Str["status_col_proc"] = @{ ko = "처리 fps"; en = "Proc fps" }
$Str["status_col_model"] = @{ ko = "모델"; en = "Models" }
$Str["status_col_where"] = @{ ko = "실행 위치"; en = "Runs on" }
$Str["status_col_why"]  = @{ ko = "마지막 이유"; en = "Last reason" }
$Str["status_stop"]     = @{ ko = "정지"; en = "Stop" }
$Str["status_restart"]  = @{ ko = "재시작"; en = "Restart" }
$Str["status_refresh"]  = @{ ko = "새로 보기"; en = "Refresh" }
$Str["status_idle"]     = @{ ko = "실행 중인 것 없음"; en = "Nothing running" }
$Str["status_reading"]  = @{ ko = "상태를 읽는 중"; en = "Reading status" }
$Str["status_unread"]   = @{ ko = "상태를 못 읽음"; en = "Could not read the status" }
$Str["status_restart_q"] = @{ ko = "데이터 스트림을 끄고 지금 고른 대로 다시 띄웁니다."; en = "This stops the data stream and starts it again with the current choices." }
$Str["status_live"]     = @{ ko = "수신 중"; en = "Receiving" }
$Str["status_ready"]    = @{ ko = "연결됨 · 새 장 없음"; en = "Connected · no new frames" }
$Str["status_down"]     = @{ ko = "안 들어옴"; en = "Not receiving" }
$Str["status_procs"]    = @{ ko = "프로세스"; en = "Processes" }

$script:StatusUi      = $null              # 만든 칸 @{root; list; summary; info; stop; restart; refresh; host}
$script:StatusTimer   = $null
$script:StatusProc    = $null              # 숨겨 띄운 run_status.py (끝나기를 기다리지 않는다)
$script:StatusProcAt  = $null
$script:StatusTicks   = 0
$script:StatusSeen    = [datetime]::MinValue
$script:StatusActive  = $false             # 마지막으로 그린 것이 「도는 중」이었나 — 멈춘 뒤 빈 상태를 한 번 더 그린다
# 결과 파일은 profile.json 옆에 둔다 (WSL 이 쓰고 Windows 가 읽는 자리). 이름이 profile.json 이 아니어도 그것을 덮지 않게
# 폴더만 떼어 쓴다. Windows 쪽 자리는 그 WSL 경로에서 거꾸로 만든다 (/mnt/c/… → C:\…, 그 밖은 \\wsl.localhost\…)
$StatusFileWsl = ($ProfileWsl -replace "[^/]*$", "") + "run_status.json"
$StatusFile    = $(if ($StatusFileWsl -match "^/mnt/([a-zA-Z])/(.*)$") { $Matches[1].ToUpper() + ":\" + ($Matches[2] -replace "/", "\") }
                   else { "\\wsl.localhost\" + $Distro + ($StatusFileWsl -replace "/", "\") })
$StatusEvery   = 3                         # 몇 틱(1초)마다 새로 읽나

function Status-Argv([bool]$toFile) {
    if ($toFile) { return @($PyWsl, "tools/run_status.py", "--out", $StatusFileWsl) }
    return @($PyWsl, "tools/run_status.py", "--json")
}
function Status-Read {
    # 기다려 읽는다 (자체 시험 · 「새로 보기」) — 실패하면 $null
    return (From-Json (Invoke-Wsl (Status-Argv $false)))
}
function Status-Launch {
    # 기다리지 않고 띄운다 — 앞의 것이 아직 돌면 건너뛴다(10초 넘게 걸리면 끊는다)
    if ($null -ne $script:StatusProc -and -not $script:StatusProc.HasExited) {
        if (((Get-Date) - $script:StatusProcAt).TotalSeconds -lt 10) { return }
        try { $script:StatusProc.Kill() } catch { }
    }
    $argv = (@("-d", $Distro, "--cd", "`"$RepoDir`"", "--exec") + @(Status-Argv $true | ForEach-Object { "`"$_`"" })) -join " "
    try {
        $script:StatusProc = Start-Process -FilePath "wsl.exe" -ArgumentList $argv -WindowStyle Hidden -PassThru
        $script:StatusProcAt = Get-Date
    } catch { $script:StatusProc = $null }
}
function Status-Collect {
    # run_status.py 가 새로 쓴 파일이 있으면 읽어 그린다
    if (-not (Test-Path $StatusFile)) { return }
    $t = (Get-Item $StatusFile).LastWriteTime
    if ($t -le $script:StatusSeen) { return }
    $script:StatusSeen = $t
    try { $snap = Get-Content $StatusFile -Raw -Encoding UTF8 | ConvertFrom-Json } catch { $snap = $null }
    if ($null -ne $snap) { Status-Render $snap }
}

function Status-Columns {
    # 실행 위치 · 마지막 이유는 길다 — 잘린 글은 줄에 마우스를 올리면 풍선말로 다 보인다
    return @(@("status_col_src", 64), @("status_col_st", 120), @("status_col_in", 60), @("status_col_proc", 62),
             @("status_col_model", 78), @("status_col_where", 150), @("status_col_why", 280))
}
function Status-StateText($st) {
    switch ([string]$st) { "live" { return (L "status_live") } "ready" { return (L "status_ready") } "down" { return (L "status_down") } }
    return [string]$st
}
function Status-StateColor($st) {
    switch ([string]$st) {
        "live"  { return [System.Drawing.Color]::ForestGreen }
        "ready" { return [System.Drawing.Color]::DarkOrange }
        "down"  { return [System.Drawing.Color]::Firebrick }
    }
    return [System.Drawing.SystemColors]::ControlText
}
function Status-Texts {
    # 언어를 바꾸면 Reg 가 단추 글을 바꾼다 — 열 머리 · 탭 이름은 그릴 때마다 다시 쓴다
    $ui = $script:StatusUi
    if ($null -eq $ui) { return }
    $cols = Status-Columns
    for ($i = 0; $i -lt $cols.Count -and $i -lt $ui.list.Columns.Count; $i++) { $ui.list.Columns[$i].Text = (L $cols[$i][0]) }
    if ($ui.host -is [System.Windows.Forms.TabPage]) { $ui.host.Text = (L "status_tab") }
}
function Status-Render($snap) {
    $ui = $script:StatusUi
    if ($null -eq $ui) { return }
    Status-Texts
    if ($null -eq $snap) { $ui.summary.Text = (L "status_unread"); return }
    $script:StatusActive = [bool]$snap.active
    $ui.list.BeginUpdate()
    $ui.list.Items.Clear()
    foreach ($x in @($snap.sources)) {
        if ($null -eq $x) { continue }
        $it = New-Object System.Windows.Forms.ListViewItem([string]$x.source)
        $it.UseItemStyleForSubItems = $false
        $st = $it.SubItems.Add((Status-StateText $x.state))
        $st.ForeColor = (Status-StateColor $x.state)
        [void]$it.SubItems.Add([string]$x.in_fps_text)
        [void]$it.SubItems.Add([string]$x.proc_fps_text)
        [void]$it.SubItems.Add($(if (@($x.models).Count) { (@($x.models) -join "+") } else { "-" }))
        [void]$it.SubItems.Add([string]$x.where)
        $why = $it.SubItems.Add($(if ($x.reason) { [string]$x.reason } else { "-" }))
        if ($x.reason_kind -in @("refused", "failed")) { $why.ForeColor = [System.Drawing.Color]::Firebrick }
        elseif ($x.reason_kind -in @("waiting", "degraded", "stopped")) { $why.ForeColor = [System.Drawing.Color]::DarkOrange }
        $it.ToolTipText = "{0}`n{1}`n{2}" -f $x.source, $x.where, $x.reason
        [void]$ui.list.Items.Add($it)
    }
    $ui.list.EndUpdate()
    if ($snap.active) {
        $procs = @($snap.processes | ForEach-Object { "{0} {1}" -f $_.role_ko, $_.pid }) -join " · "
        $ui.summary.Text = [string]$snap.orchestrator.text
        $ui.info.Text = "{0}`n{1}: {2}" -f $snap.gpu_text, (L "status_procs"), $procs
    } else {
        $ui.summary.Text = (L "status_idle")
        $ui.info.Text = [string]$snap.gpu_text
    }
    $ui.stop.Enabled = ($script:Running -or [bool]$snap.active)
}

function Hook-StatusPanel($parent) {
    # $parent 안을 꽉 채운다: 위 = 요약 + 단추, 가운데 = 소스 표, 아래 = GPU · 프로세스 줄
    $root = New-Object System.Windows.Forms.Panel
    $root.Dock = "Fill"
    $list = New-Object System.Windows.Forms.ListView
    $list.View = "Details"
    $list.FullRowSelect = $true
    $list.GridLines = $true
    $list.HeaderStyle = "Nonclickable"
    $list.ShowItemToolTips = $true
    $list.Dock = "Fill"
    foreach ($c in (Status-Columns)) { [void]$list.Columns.Add((L $c[0]), $c[1]) }
    $root.Controls.Add($list)

    $top = New-Object System.Windows.Forms.Panel
    $top.Dock = "Top"
    $top.Height = 34
    $summary = New-Object System.Windows.Forms.Label
    $summary.AutoSize = $false
    $summary.AutoEllipsis = $true
    $summary.Location = New-Object System.Drawing.Point(2, 8)
    $summary.Size = New-Object System.Drawing.Size(300, 22)
    $summary.Anchor = "Top, Left, Right"
    $summary.Text = (L "status_idle")
    $top.Controls.Add($summary)
    $btns = New-Object System.Windows.Forms.FlowLayoutPanel
    $btns.FlowDirection = "RightToLeft"
    $btns.Dock = "Right"
    $btns.Width = 290
    $btns.WrapContents = $false
    $stop = Reg (New-Object System.Windows.Forms.Button) "status_stop"
    $restart = Reg (New-Object System.Windows.Forms.Button) "status_restart"
    $refresh = Reg (New-Object System.Windows.Forms.Button) "status_refresh"
    foreach ($b in @($stop, $restart, $refresh)) { $b.AutoSize = $true; $b.Height = 28; [void]$btns.Controls.Add($b) }
    $top.Controls.Add($btns)
    $root.Controls.Add($top)

    $info = New-Object System.Windows.Forms.Label
    $info.AutoSize = $false
    $info.AutoEllipsis = $true
    $info.Dock = "Bottom"
    $info.Height = 42
    $root.Controls.Add($info)
    $list.BringToFront()                   # 채우는 것은 마지막에 자리를 잡아야 위 · 아래 줄을 덮지 않는다
    $parent.Controls.Add($root)

    $stop.Add_Click({ [void](Hook-StopRun) })
    $restart.Add_Click({ [void](Hook-RestartRun) })
    $refresh.Add_Click({ Status-Render (Status-Read) })
    $script:StatusUi = @{ root = $root; list = $list; summary = $summary; info = $info; stop = $stop; restart = $restart
                          refresh = $refresh; host = $parent }
    Status-Texts
    $stop.Enabled = [bool]$script:Running

    if ($null -eq $script:StatusTimer) {
        $script:StatusTimer = New-Object System.Windows.Forms.Timer
        $script:StatusTimer.Interval = 1000
        $script:StatusTimer.Add_Tick({ Hook-StatusTick })
        $script:StatusTimer.Start()
    }
}

function Hook-StatusTick {
    # 1초마다: 숨겨 띄운 것이 남긴 파일을 읽고, 3초마다 새로 띄운다 — 실행 중일 때만(멈춘 직후 빈 상태 한 번은 그린다)
    if ($null -eq $script:StatusUi) { return }
    Status-Collect
    $script:StatusTicks++
    if ($script:StatusTicks % $StatusEvery -ne 0) { return }
    if ($script:Running -or $script:StatusActive) { Status-Launch }
}
function Hook-StatusStart {
    # 「실행」 직후 — 다음 3초를 기다리지 않는다
    $script:StatusTicks = 0
    if ($null -ne $script:StatusUi) { $script:StatusUi.summary.Text = (L "status_reading"); $script:StatusUi.stop.Enabled = $true }
    Status-Launch
}

function Hook-StopRun([bool]$ask = $true) {
    # 창을 닫을 때(「데이터 스트림을 끕니다」)와 같은 끄기 — 다만 런처는 열어 둔 채 「실행」 단추를 되살린다
    if ($ask) {
        $r = [System.Windows.Forms.MessageBox]::Show((L "m_stop"), (L "d_title"), [System.Windows.Forms.MessageBoxButtons]::OKCancel)
        if ($r -ne [System.Windows.Forms.DialogResult]::OK) { return $false }
    }
    $f = Get-Command -Name "Stop-All" -CommandType Function -ErrorAction SilentlyContinue
    if ($f) { & $f } else {
        Save-Settings
        Set-Status (L "s_stopping") $true
        $debounce.Stop()
        Stop-Windows (@(Vision-Names) + @("ds_slice", "ds_server"))
    }
    $script:Running = $false
    $script:Applied = $null
    $run.Visible = $true
    Set-Status ""
    Hook-StatusStart                        # 빈 상태를 바로 한 번 그린다
    return $true
}
function Hook-RestartRun {
    # 끄고 지금 고른 대로 다시 띄운다 (「실행」과 같은 Start-All)
    $r = [System.Windows.Forms.MessageBox]::Show((L "status_restart_q"), (L "d_title"), [System.Windows.Forms.MessageBoxButtons]::OKCancel)
    if ($r -ne [System.Windows.Forms.DialogResult]::OK) { return $false }
    [void](Hook-StopRun $false)
    Start-All
    Hook-StatusStart
    return $true
}

function PartTest-Status {
    # 자체 시험: python 한 장이 읽히는지 · 칸이 창 없이 만들어져 그 한 장과 가짜 한 줄을 그리는지
    $r = [ordered]@{}
    $snap = Status-Read
    $r.parsed = ($null -ne $snap)
    if ($null -ne $snap) {
        $r.schema = [string]$snap.schema
        $r.active = [bool]$snap.active
        $r.sources = @($snap.sources).Count
        $r.gpu_text = [string]$snap.gpu_text
        $r.summary = [string]$snap.summary
    }
    $saved = $script:StatusUi
    $nTexts = $script:Texts.Count                    # 시험용 단추가 Reg 로 남지 않게 — 끝나면 덜어 낸다
    $box = New-Object System.Windows.Forms.TabPage
    $box.Size = New-Object System.Drawing.Size(616, 300)
    $madeTimer = ($null -eq $script:StatusTimer)
    Hook-StatusPanel $box
    $ui = $script:StatusUi
    if ($madeTimer -and $null -ne $script:StatusTimer) { $script:StatusTimer.Stop(); $script:StatusTimer.Dispose(); $script:StatusTimer = $null }
    Status-Render $snap
    $r.rows_real = $ui.list.Items.Count
    $fake = [pscustomobject]@{ active = $true; gpu_text = "GPU 3.2 / 6.0 GB"
        orchestrator = [pscustomobject]@{ text = "오케스트레이터 실행 중" }
        processes = @([pscustomobject]@{ role_ko = "데이터 서버"; pid = 1 })
        sources = @([pscustomobject]@{ source = "robot1"; state = "live"; in_fps_text = "10.0"; proc_fps_text = "2.1"
                                       models = @("targeted"); where = "book GPU — 이 기기 · WSL"; reason = "처리 중"; reason_kind = "ok" }) }
    Status-Render $fake
    $r.columns = @($ui.list.Columns | ForEach-Object { $_.Text }) -join " | "
    $r.row_fake = @($ui.list.Items[0].SubItems | ForEach-Object { $_.Text }) -join " | "
    $r.buttons = @($ui.stop.Text, $ui.restart.Text, $ui.refresh.Text) -join " | "
    $r.info = $ui.info.Text
    $r.tab = $box.Text
    $r.file_wsl = $StatusFileWsl
    $r.file_win = $StatusFile
    $box.Dispose()
    if ($script:Texts.Count -gt $nTexts) { $script:Texts.RemoveRange($nTexts, $script:Texts.Count - $nTexts) }
    $script:StatusUi = $saved
    $script:StatusActive = $false
    return [pscustomobject]$r
}
