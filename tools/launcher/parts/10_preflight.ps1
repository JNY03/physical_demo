# 시작 점검 배너 (2026-11-27) — 런처를 열 때 이 기기에서 지금 고칠 것을 한 줄씩 보여 준다.
# 문제가 늦게 드러나는 것을 막는다(예: k3s PriorityClass ds-mission 이 빠져 파이 CPU 작업이 막혀 있었다 · 가중치가 없다).
#
# 내용은 `python -m ds_resources.cli preflight --json` 의 issues 한 곳에서 온다 — [{id, level, title, detail, fix}].
#   한 줄 = 점검 하나 (색은 level: error 빨강 · warn 노랑 · info 파랑), 자세한 까닭은 풍선말.
#   fix 가 있으면 「고치기」 → `cli fix <id>` (파이썬이 다시 점검해서 얻은 명령만 돌린다) → 다시 점검.
#     fix.long(가중치 받기처럼 오래 걸림)은 화면을 멈추지 않게 따로 WSL 창에서 돌리고, 끝나면 「다시 점검」.
#     fix.needs_sudo 는 돌리지 않고 명령을 클립보드에 넣는다(WSL 관리자 작업은 사람이 한다).
#   문제가 없으면 초록 한 줄 「점검 이상 없음」으로 접힌다.
#
# 본문이 부르는 자리
#   Part-Hook "PreflightBanner" @($panel [, $preflight])   창을 만들 때 — 빈 패널을 주면 그 안에 그린다.
#       $preflight(이미 읽은 cli preflight --json 결과)를 주면 그것을 쓰고, 없으면 창이 뜬 뒤 한 번 점검한다.
#   Part-Hook "PreflightResult" @($pj)                    Load-Functions 가 preflight 를 읽은 뒤 — 같은 결과를
#       다시 묻지 않게 넘겨 준다(점검을 두 번 돌리지 않는다). 배너가 아직 없으면 결과만 들고 있다가 쓴다.
# 패널 높이는 줄 수에 맞춰 바뀐다(한 줄 26px) — 본문은 $panel.Add_SizeChanged 로 아래 컨트롤을 옮기면 된다.

$Str["pf_checking"] = @{ ko = "시작 점검 중…"; en = "Checking this machine..." }
$Str["pf_ok"]       = @{ ko = "점검 이상 없음"; en = "All checks passed" }
$Str["pf_unknown"]  = @{ ko = "점검 결과를 못 읽었습니다"; en = "Could not read the check result" }
$Str["pf_fix"]      = @{ ko = "고치기"; en = "Fix" }
$Str["pf_recheck"]  = @{ ko = "다시 점검"; en = "Re-check" }
$Str["pf_fixing"]   = @{ ko = "고치는 중…"; en = "Fixing..." }
$Str["pf_fix_fail"] = @{ ko = "고치지 못했습니다"; en = "Could not fix it" }
$Str["pf_sudo"]     = @{ ko = "관리자 권한이 필요합니다. 명령을 클립보드에 넣었습니다 — WSL 에서 직접 실행하세요:";
                         en = "This needs root. The command is on the clipboard - run it in WSL:" }
$Str["pf_long"]     = @{ ko = "새 창에서 돌립니다. 끝나면 「다시 점검」을 누르세요.";
                         en = "Running in a new window. Press Re-check when it is done." }
$Str["pf_more"]     = @{ ko = "외 {0}건"; en = "+{0} more" }

$script:PfPanel   = $null      # 배너를 그릴 패널 (본문이 준다)
$script:PfResult  = $null      # 마지막 preflight 결과 (PSCustomObject)
$script:PfMaxRows = 4          # 이보다 많으면 마지막 줄에 「외 N건」 — 배너는 얇게
$script:PfRowH    = 26
$script:PfTip     = $null

function Pf-Colors($level) {
    # → @(배경, 글자)
    switch ([string]$level) {
        "error" { return @([System.Drawing.Color]::MistyRose, [System.Drawing.Color]::DarkRed) }
        "warn"  { return @([System.Drawing.Color]::LightYellow, [System.Drawing.Color]::SaddleBrown) }
        "info"  { return @([System.Drawing.Color]::AliceBlue, [System.Drawing.Color]::SteelBlue) }
        default { return @([System.Drawing.Color]::Honeydew, [System.Drawing.Color]::DarkGreen) }
    }
}
function Pf-Restore-Status {
    if (Get-Command Set-Status -ErrorAction SilentlyContinue) { Set-Status $(if ($script:Running) { L "s_running" } else { "" }) }
}

function Pf-Row($panel, $i, $w, $level, $text, $tip) {
    # 한 줄: 색 띠 + 글. → 그 줄 패널 (단추는 부르는 쪽이 얹는다)
    $c = Pf-Colors $level
    $row = New-Object System.Windows.Forms.Panel
    $row.Location = New-Object System.Drawing.Point(0, ($i * $script:PfRowH))
    $row.Size = New-Object System.Drawing.Size($w, ($script:PfRowH - 2))
    $row.BackColor = $c[0]
    $lb = New-Object System.Windows.Forms.Label
    $lb.AutoSize = $false
    $lb.AutoEllipsis = $true
    $lb.Location = New-Object System.Drawing.Point(6, 3)
    $lb.Size = New-Object System.Drawing.Size(($w - 182), ($script:PfRowH - 6))
    $lb.ForeColor = $c[1]
    $lb.Text = $text
    $row.Controls.Add($lb)
    if ($tip) { $script:PfTip.SetToolTip($lb, [string]$tip) }
    $panel.Controls.Add($row)
    return $row
}
function Pf-Button($row, $text, $x, $w, $tag, [scriptblock]$onClick) {
    $b = New-Object System.Windows.Forms.Button
    $b.Text = $text
    $b.Location = New-Object System.Drawing.Point($x, 0)
    $b.Size = New-Object System.Drawing.Size($w, ($script:PfRowH - 2))
    $b.FlatStyle = "Flat"
    $b.FlatAppearance.BorderSize = 1
    $b.BackColor = [System.Drawing.Color]::White
    $b.Tag = $tag
    $b.Add_Click($onClick)
    $row.Controls.Add($b)
    return $b
}

function Pf-Render($panel = $script:PfPanel, $result = $script:PfResult, [switch]$Checking) {
    # 배너를 다시 그린다 → @{ rows; fixes; level } (시험이 센다)
    $out = @{ rows = 0; fixes = 0; level = "ok"; texts = @() }
    if ($null -eq $panel) { return $out }
    if ($null -eq $script:PfTip) { $script:PfTip = New-Object System.Windows.Forms.ToolTip }
    $panel.SuspendLayout()
    foreach ($c in @($panel.Controls)) { $panel.Controls.Remove($c); $c.Dispose() }
    $w = [int]$panel.ClientSize.Width
    if ($w -lt 200) { $w = 600 }
    $recheck = { Pf-Check }
    $rows = @()
    if ($Checking) {
        $rows += Pf-Row $panel 0 $w "info" (L "pf_checking") ""
        $out.level = "info"; $out.texts += (L "pf_checking")
    } elseif ($null -eq $result) {
        $rows += Pf-Row $panel 0 $w "warn" (L "pf_unknown") ""
        $out.level = "warn"; $out.texts += (L "pf_unknown")
    } else {
        $iss = @($result.issues | Where-Object { $null -ne $_ })
        if ($iss.Count -eq 0) {
            $rows += Pf-Row $panel 0 $w "ok" ("✓ " + (L "pf_ok")) ""
            $out.texts += (L "pf_ok")
        } else {
            $out.level = [string]$iss[0].level            # 무거운 것부터 온다
            $show = [Math]::Min($iss.Count, $script:PfMaxRows)
            for ($i = 0; $i -lt $show; $i++) {
                $x = $iss[$i]
                $text = "● " + [string]$x.title
                $tip = [string]$x.detail
                if ($i -eq $show - 1 -and $iss.Count -gt $show) {
                    $rest = @($iss[$show..($iss.Count - 1)])
                    $text += "  (" + ((L "pf_more") -f $rest.Count) + ")"
                    $tip += "`n" + (($rest | ForEach-Object { "● " + $_.title }) -join "`n")
                }
                $row = Pf-Row $panel $i $w $x.level $text $tip
                $rows += $row
                $out.texts += $text
                if ($null -ne $x.fix) {
                    $b = Pf-Button $row (L "pf_fix") ($w - 172) 80 $x { param($sender, $e) Pf-Fix $sender.Tag }
                    if ($x.fix.title) { $script:PfTip.SetToolTip($b, [string]$x.fix.title + "`n" + (@($x.fix.argv) -join " ")) }
                    $out.fixes++
                }
            }
        }
    }
    if (-not $Checking) { [void](Pf-Button $rows[0] (L "pf_recheck") ($w - 88) 86 $null $recheck) }
    $out.rows = $rows.Count
    $panel.Height = $rows.Count * $script:PfRowH
    $panel.ResumeLayout()
    return $out
}

function Pf-Check {
    # 점검을 한 번 돌리고 다시 그린다 (WSL 에서 cli preflight — 몇 초)
    if ($null -ne $script:PfPanel) { [void](Pf-Render -Checking) }
    [System.Windows.Forms.Application]::DoEvents()
    $txt = Invoke-Wsl @($PyWsl, "-m", "ds_resources.cli", "preflight", "--json")
    $script:PfResult = From-Json $txt
    [void](Pf-Render)
}

function Pf-Fix($iss) {
    if ($null -eq $iss -or $null -eq $iss.fix) { return }
    $id = [string]$iss.id
    if ($id -notmatch '^[A-Za-z0-9_\-]+$') { return }   # bash 줄에 넣는다 — 이름 꼴만 받는다
    $fx = $iss.fix
    $cmd = (@($fx.argv) -join " ")
    if ($fx.needs_sudo) {
        try { [System.Windows.Forms.Clipboard]::SetText("sudo " + $cmd) } catch { }
        [void][System.Windows.Forms.MessageBox]::Show((L "pf_sudo") + "`n`nsudo " + $cmd, (L "d_title"))
        return
    }
    if ($fx.long) {
        # 오래 걸리는 것(가중치 받기)은 따로 창에서 — 진행이 보이고 런처는 멈추지 않는다
        $inner = "$PyWsl -m ds_resources.cli fix $id; echo; read -r -p 'Enter to close' _"
        Start-Process -FilePath "wsl.exe" -ArgumentList "-d $Distro --cd $RepoDir --exec bash -c `"$inner`""
        [void][System.Windows.Forms.MessageBox]::Show((L "pf_long") + "`n`n" + $cmd, (L "d_title"))
        return
    }
    if (Get-Command Set-Status -ErrorAction SilentlyContinue) { Set-Status (L "pf_fixing") $true }
    $txt = Invoke-Wsl @($PyWsl, "-m", "ds_resources.cli", "fix", $id)
    $r = From-Json $txt
    Pf-Restore-Status
    if ($null -eq $r -or -not $r.ok) {
        $why = if ($null -ne $r -and $r.why) { [string]$r.why } else { [string]$txt }
        [void][System.Windows.Forms.MessageBox]::Show((L "pf_fix_fail") + "`n`n" + $cmd + "`n" + $why, (L "d_title"))
    }
    Pf-Check
}

function Hook-PreflightBanner($parent, $result = $null) {
    $script:PfPanel = $parent
    if ($null -ne $result) { $script:PfResult = $result }
    if ($null -ne $script:PfResult) { [void](Pf-Render); return }
    # 아직 결과가 없다 — 「점검 중」을 그려 두고, 창이 뜬 뒤(메시지 루프가 돌 때) 한 번 점검한다.
    # 그 사이 본문이 PreflightResult 로 결과를 넘겨 주면 그것을 쓴다(두 번 묻지 않는다)
    [void](Pf-Render -Checking)
    $t = New-Object System.Windows.Forms.Timer
    $t.Interval = 400
    $t.Add_Tick({
        param($sender, $e)
        $sender.Stop(); $sender.Dispose()
        if ($null -eq $script:PfResult) { Pf-Check } else { [void](Pf-Render) }
    })
    $t.Start()
}

function Hook-PreflightResult($result) {
    if ($null -eq $result) { return }
    $script:PfResult = $result
    if ($null -ne $script:PfPanel) { [void](Pf-Render) }
}

# 자체 시험 — 가짜 결과로 화면 밖에서 그려 줄 수 · 고치기 단추 수 · 색을 센다 (WSL 을 부르지 않는다)
function PartTest-Preflight {
    $keep = @($script:PfPanel, $script:PfResult)
    $p = New-Object System.Windows.Forms.Panel
    $p.Size = New-Object System.Drawing.Size(600, 10)
    $mk = { param($id, $lv, $fix) [pscustomobject]@{ id = $id; level = $lv; title = "t-$id"; detail = "d-$id"; fix = $fix } }
    $fx = [pscustomobject]@{ id = "k3s_priorityclass"; title = "f"; argv = @("kubectl", "apply", "-f", "orchestrator/k3s/"); needs_sudo = $false; long = $false }
    $three = [pscustomobject]@{ issues = @((& $mk "k3s_priorityclass" "error" $fx), (& $mk "weights" "warn" $fx), (& $mk "hosts" "info" $null)) }
    $r = [ordered]@{}
    $a = Pf-Render $p $three
    $r.three = @{ rows = $a.rows; fixes = $a.fixes; level = $a.level; height = $p.Height;
                  back0 = $p.Controls[0].BackColor.Name; buttons = @($p.Controls | ForEach-Object { $_.Controls } | Where-Object { $_ -is [System.Windows.Forms.Button] } | ForEach-Object { $_.Text }) }
    $six = [pscustomobject]@{ issues = @(1..6 | ForEach-Object { & $mk "i$_" "warn" $null }) }
    $b = Pf-Render $p $six
    $r.six = @{ rows = $b.rows; fixes = $b.fixes; last = $b.texts[-1] }
    $c = Pf-Render $p ([pscustomobject]@{ issues = @() })
    $r.ok = @{ rows = $c.rows; text = $c.texts[0]; back = $p.Controls[0].BackColor.Name; height = $p.Height }
    $d = Pf-Render $p $null
    $r.unknown = @{ rows = $d.rows; level = $d.level }
    # 결과 넘겨받기 — 배너가 없을 때는 들고만 있다
    $script:PfPanel = $null
    Hook-PreflightResult $three
    $r.held = (@($script:PfResult.issues).Count -eq 3)
    $p.Dispose()
    $script:PfPanel = $keep[0]; $script:PfResult = $keep[1]
    return $r
}
