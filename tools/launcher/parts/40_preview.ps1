# 부품: 배치 미리보기를 창 안에 (2026-11-27)
# 「배치 미리보기」의 결과를 대화 상자(긴 글이 잘린다) 대신 배치 단계 안에 그린다 — 맨 위 요약 한 줄,
# 노드마다 「이 작업 → 이 노드」 카드, 그 아래 거절 · 낮춤 · 대기 목록과 까닭. 누를 때마다 다시 그린다.
# 데이터는 `python -m ds_resources.cli plan … --json` (cli.plan_json) 이 준다.
#
# 본문이 부르는 자리
#   Part-Hook "PreviewRun"  @($tPlace, $argv)   — 런처가 만든 plan 명령(argv)에 --json 을 붙여 돌리고 그린다.
#                                                 그렸으면 $true (못 읽으면 원문을 같은 자리에 글로 보인다)
#   Part-Hook "PreviewShow" @($tPlace, $plan)   — 이미 읽은 결과(객체 또는 JSON 글)를 그린다
# 그림 자리: 부모(배치 탭)의 (8,146) 부터 아래 끝까지 — 수동 배치 막대 자리를 덮고, 「닫기」로 막대가 다시 보인다.

$Str["pv_title"]      = @{ ko = "배치 미리보기"; en = "Plan preview" }
$Str["pv_close"]      = @{ ko = "닫기"; en = "Close" }
$Str["pv_mode"]       = @{ ko = "방식 {0} ({1})"; en = "Mode {0} ({1})" }
$Str["pv_live"]       = @{ ko = "지금 잰 자원"; en = "measured now" }
$Str["pv_saved"]      = @{ ko = "기억한 자원"; en = "remembered" }
$Str["pv_counts"]     = @{ ko = "GPU 단위 {0} · CPU 작업 {1} · 거절 {2} · 낮춤 {3} · 대기 {4}";
                           en = "GPU units {0} · CPU tasks {1} · refused {2} · degraded {3} · waiting {4}" }
$Str["pv_gpu_mem"]    = @{ ko = "GPU {0:n1} / {1:n1} GB 사용"; en = "GPU {0:n1} / {1:n1} GB used" }
$Str["pv_notready"]   = @{ ko = "응답 없음"; en = "not responding" }
$Str["pv_gpu_row"]    = @{ ko = "GPU  {0}  ←  {1}"; en = "GPU  {0}  <-  {1}" }
$Str["pv_need"]       = @{ ko = "필요 {0} MB ({1})"; en = "needs {0} MB ({1})" }
$Str["pv_measured"]   = @{ ko = "실측"; en = "measured" }
$Str["pv_estimate"]   = @{ ko = "레지스트리 추정"; en = "registry estimate" }
$Str["pv_room"]       = @{ ko = "지금 여유 {0} MB"; en = "{0} MB free now" }
$Str["pv_admit_ok"]   = @{ ko = "들일 수 있음"; en = "fits" }
$Str["pv_admit_no"]   = @{ ko = "안 됨: {0}"; en = "does not fit: {0}" }
$Str["pv_fps"]        = @{ ko = "fps 상한 {0}"; en = "fps cap {0}" }
$Str["pv_util"]       = @{ ko = "GPU 사용률 {0:n0}%"; en = "GPU load {0:n0}%" }
$Str["pv_waived"]     = @{ ko = "한도 면제"; en = "limits waived" }
$Str["pv_limited"]    = @{ ko = "한도 적용"; en = "within limits" }
$Str["pv_cpu_row"]    = @{ ko = "CPU  {0}"; en = "CPU  {0}" }
$Str["pv_manual"]     = @{ ko = "수동"; en = "manual" }
$Str["pv_auto"]       = @{ ko = "자동"; en = "auto" }
$Str["pv_req"]        = @{ ko = "요청 {0:n1}코어 {1:n0}MB"; en = "requests {0:n1} cores {1:n0} MB" }
$Str["pv_refused"]    = @{ ko = "거절 (못 함)"; en = "Refused" }
$Str["pv_degraded"]   = @{ ko = "낮춤"; en = "Degraded" }
$Str["pv_waiting"]    = @{ ko = "대기"; en = "Waiting" }
$Str["pv_vision_off"] = @{ ko = "비전 없음: {0} — 기기별 모델 매칭에서 비워 둠"; en = "No vision: {0} (left empty in per-device matching)" }
$Str["pv_cpu_block"]  = @{ ko = "파이 CPU 작업: 막힘 — {0}"; en = "Pi CPU tasks blocked: {0}" }
$Str["pv_no_units"]   = @{ ko = "올리는 작업 없음"; en = "Nothing placed" }
$Str["pv_idle"]       = @{ ko = "작업 없는 노드: {0}"; en = "Nodes without work: {0}" }
$Str["pv_raw"]        = @{ ko = "결과를 읽지 못해 원문을 그대로 보입니다"; en = "Could not read the result; showing the raw output" }

# 색 — 배치 막대 · 범례와 같은 뜻으로 쓴다 (파랑 = 올린 작업, 빨강 = 넘침 · 거절, 회색 = 기다림)
$script:PvColor = @{
    gpu = [System.Drawing.Color]::SteelBlue; cpu = [System.Drawing.Color]::LightSteelBlue
    refused = [System.Drawing.Color]::IndianRed; degraded = [System.Drawing.Color]::Goldenrod
    waiting = [System.Drawing.Color]::Silver; okText = [System.Drawing.Color]::DarkGreen
    badText = [System.Drawing.Color]::Firebrick; dim = [System.Drawing.Color]::DimGray
}

function Pv-Text($ctlParent, $text, $x, $y, $w, $color = $null, $bold = $false) {
    # 줄바꿈되는 글 — 높이는 폭에 맞춰 잰다 (AutoSize 는 창이 뜨기 전에 높이를 모른다)
    $l = New-Object System.Windows.Forms.Label
    $l.AutoSize = $false
    $l.Text = [string]$text
    $f = $ctlParent.Font
    if ($bold) { $f = New-Object System.Drawing.Font($ctlParent.Font, [System.Drawing.FontStyle]::Bold); $l.Font = $f }
    $sz = [System.Windows.Forms.TextRenderer]::MeasureText($l.Text, $f, (New-Object System.Drawing.Size($w, 0)),
          ([System.Windows.Forms.TextFormatFlags]::WordBreak -bor [System.Windows.Forms.TextFormatFlags]::NoPadding))
    $l.Location = New-Object System.Drawing.Point($x, $y)
    $l.Size = New-Object System.Drawing.Size($w, ([Math]::Max(16, $sz.Height + 2)))
    if ($null -ne $color) { $l.ForeColor = $color }
    $ctlParent.Controls.Add($l)
    return $l
}
function Pv-Swatch($ctlParent, $color, $x, $y) {
    $p = New-Object System.Windows.Forms.Panel
    $p.Location = New-Object System.Drawing.Point($x, ($y + 4))
    $p.Size = New-Object System.Drawing.Size(10, 10)
    $p.BackColor = $color
    $p.BorderStyle = "FixedSingle"
    $ctlParent.Controls.Add($p)
}
function Pv-List($o) {
    # 늘 배열로 — 하나뿐이면 PowerShell 이 펴 버려 .Count 가 비는 것을 막는다(쉼표)
    if ($null -eq $o) { return ,@() }
    return ,@($o)
}
function Pv-Fps($caps) {
    if ($null -eq $caps) { return "" }
    return ((@($caps.PSObject.Properties | Sort-Object Name | ForEach-Object {
        "{0} {1:n1}" -f $_.Name, [double]$_.Value })) -join ", ")
}
function Pv-Box($parent) {
    # 배치 탭 안의 그림 자리 — 있으면 다시 쓰고 없으면 만든다
    foreach ($c in $parent.Controls) { if ($c.Name -eq "pvBox") { return $c } }
    $box = New-Object System.Windows.Forms.Panel
    $box.Name = "pvBox"
    $box.BorderStyle = "FixedSingle"
    $w = [Math]::Max(300, $parent.ClientSize.Width - 16)
    $h = [Math]::Max(200, $parent.ClientSize.Height - 154)
    $box.Location = New-Object System.Drawing.Point(8, 146)
    $box.Size = New-Object System.Drawing.Size($w, $h)
    $head = New-Object System.Windows.Forms.Label
    $head.Name = "pvHead"
    $head.AutoSize = $false
    $head.Location = New-Object System.Drawing.Point(6, 4)
    $head.Size = New-Object System.Drawing.Size(($w - 90), 36)
    $head.Font = New-Object System.Drawing.Font($parent.Font, [System.Drawing.FontStyle]::Bold)
    $box.Controls.Add($head)
    $close = New-Object System.Windows.Forms.Button
    $close.Name = "pvClose"
    $close.Location = New-Object System.Drawing.Point(($w - 80), 4)
    $close.Width = 70
    $close.Add_Click({ $this.Parent.Visible = $false })
    $box.Controls.Add($close)
    $body = New-Object System.Windows.Forms.Panel
    $body.Name = "pvBody"
    $body.AutoScroll = $true
    $body.BackColor = [System.Drawing.Color]::White
    $body.Location = New-Object System.Drawing.Point(0, 42)
    $body.Size = New-Object System.Drawing.Size(($w - 2), ($h - 44))
    $box.Controls.Add($body)
    $parent.Controls.Add($box)
    return $box
}
function Pv-Node-Head($n) {
    if ($null -eq $n) { return "" }
    $bits = @()
    if ($n.source) { $bits += [string]$n.source }
    if ($n.role) { $bits += [string]$n.role }
    $t = [string]$n.name
    if ($bits.Count) { $t += "  (" + ($bits -join ", ") + ")" }
    if ($null -ne $n.gpu) { $t += "  —  " + ((L "pv_gpu_mem") -f ([double]$n.gpu.used_mb / 1024), ([double]$n.gpu.total_mb / 1024)) }
    if (-not $n.ready -or $n.stale) { $t += "  —  " + (L "pv_notready") }
    return $t
}
function Hook-PreviewShow($parent, $planJson) {
    # 결과를 그린다. $planJson 은 cli plan --json 을 읽은 객체거나 그 글이다. 못 읽으면 원문을 그대로 보인다
    $p = $planJson
    $raw = $null
    if ($p -is [string] -or $p -is [array]) { $raw = (@($p) -join "`n"); $p = From-Json $raw }
    $box = Pv-Box $parent
    $head = $box.Controls["pvHead"]
    $body = $box.Controls["pvBody"]
    $box.Controls["pvClose"].Text = (L "pv_close")
    $body.SuspendLayout()
    $body.AutoScrollPosition = New-Object System.Drawing.Point(0, 0)
    foreach ($c in @($body.Controls)) { $c.Dispose() }
    $body.Controls.Clear()
    $w = $body.ClientSize.Width - 30                     # 세로 막대 자리 (가로 막대가 생기지 않게)
    $y = 6
    $count = @{ cards = 0; gpu = 0; cpu = 0; refused = 0; degraded = 0; waiting = 0 }
    if ($null -eq $p -or $null -eq $p.gpu_units) {
        $head.Text = (L "pv_title")
        [void](Pv-Text $body (L "pv_raw") 8 $y $w $script:PvColor.badText $true)
        $y += 24
        $tb = New-Object System.Windows.Forms.TextBox
        $tb.Multiline = $true; $tb.ReadOnly = $true; $tb.ScrollBars = "Both"; $tb.WordWrap = $false
        $tb.Location = New-Object System.Drawing.Point(8, $y)
        $tb.Size = New-Object System.Drawing.Size($w, [Math]::Max(120, $body.ClientSize.Height - $y - 8))
        $tb.Text = $(if ($raw) { $raw -replace "`r?`n", "`r`n" } else { L "m_no_plan" })
        $body.Controls.Add($tb)
        $body.ResumeLayout()
        $box.Visible = $true; $box.BringToFront()
        return $count
    }
    $gpuU = Pv-List $p.gpu_units; $cpuU = Pv-List $p.cpu_units
    $refused = Pv-List $p.refused; $degraded = Pv-List $p.degraded; $waiting = Pv-List $p.waiting
    $basis = if ($p.basis -eq "live") { L "pv_live" } else { L "pv_saved" }
    $head.Text = ((L "pv_mode") -f $p.mode, $basis) + $(if ($p.why) { " — " + $p.why } else { "" }) + "`n" +
                 ((L "pv_counts") -f $gpuU.Count, $cpuU.Count, $refused.Count, $degraded.Count, $waiting.Count)
    $head.ForeColor = $(if ($refused.Count) { $script:PvColor.badText } else { [System.Drawing.SystemColors]::ControlText })
    if ($p.caps -and $p.caps.cpu_tasks -eq $false) {
        $l = Pv-Text $body ((L "pv_cpu_block") -f $p.caps.why) 8 $y $w $script:PvColor.badText
        $y += $l.Height + 4
    }
    foreach ($s in (Pv-List $p.vision_off)) {
        $l = Pv-Text $body ((L "pv_vision_off") -f $s) 8 $y $w $script:PvColor.dim
        $y += $l.Height + 2
    }
    # 노드 순서: 스냅숏 순서(서버 먼저) → 거기에 없는 노드
    $nodeInfo = [ordered]@{}
    foreach ($n in (Pv-List $p.nodes)) { $nodeInfo[[string]$n.name] = $n }
    $order = @($nodeInfo.Keys)
    foreach ($u in ($gpuU + $cpuU)) { if ($order -notcontains [string]$u.node) { $order += [string]$u.node } }
    $idle = @()
    foreach ($nn in $order) {
        $mineG = @($gpuU | Where-Object { [string]$_.node -eq $nn })
        $mineC = @($cpuU | Where-Object { [string]$_.node -eq $nn })
        if ($mineG.Count -eq 0 -and $mineC.Count -eq 0) { $idle += $nn; continue }
        $card = New-Object System.Windows.Forms.Panel
        $card.BorderStyle = "FixedSingle"
        $card.BackColor = [System.Drawing.Color]::FromArgb(246, 248, 251)
        $card.Location = New-Object System.Drawing.Point(6, $y)
        $card.Width = $w
        $body.Controls.Add($card)
        $count.cards++
        $iw = $w - 34
        $nh = Pv-Node-Head $nodeInfo[$nn]
        if (-not $nh) { $nh = $nn }
        $cy = 4
        $l = Pv-Text $card $nh 6 $cy ($w - 14) $null $true
        $cy += $l.Height + 2
        foreach ($u in $mineG) {
            $count.gpu++
            Pv-Swatch $card $script:PvColor.gpu 8 $cy
            $l = Pv-Text $card ((L "pv_gpu_row") -f ((Pv-List $u.models) -join "+"), ((Pv-List $u.sources) -join ", ")) 24 $cy $iw
            $cy += $l.Height
            $bits = @((L "pv_need") -f $u.need_mb, $(if ($u.need_basis -eq "measured") { L "pv_measured" } else { L "pv_estimate" }))
            $col = $script:PvColor.dim
            if ($null -ne $u.admit) {
                $bits += ((L "pv_room") -f $(if ($null -ne $u.admit.room_mb) { $u.admit.room_mb } else { "?" }))
                if ($u.admit.ok) { $bits += (L "pv_admit_ok"); $col = $script:PvColor.okText }
                else { $bits += ((L "pv_admit_no") -f $u.admit.why); $col = $script:PvColor.badText }
            }
            $l = Pv-Text $card ($bits -join " · ") 24 $cy $iw $col
            $cy += $l.Height
            $bits = @((L "pv_fps") -f (Pv-Fps $u.fps_cap))
            if ($u.limit_mode -eq "none") { $bits += (L "pv_waived") }
            elseif ($null -ne $u.gpu_util) { $bits += ((L "pv_util") -f (100 * [double]$u.gpu_util)) }
            $l = Pv-Text $card ($bits -join " · ") 24 $cy $iw $script:PvColor.dim
            $cy += $l.Height + 4
        }
        foreach ($u in $mineC) {
            $count.cpu++
            Pv-Swatch $card $script:PvColor.cpu 8 $cy
            $l = Pv-Text $card ((L "pv_cpu_row") -f $u.task) 24 $cy $iw
            $cy += $l.Height
            $bits = @($(if ($u.placement_by -eq "manual") { L "pv_manual" } else { L "pv_auto" }))
            if ($u.limit_mode -eq "none") {
                $wv = (Pv-List $u.waived) -join ", "
                $bits += ((L "pv_waived") + $(if ($wv) { " ($wv)" } else { "" }))
            } else { $bits += (L "pv_limited") }
            if ($null -ne $u.requests) {
                $bits += ((L "pv_req") -f ([double]$u.requests.cpu_m / 1000), [double]$u.requests.mem_mb)
            }
            if ($u.reason) { $bits += [string]$u.reason }
            $l = Pv-Text $card ($bits -join " · ") 24 $cy $iw $script:PvColor.dim
            $cy += $l.Height + 4
        }
        $card.Height = $cy + 2
        $y += $card.Height + 6
    }
    if ($gpuU.Count -eq 0 -and $cpuU.Count -eq 0) {
        $l = Pv-Text $body (L "pv_no_units") 8 $y $w $script:PvColor.dim
        $y += $l.Height + 4
    }
    if ($idle.Count) {
        $l = Pv-Text $body ((L "pv_idle") -f ($idle -join ", ")) 8 $y $w $script:PvColor.dim
        $y += $l.Height + 6
    }
    foreach ($sec in @(@("refused", "pv_refused"), @("degraded", "pv_degraded"), @("waiting", "pv_waiting"))) {
        $items = Pv-List $p.($sec[0])
        if ($items.Count -eq 0) { continue }
        $l = Pv-Text $body ("{0} ({1})" -f (L $sec[1]), $items.Count) 8 $y $w $null $true
        $y += $l.Height + 2
        foreach ($it in $items) {
            $count[$sec[0]]++
            Pv-Swatch $body $script:PvColor[$sec[0]] 12 $y
            $l = Pv-Text $body ("{0} — {1}" -f $it.item, $it.reason) 28 $y ($w - 24)
            $y += $l.Height + 2
        }
        $y += 6
    }
    [void](Part-Hook "PreviewCards" @($body, $p))      # 카드마다 공유 작업 줄 (부품 50_workgraph)
    $body.ResumeLayout()
    $box.Visible = $true
    $box.BringToFront()
    return $count
}
function Hook-PreviewRun($parent, [string[]]$argv) {
    # 런처가 만든 plan 명령(argv) 그대로에 --json 을 붙여 돌리고 그린다 — 본문은 대화 상자 대신 이것만 부른다
    $a = @($argv)
    if ($a -notcontains "--json") { $a += "--json" }
    $txt = Invoke-Wsl $a
    $j = From-Json $txt
    if ($null -ne $j) { [void](Hook-PreviewShow $parent $j) } else { [void](Hook-PreviewShow $parent ([string]$txt)) }
    return $true
}

$script:PvFake = @'
{"gpu_units":[{"unit":"gpu:yoloe","node":"book-4vg1edpvi5","models":["yoloe"],"sources":["pi13","robot1"],
  "fps_cap":{"pi13":10.0,"robot1":6.29},"need_mb":1560,"need_basis":"estimate","gpu_util":2.45,"limit_mode":"none",
  "admit":{"ok":true,"why":"","free_mb":6141,"room_mb":5841,"need_mb":1560}},
 {"unit":"gpu:river","node":"book-4vg1edpvi5","models":["river"],"sources":["a2"],"fps_cap":{"a2":5.0},
  "need_mb":4200,"need_basis":"measured","gpu_util":0.6,"admit":{"ok":false,"why":"GPU 메모리 부족","room_mb":4281}}],
 "cpu_units":[{"task":"object_detect.server","node":"book-4vg1edpvi5","limit_mode":"none","placement_by":"auto",
  "requests":{"cpu_m":500,"mem_mb":2048}},
  {"task":"object_detect.device@pi13","node":"pi13","placement_by":"manual","limit_mode":"none","waived":["cpu"],
   "requests":{"cpu_m":300,"mem_mb":256}},
  {"task":"cam360_slice","node":"pi6","placement_by":"auto","reason":"비교 측정 전"}],
 "refused":[{"item":"cpu:object_detect.device@robot1","code":"no_sensor","reason":"그 작업이 필요한 센서를 가진 노드가 없음 — book-4vg1edpvi5 센서 camera 가 노드에 없음"}],
 "degraded":[{"item":"fn:object_detect","reason":"device 단계가 없음 — optional 이라 품질만 낮춤 (cascade)"}],
 "waiting":[{"item":"input:pi13","reason":"입력 대기 — 영상 없음. 들어오면 바로 처리"},
            {"item":"input:robot1","reason":"입력 대기 — 영상 없음. 들어오면 바로 처리"}],
 "functions":{},"mode":"k3s","basis":"live","why":"","caps":{"cpu_tasks":true},"ok":true,"vision_off":["drone2"],
 "nodes":[{"name":"book-4vg1edpvi5","role":"gpu","ready":true,"stale":false,"source":null,
           "gpu":{"total_mb":6141,"used_mb":300,"committed":0.0}},
          {"name":"pi13","role":"small","ready":true,"stale":false,"source":"pi13"},
          {"name":"pi6","role":"sensor","ready":true,"stale":false,"source":"realsense"},
          {"name":"pi1","role":"robot","ready":false,"stale":true,"source":"ep-001"}]}
'@
function PartTest-Preview {
    # 가짜 배치 결과를 창에 띄우지 않고 그려 컨트롤 수를 센다. 두 번 그려도 쌓이지 않는지(다시 그리기)도 본다.
    # DS_LAUNCHER_TESTSHOT 이 있으면 그림 preview_inline.png 를 남긴다
    $host2 = New-Object System.Windows.Forms.Panel
    $host2.Font = New-Object System.Drawing.Font("Malgun Gothic", 10)
    $host2.Size = New-Object System.Drawing.Size(616, 544)
    $plan = $script:PvFake | ConvertFrom-Json
    $first = Hook-PreviewShow $host2 $plan
    $c1 = $host2.Controls["pvBox"].Controls["pvBody"].Controls.Count
    $cnt = Hook-PreviewShow $host2 $script:PvFake          # 글로 줘도 읽는다 · 다시 그려도 그대로
    $box = $host2.Controls["pvBox"]
    $body = $box.Controls["pvBody"]
    $r = [ordered]@{
        boxes = @($host2.Controls | Where-Object { $_.Name -eq "pvBox" }).Count
        body_controls = $body.Controls.Count
        same_on_rerender = ($c1 -eq $body.Controls.Count)
        cards = $cnt.cards; gpu = $cnt.gpu; cpu = $cnt.cpu
        refused = $cnt.refused; degraded = $cnt.degraded; waiting = $cnt.waiting
        head = $box.Controls["pvHead"].Text
        waived_shown = [bool](@($body.Controls | ForEach-Object { $_.Controls } | Where-Object { $_.Text -like ("*" + (L "pv_waived") + "*") }).Count)
        content_height = ($body.Controls | ForEach-Object { $_.Bottom } | Measure-Object -Maximum).Maximum
    }
    if ($env:DS_LAUNCHER_TESTSHOT) {
        $bmp = New-Object System.Drawing.Bitmap $host2.Width, $host2.Height
        $host2.DrawToBitmap($bmp, (New-Object System.Drawing.Rectangle 0, 0, $host2.Width, $host2.Height))
        $f = Join-Path $env:DS_LAUNCHER_TESTSHOT "preview_inline.png"
        $bmp.Save($f, [System.Drawing.Imaging.ImageFormat]::Png)
        $bmp.Dispose()
        $r.shot = $f
    }
    $bad = Hook-PreviewShow $host2 "Traceback: 무엇이 깨짐"   # 못 읽는 글 → 원문 상자
    $r.raw_fallback = (@($host2.Controls["pvBox"].Controls["pvBody"].Controls | Where-Object { $_ -is [System.Windows.Forms.TextBox] }).Count -eq 1)
    $host2.Dispose()
    return $r
}
