# 부품 50_workgraph — 함께 쓰는 작업 · 중복 알림 · 추적 입력 · 재식별 범위 (2026-11-27)
# 기능들이 같은 소스에서 같은 산출물(거리 · 탐지 …)을 쓰면 백엔드의 작업 그래프가 한 번만 돌린다. 그 결과를 짧게 보인다.
#   기능 단계 오른쪽 상세 — 재식별: 범위(소스 안 / 소스 묶음) · 묶음 소스 / 물체 추적: 입력 기능 / 모든 기능: 함께 쓰는 작업
#   기능 단계 왼쪽 아래  — 중복 알림 한 줄씩 「pi12 거리 2중 · 2.0 GB 절약 가능 [맞추기]」
#   배치 미리보기 카드  — 노드마다 「공유 2 · 중복 1」 (풍선말에 작업 목록)
#
# 데이터 (백엔드 계약 — 없으면 그 칸을 그리지 않는다)
#   cli places --json : function_params {reid: {scope: {choices, default, labels}, group_sources: {min}},
#                                        object_track: {input: {choices, default}}} · artifacts {기능: {produces, consumes}}
#   cli plan --json (또는 places 의 자동 배치) : work [{source, node, artifact, model, stage, fps, consumers, shared, gpu_mb, unit}]
#                                               work_warnings [{source, artifact, models, functions, save_mb, code, reason}]
# WSL 을 더 부르지 않는다 — 시작 때 읽은 places · 「배치 미리보기」 결과 · Load-Places 결과를 다시 쓰고, 지금 고른 기능 · 소스로 거른다.
#
# 본문이 부르는 자리
#   Part-Hook "SettingsLoad"  @($saved)          — settings.json 의 functionParams 를 읽는다
#   Part-Hook "PlacesLoaded"  @($r)              — 시작 때 읽은 cli places (function_params · artifacts)
#   Part-Hook "MatchFoot"     @($gMatch)         — 왼쪽 아래 중복 알림 (Update-GpuTotal 끝)
#   Part-Hook "DetailExtra"   @($fn, $detailBody) — 상세 아래 칸 (Build-Detail 끝)
#   Part-Hook "ProfileExtra"  @($prof)           — 프로필에 function_params
#   Part-Hook "OrchArgs"      @($profWsl)        — 기본값이 아닐 때만 --function-params '<json>'
#   Part-Hook "PreviewCards"  @($body, $plan)    — 미리보기 카드에 공유 줄 (40_preview 의 Hook-PreviewShow 끝)

$Str["wg_reid_head"]   = @{ ko = "재식별 범위"; en = "Re-id scope" }
$Str["wg_scope"]       = @{ ko = "범위"; en = "Scope" }
$Str["wg_scope_source"] = @{ ko = "소스 안"; en = "Per source" }
$Str["wg_scope_group"] = @{ ko = "소스 묶음"; en = "Source group" }
$Str["wg_group_srcs"]  = @{ ko = "묶을 소스"; en = "Grouped" }
$Str["wg_need_min"]    = @{ ko = "{0}곳 이상"; en = "{0}+ needed" }
$Str["wg_scope_tip"]   = @{ ko = "소스 안: 같은 소스 안에서만 같은 물체를 잇는다`n소스 묶음: 고른 소스끼리 같은 물체를 잇는다";
                            en = "Per source: match within one source`nSource group: match across the chosen sources" }
$Str["wg_input"]       = @{ ko = "입력"; en = "Input" }
$Str["wg_input_tip"]   = @{ ko = "추적이 받을 탐지 — 켠 탐지 기능 중에서"; en = "Detections to track — from the detection capabilities that are on" }
$Str["wg_input_off"]   = @{ ko = "꺼짐"; en = "off" }
$Str["wg_work_head"]   = @{ ko = "함께 쓰는 작업"; en = "Shared work" }
$Str["wg_shared"]      = @{ ko = "공유"; en = "shared" }
$Str["wg_warn"]        = @{ ko = "{0} {1} {2}중 · {3:0.0} GB 절약 가능"; en = "{0} {1} x{2} · {3:0.0} GB to save" }
$Str["wg_align"]       = @{ ko = "맞추기"; en = "Align" }
$Str["wg_align_tip"]   = @{ ko = "{0} 의 모델을 {1} 로"; en = "Set {0} to {1}" }
$Str["wg_align_none"]  = @{ ko = "같은 모델을 고를 수 없음"; en = "No common model to pick" }
$Str["wg_card"]        = @{ ko = "공유 {0} · 중복 {1}"; en = "shared {0} · dup {1}" }
$Str["wg_card0"]       = @{ ko = "공유 {0}"; en = "shared {0}" }
$Str["wg_tip_item"]    = @{ ko = "{0}  노드 {1} · {2} fps · {3} MB"; en = "{0}  node {1} · {2} fps · {3} MB" }
# 산출물 이름 — 계약 키 → 짧은 이름 (모르면 키 그대로)
$Str["wg_a_depth"]      = @{ ko = "거리"; en = "depth" }
$Str["wg_a_detections"] = @{ ko = "탐지"; en = "detections" }
$Str["wg_a_tracks"]     = @{ ko = "궤적"; en = "tracks" }
$Str["wg_a_embeddings"] = @{ ko = "임베딩"; en = "embeddings" }
$Str["wg_a_masks"]      = @{ ko = "분할"; en = "masks" }
$Str["wg_a_frames"]     = @{ ko = "영상"; en = "frames" }

$script:FnParams   = @{}       # settings.json functionParams — @{ reid = @{scope; group_sources}; object_track = @{input} }
$script:WgPlaces   = $null     # 시작 때 읽은 cli places (function_params · artifacts)
$script:WgPlan     = $null     # work · work_warnings 가 있는 마지막 결과 (plan 미리보기 또는 places 자동 배치)
$script:WgSeen     = $null     # 마지막으로 살펴본 $script:Places (바뀌었으면 그 안의 work 를 다시 쓴다)
$script:WgAligned  = @{}       # 「맞추기」로 푼 알림 — "소스|산출물" (새 결과가 오면 비운다)
$script:WgNoReplan = $false    # 자체 시험 — 맞추기 뒤 다시 배치하지 않는다 (WSL 을 부르지 않게)
$script:WgIcon     = $null
$WgAmber = [System.Drawing.Color]::FromArgb(176, 104, 0)
$WgRed   = [System.Drawing.Color]::Firebrick

# ---------------------------------------------------------------- 데이터
function WG-Hash($o) {
    # PSCustomObject(설정 파일) → 해시 (배열은 그대로)
    if ($null -eq $o) { return @{} }
    if ($o -is [hashtable]) { return $o }
    $h = @{}
    foreach ($p in $o.PSObject.Properties) {
        $v = $p.Value
        if ($v -is [System.Management.Automation.PSCustomObject]) { $v = WG-Hash $v }
        $h[$p.Name] = $v
    }
    return $h
}
function WG-Has($o, $name) { return ($null -ne $o -and @($o.PSObject.Properties.Name) -contains $name) }
function WG-Spec {
    # function_params · artifacts 를 주는 places — 시작 때 것, 없으면 배치 단계가 읽은 것
    foreach ($o in @($script:WgPlaces, $script:Places)) { if (WG-Has $o "function_params") { return $o } }
    return $null
}
function WG-Param($grp) { $sp = WG-Spec; if ($null -eq $sp) { return $null }; return $sp.function_params.$grp }
function WG-Artifacts {
    foreach ($o in @($script:WgPlaces, $script:Places)) { if (WG-Has $o "artifacts") { return $o.artifacts } }
    return $null
}
function WG-ReidFn {
    if ($null -eq (WG-Param "reid")) { return $null }
    if (@($script:FnList) -contains "reid") { return "reid" }
    return $null
}
function WG-TrackFn {
    # 물체 추적 기능 키 — object_track 이 있으면 그것, 없으면 궤적(tracks)을 내는 한 카메라 추적
    if ($null -eq (WG-Param "object_track")) { return $null }
    $fl = @($script:FnList)
    if ($fl -contains "object_track") { return "object_track" }
    $art = WG-Artifacts
    if ($null -ne $art) {
        $prod = @($art.PSObject.Properties | Where-Object { @($_.Value.produces) -contains "tracks" -and $fl -contains $_.Name } |
                  ForEach-Object { $_.Name } | Sort-Object @{ Expression = { $_ -ne "single_cam_track" } }, @{ Expression = { $_ } })
        if ($prod.Count) { return $prod[0] }
    }
    if ($fl -contains "single_cam_track") { return "single_cam_track" }
    return $null
}
function WG-Fp($grp) {
    if ($null -eq $script:FnParams) { $script:FnParams = @{} }
    if ($null -eq $script:FnParams[$grp]) { $script:FnParams[$grp] = @{} }
    return $script:FnParams[$grp]
}
function WG-ReidScope {
    $spec = WG-Param "reid"
    $ch = @($spec.scope.choices)
    $v = [string](WG-Fp "reid")["scope"]
    if ($v -and ($ch.Count -eq 0 -or $ch -contains $v)) { return $v }
    if ($spec.scope.default) { return [string]$spec.scope.default }
    return "source"
}
function WG-ReidGroup {
    # 묶을 소스 — 재식별을 켠 소스 중 고른 것. 고른 적이 없으면 전부
    $rf = WG-ReidFn
    $all = @(if ($rf) { Fn-Sources $rf })
    $fp = WG-Fp "reid"
    if (-not $fp.ContainsKey("group_sources") -or $null -eq $fp["group_sources"]) { return $all }
    $want = @($fp["group_sources"])
    return @($all | Where-Object { $want -contains $_ })
}
function WG-GroupMin { $m = (WG-Param "reid").group_sources.min; if ($m) { return [int]$m }; return 2 }
function WG-TrackChoices {
    # 입력으로 고를 탐지 기능 — 켠 것만. 켠 것이 없으면 전부(흐리게 보인다)
    $spec = WG-Param "object_track"
    $ch = @($spec.input.choices | Where-Object { $_ })
    $on = @($ch | Where-Object { Fn-On $_ })
    if ($on.Count) { return $on }
    return $ch
}
function WG-TrackInput {
    $spec = WG-Param "object_track"
    $ch = @(WG-TrackChoices)
    $v = [string](WG-Fp "object_track")["input"]
    if ($v -and $ch -contains $v) { return $v }
    $d = [string]$spec.input.default
    if ($d -and $ch -contains $d) { return $d }
    if ($ch.Count) { return [string]$ch[0] }
    return $d
}
function WG-Params($onlyChanged = $false) {
    # 프로필 · 명령줄의 function_params. $onlyChanged 면 기본값과 다른 것만
    $out = [ordered]@{}
    $rf = WG-ReidFn
    if ($rf -and (Fn-On $rf)) {
        $sc = WG-ReidScope
        $def = [string](WG-Param "reid").scope.default
        if (-not $def) { $def = "source" }
        if (-not $onlyChanged -or $sc -ne $def) {
            $r = [ordered]@{ scope = $sc }
            if ($sc -eq "group") { $r.group_sources = @(WG-ReidGroup) }
            $out.reid = $r
        }
    }
    $tf = WG-TrackFn
    if ($tf -and (Fn-On $tf)) {
        $in = WG-TrackInput
        $o = [ordered]@{}
        if ($in -and (-not $onlyChanged -or $in -ne [string](WG-Param "object_track").input.default)) { $o.input = $in }
        $m = Fn-Model $tf
        if ($m -and -not $onlyChanged) { $o.tracker = $m }
        if ($o.Count) { $out.object_track = $o }
    }
    return $out
}
function WG-Plan {
    # 쓸 결과 — 배치 단계가 places 를 새로 읽었고 그 안에 work 가 있으면 그것이 가장 새것이다
    if ($null -ne $script:Places -and -not [object]::ReferenceEquals($script:Places, $script:WgSeen)) {
        $script:WgSeen = $script:Places
        $pl = WG-PlanOf $script:Places
        if ($null -ne $pl) { $script:WgPlan = $pl; $script:WgAligned = @{} }
    }
    return $script:WgPlan
}
function WG-PlanOf($o) {
    foreach ($c in @($o, $o.plan, $o.auto_plan)) { if (WG-Has $c "work") { return $c } }
    return $null
}
function WG-Art($a) { $k = "wg_a_" + $a; if ($null -ne $Str[$k]) { return (L $k) }; return [string]$a }
function WG-Names($fns) { return ((@($fns) | ForEach-Object { Fn-Text2 $_ }) -join ", ") }
function WG-Work {
    # 지금 고른 기능 · 소스로 거른 작업 — 소비 기능 중 그 소스에 지금 켠 것만 남긴다
    $p = WG-Plan
    if ($null -eq $p) { return @() }
    $out = @()
    foreach ($w in @($p.work)) {
        if ($null -eq $w) { continue }
        $src = [string]$w.source
        $cons = @(@($w.consumers) | Where-Object { $_ -and (@(Fn-Sources $_) -contains $src) })
        if ($cons.Count -eq 0) { continue }
        $out += [pscustomobject]@{ source = $src; node = [string]$w.node; artifact = [string]$w.artifact; model = [string]$w.model
                                   fps = $w.fps; gpu_mb = $w.gpu_mb; unit = [string]$w.unit; consumers = $cons
                                   shared = ([bool]$w.shared -and $cons.Count -ge 2) }
    }
    return $out
}
function WG-Warnings {
    # 지금도 유효한 중복 알림 — 기능이 모두 그 소스에 켜져 있고 「맞추기」로 풀지 않은 것
    $p = WG-Plan
    if ($null -eq $p) { return @() }
    $out = @()
    foreach ($w in @($p.work_warnings)) {
        if ($null -eq $w) { continue }
        $src = [string]$w.source
        $fns = @($w.functions)
        if ($fns.Count -lt 2) { continue }
        if (@($fns | Where-Object { @(Fn-Sources $_) -notcontains $src }).Count) { continue }
        if ($script:WgAligned.ContainsKey($src + "|" + $w.artifact)) { continue }
        $out += $w
    }
    return $out
}
function WG-WorkText($w) {
    return ("{0} · {1}({2}) ← {3}" -f $w.source, (WG-Art $w.artifact), $w.model, (WG-Names $w.consumers))
}
function WG-WarnText($w) {
    $n = [Math]::Max(@($w.functions).Count, @($w.models).Count)
    return ((L "wg_warn") -f $w.source, (WG-Art $w.artifact), $n, ([double]$w.save_mb / 1024.0))
}
function WG-AlignPlan($w) {
    # 맞출 자리 — 두 번째부터의 기능에서 첫 기능의 모델을 고를 수 있는 슬롯. [{fn, slot}] (없으면 빈 목록)
    $m0 = [string]@($w.models)[0]
    $out = @()
    $fns = @($w.functions)
    for ($i = 1; $i -lt $fns.Count; $i++) {
        foreach ($sl in @(Fn-Slots $fns[$i])) {
            if (@(@($sl.choices) | ForEach-Object { [string]$_.v }) -contains $m0) { $out += @{ fn = $fns[$i]; slot = $sl; v = $m0 }; break }
        }
    }
    return $out
}
function WG-Align($w) {
    # 「맞추기」 — 두 번째 기능의 그 슬롯을 첫 기능의 모델로 (On-SlotPick 과 같은 저장 자리), 다시 그리고 다시 배치한다
    $todo = @(WG-AlignPlan $w)
    if ($todo.Count -eq 0) { return $false }
    foreach ($t in $todo) {
        if ($t.slot.store -eq "fnModel") { $script:FnModel[$t.slot.key] = $t.v } else { $script:CuePick[$t.slot.key] = $t.v }
    }
    $script:WgAligned[[string]$w.source + "|" + $w.artifact] = $true
    Build-Match
    Build-CueTable
    Build-CueBoxes
    Build-Overlay
    Changed
    if (-not $script:WgNoReplan -and (Get-Command Show-Plan -ErrorAction SilentlyContinue) -and
        (Chosen-Models).Count -and (On-All).Count) { Show-Plan }
    return $true
}

# ---------------------------------------------------------------- 그림 조각
function WG-LinkIcon {
    # 작은 고리 둘 — 「함께 쓴다」 표시
    if ($null -ne $script:WgIcon) { return $script:WgIcon }
    $b = New-Object System.Drawing.Bitmap 16, 16
    $g = [System.Drawing.Graphics]::FromImage($b)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $pen = New-Object System.Drawing.Pen([System.Drawing.Color]::SteelBlue, 1.8)
    $g.DrawEllipse($pen, 1, 5, 8, 6)
    $g.DrawEllipse($pen, 7, 5, 8, 6)
    $pen.Dispose(); $g.Dispose()
    $script:WgIcon = $b
    return $b
}
function WG-Icon($parent, $x, $y, $tip) {
    $pb = New-Object System.Windows.Forms.PictureBox
    $pb.Image = (WG-LinkIcon)
    $pb.Location = New-Object System.Drawing.Point($x, $y)
    $pb.Size = New-Object System.Drawing.Size(16, 16)
    $pb.Tag = @{ wgIcon = $true }
    $parent.Controls.Add($pb)
    if ($tip) { $modelTip.SetToolTip($pb, $tip) }
    return $pb
}
function WG-Lbl($parent, $text, $x, $y, $w, $color = $null) {
    $l = New-Object System.Windows.Forms.Label
    $l.AutoSize = $false
    $l.AutoEllipsis = $true
    $l.Text = [string]$text
    $l.Location = New-Object System.Drawing.Point($x, $y)
    $l.Size = New-Object System.Drawing.Size($w, 20)
    if ($null -ne $color) { $l.ForeColor = $color }
    $parent.Controls.Add($l)
    return $l
}
function WG-Head($parent, $text, $y) {
    $l = New-Object System.Windows.Forms.Label
    $l.AutoSize = $true
    $l.Text = $text
    $l.Font = $fontB
    $l.Location = New-Object System.Drawing.Point(4, $y)
    $parent.Controls.Add($l)
    return ($y + 24)
}

# ---------------------------------------------------------------- 상세 칸 (기능 단계 오른쪽)
function WG-DetailPanel($body) {
    foreach ($c in $body.Controls) { if ($c.Name -eq "wgDetail") { return $c } }
    $p = New-Object System.Windows.Forms.Panel
    $p.Name = "wgDetail"
    $p.Size = New-Object System.Drawing.Size(570, 10)
    $p.Anchor = "Top, Left, Right"
    $body.Controls.Add($p)
    # 위 칸의 높이가 바뀌면(대상 표 펼침 · 근거 표) 따라 내려간다
    $tgtSection.Add_SizeChanged({ WG-PlaceDetail })
    $infoPanel.Add_SizeChanged({ WG-PlaceDetail })
    return $p
}
function WG-PlaceDetail {
    $p = $null
    foreach ($c in $detailBody.Controls) { if ($c.Name -eq "wgDetail") { $p = $c } }
    if ($null -eq $p) { return }
    $ref = $(if ($script:WgDetailKind -eq "targeted") { $tgtSection } else { $infoPanel })
    $p.Location = New-Object System.Drawing.Point(0, $ref.Bottom)
    $p.Width = $ref.Width
}
function Hook-DetailExtra($fn, $body) {
    if ($null -eq $body) { return }
    $p = WG-DetailPanel $body
    $wasB = $script:Building
    $script:Building = $true
    $p.SuspendLayout()
    $p.Controls.Clear()
    $script:WgDetailKind = $(if ($fn) { Fn-Kind $fn } else { "none" })
    $y = 0
    if ($fn) {
        if ($fn -eq (WG-ReidFn)) { $y = WG-DrawReid $p $y }
        if ($fn -eq (WG-TrackFn)) { $y = WG-DrawTrack $p $y }
        $y = WG-DrawWork $p $fn $y
    }
    $p.Height = [Math]::Max(1, $y)
    $p.ResumeLayout()
    $script:Building = $wasB
    WG-PlaceDetail
    return $y
}
function WG-DrawReid($p, $y) {
    # 재식별 범위 — 「소스 안 / 소스 묶음」, 묶음이면 묶을 소스(재식별을 켠 소스만)
    $spec = WG-Param "reid"
    $y = WG-Head $p (L "wg_reid_head") $y
    $k = WG-Lbl $p (L "wg_scope") 4 ($y + 4) 142 ([System.Drawing.SystemColors]::GrayText)
    $modelTip.SetToolTip($k, (L "wg_scope_tip"))
    $rp = New-Object System.Windows.Forms.Panel
    $rp.Location = New-Object System.Drawing.Point(150, $y)
    $rp.Size = New-Object System.Drawing.Size(320, 26)
    $rp.Name = "wgScope"
    $p.Controls.Add($rp)
    $cur = WG-ReidScope
    $x = 0
    $choices = @($spec.scope.choices)
    if ($choices.Count -eq 0) { $choices = @("source", "group") }
    foreach ($v in $choices) {
        $t = $(if ($script:Lang -ne "en" -and $spec.scope.labels.$v) { [string]$spec.scope.labels.$v }
               elseif ($null -ne $Str["wg_scope_" + $v]) { L ("wg_scope_" + $v) } else { [string]$v })
        $rb = New-Object System.Windows.Forms.RadioButton
        $rb.Text = $t
        $rb.AutoSize = $true
        $rb.Location = New-Object System.Drawing.Point($x, 3)
        $rb.Checked = ($v -eq $cur)
        $rb.Tag = @{ wgScope = $v }
        $rb.Add_CheckedChanged({ if ($this.Checked) { WG-OnScope $this.Tag.wgScope } })
        $rp.Controls.Add($rb)
        $modelTip.SetToolTip($rb, (L "wg_scope_tip"))
        $x += (Text-W $t) + 34
    }
    $y += 30
    if ($cur -eq "group") {
        [void](WG-Lbl $p (L "wg_group_srcs") 4 ($y + 4) 142 ([System.Drawing.SystemColors]::GrayText))
        $rf = WG-ReidFn
        $grp = @(WG-ReidGroup)
        $x = 150
        $maxX = [Math]::Max(300, $p.ClientSize.Width - 8)
        foreach ($s2 in @(Fn-Sources $rf)) {
            $w = (Text-W $s2) + 26
            if ($x + $w -gt $maxX) { $x = 150; $y += 26 }
            $cb = New-Object System.Windows.Forms.CheckBox
            $cb.Text = $s2
            $cb.AutoSize = $true
            $cb.Location = New-Object System.Drawing.Point($x, ($y + 2))
            $cb.Checked = ($grp -contains $s2)
            $cb.Tag = @{ wgGroup = $s2 }
            $cb.Add_CheckedChanged({ WG-OnGroup $this })
            $p.Controls.Add($cb)
            $x += $w + 8
        }
        $min = WG-GroupMin
        $note = (L "wg_need_min") -f $min
        if ($x + (Text-W $note) + 8 -gt $maxX) { $x = 150; $y += 26 }
        $nl = WG-Lbl $p $note ($x + 4) ($y + 4) ((Text-W $note) + 8) $WgRed
        $nl.Name = "wgGroupNote"
        $nl.Visible = ($grp.Count -lt $min)
        $y += 30
    }
    return ($y + 4)
}
function WG-OnScope($v) {
    if ($script:Building) { return }
    (WG-Fp "reid")["scope"] = $v
    Build-Detail
    Changed
}
function WG-OnGroup($cb) {
    if ($script:Building) { return }
    $fp = WG-Fp "reid"
    $cur = @(WG-ReidGroup)
    $s2 = $cb.Tag.wgGroup
    $cur = @($cur | Where-Object { $_ -ne $s2 })
    if ($cb.Checked) { $cur += $s2 }
    $fp["group_sources"] = @($cur)
    # 표시만 고친다(다시 그리면 누른 상자가 사라진다)
    foreach ($c in $cb.Parent.Controls) { if ($c.Name -eq "wgGroupNote") { $c.Visible = ($cur.Count -lt (WG-GroupMin)) } }
    Changed
}
function WG-DrawTrack($p, $y) {
    # 물체 추적 — 입력(켠 탐지 기능). 추적기 모델은 위 슬롯 줄이 맡는다
    $ch = @(WG-TrackChoices)
    if ($ch.Count -eq 0) { return $y }
    $k = WG-Lbl $p (L "wg_input") 4 ($y + 4) 142 ([System.Drawing.SystemColors]::GrayText)
    $modelTip.SetToolTip($k, (L "wg_input_tip"))
    $cb = New-Object System.Windows.Forms.ComboBox
    $cb.DropDownStyle = "DropDownList"
    $cb.Name = "wgInput"
    $cb.Location = New-Object System.Drawing.Point(150, $y)
    $cb.Width = 168
    $cur = WG-TrackInput
    foreach ($f in $ch) {
        $t = Fn-Text2 $f
        if (-not (Fn-On $f)) { $t += " (" + (L "wg_input_off") + ")" }
        [void]$cb.Items.Add($t)
    }
    $cb.SelectedIndex = [Math]::Max(0, [Array]::IndexOf($ch, $cur))
    $cb.Tag = @{ wgChoices = $ch }
    $cb.Add_SelectionChangeCommitted({ WG-OnInput $this })
    $modelTip.SetToolTip($cb, (L "wg_input_tip"))
    $p.Controls.Add($cb)
    return ($y + 32)
}
function WG-OnInput($cb) {
    if ($script:Building) { return }
    (WG-Fp "object_track")["input"] = [string]@($cb.Tag.wgChoices)[$cb.SelectedIndex]
    Changed
}
function WG-DrawWork($p, $fn, $y) {
    # 함께 쓰는 작업 — 이 기능이 받는 산출물마다 한 줄. 결과(work)가 없으면 그리지 않는다
    $mine = @(WG-Work | Where-Object { @($_.consumers) -contains $fn })
    if ($mine.Count -eq 0) { return $y }
    $y = WG-Head $p (L "wg_work_head") $y
    $tw = [Math]::Max(200, $p.ClientSize.Width - 32)
    foreach ($w in $mine) {
        $tip = (L "wg_tip_item") -f (WG-WorkText $w), $w.node, $w.fps, $w.gpu_mb
        if ($w.shared) { [void](WG-Icon $p 4 ($y + 2) ((L "wg_shared") + "`n" + $tip)) }
        $l = WG-Lbl $p (WG-WorkText $w) 24 $y $tw
        $l.Tag = @{ wgWork = $true; shared = [bool]$w.shared }
        $modelTip.SetToolTip($l, $tip)
        $y += 22
    }
    return ($y + 6)
}

# ---------------------------------------------------------------- 중복 알림 (기능 단계 왼쪽 아래)
function Hook-MatchFoot($g) {
    if ($null -eq $g -or $null -eq $matchPanel -or $null -eq $matchNote) { return }
    $box = $null
    foreach ($c in $g.Controls) { if ($c.Name -eq "wgWarn") { $box = $c } }
    if ($null -eq $box) {
        $box = New-Object System.Windows.Forms.Panel
        $box.Name = "wgWarn"
        $box.Anchor = "Bottom, Left, Right"
        $g.Controls.Add($box)
    }
    $box.SuspendLayout()
    $box.Controls.Clear()
    $ws = @(if ($visOn.Checked) { WG-Warnings })
    $y = 0
    $bw = $matchNote.Width
    foreach ($w in $ws) {
        $t = WG-WarnText $w
        $tw = [Math]::Min((Text-W $t) + 6, $bw - 80)
        $l = WG-Lbl $box $t 0 $y $tw $WgAmber
        $l.Tag = @{ wgWarn = $true }
        $tip = @()
        if ($w.reason) { $tip += [string]$w.reason }
        $fns = @($w.functions); $ms = @($w.models)
        for ($i = 0; $i -lt $fns.Count; $i++) { $tip += ("{0}: {1}" -f (Fn-Text2 $fns[$i]), $(if ($i -lt $ms.Count) { $ms[$i] } else { "-" })) }
        $modelTip.SetToolTip($l, ($tip -join "`n"))
        $lk = New-Object System.Windows.Forms.LinkLabel
        $lk.AutoSize = $true
        $lk.Text = (L "wg_align")
        $lk.Location = New-Object System.Drawing.Point(($tw + 4), $y)
        $lk.Tag = @{ wgWarnItem = $w }
        $plan = @(WG-AlignPlan $w)
        if ($plan.Count) {
            $modelTip.SetToolTip($lk, ((L "wg_align_tip") -f (WG-Names @($plan | ForEach-Object { $_.fn })), $plan[0].v))
            $lk.Add_LinkClicked({ [void](WG-Align $this.Tag.wgWarnItem) })
        } else {
            $lk.Enabled = $false
            $modelTip.SetToolTip($lk, (L "wg_align_none"))
        }
        $box.Controls.Add($lk)
        $y += 22
    }
    $h = $y
    $box.Size = New-Object System.Drawing.Size($bw, [Math]::Max(1, $h))
    $box.Location = New-Object System.Drawing.Point($matchNote.Left, ($matchNote.Top - $h - 2))
    $box.Visible = ($h -gt 0)
    $box.ResumeLayout()
    # 격자는 알림 줄만큼 줄어든다 (격자 아래 끝 = 알림 위)
    $matchPanel.Height = [Math]::Max(60, $matchNote.Top - 4 - $(if ($h) { $h + 2 } else { 0 }) - $matchPanel.Top)
}

# ---------------------------------------------------------------- 프로필 · 명령줄 · 설정
function Hook-SettingsLoad($saved) {
    $script:FnParams = $(if ($null -ne $saved) { WG-Hash $saved.functionParams } else { @{} })
}
function Hook-PlacesLoaded($r) { $script:WgPlaces = $r }
function Hook-ProfileExtra($prof) {
    if ($null -eq (WG-Spec)) { return }
    $fp = WG-Params
    if ($fp.Count) { $prof["function_params"] = $fp }
}
function Hook-OrchArgs($profWsl) {
    # 기본값과 다를 때만 — 아니면 명령줄이 예전 그대로다(백엔드가 이 인자를 몰라도 깨지지 않게)
    if ($null -eq (WG-Spec)) { return "" }
    $fp = WG-Params $true
    if ($fp.Count -eq 0) { return "" }
    $j = ($fp | ConvertTo-Json -Compress -Depth 6) -replace '"', '\"'
    # 미리보기(& wsl.exe 인자 배열): \" 만. 실행(Start-WslWindow 의 bash -c "…"): bash 가 글 그대로 받게 작은따옴표로
    if ($profWsl -ne $PreviewWsl) { $j = "'" + $j + "'" }
    return " --function-params " + $j
}

# ---------------------------------------------------------------- 배치 미리보기 카드
function Hook-PreviewCards($body, $p) {
    $pl = $(if ($null -ne $p) { WG-PlanOf $p } else { $null })
    if ($null -eq $pl) { return }
    $script:WgPlan = $pl
    $script:WgAligned = @{}
    $work = @($pl.work | Where-Object { $_ })
    $warns = @($pl.work_warnings | Where-Object { $_ })
    $shift = 0
    foreach ($c in @($body.Controls | Sort-Object Top)) {
        if ($shift) { $c.Top += $shift }
        if (-not ($c -is [System.Windows.Forms.Panel]) -or $c.Controls.Count -eq 0) { continue }
        $node = ([string]$c.Controls[0].Text -split "  ")[0]
        $mine = @($work | Where-Object { [string]$_.node -eq $node })
        if ($mine.Count -eq 0) { continue }
        $sh = @($mine | Where-Object { $_.shared }).Count
        $dupW = @($warns | Where-Object { $ww = $_; @($mine | Where-Object { $_.source -eq $ww.source -and $_.artifact -eq $ww.artifact }).Count })
        $t = $(if ($dupW.Count) { (L "wg_card") -f $sh, $dupW.Count } else { (L "wg_card0") -f $sh })
        $tip = @($mine | ForEach-Object { $(if ($_.shared) { "● " } else { "   " }) + (WG-WorkText $_) })
        foreach ($ww in $dupW) { $tip += ("⚠ " + (WG-WarnText $ww)) }
        $cy = $c.Height - 2
        [void](WG-Icon $c 8 ($cy + 1) ($tip -join "`n"))
        $l = WG-Lbl $c $t 26 $cy ([Math]::Max(120, $c.Width - 36)) $(if ($dupW.Count) { $WgAmber } else { [System.Drawing.Color]::DimGray })
        $l.Tag = @{ wgCard = $node }
        $modelTip.SetToolTip($l, ($tip -join "`n"))
        $c.Height += 20
        $shift += 20
    }
    # 기능 단계도 같은 결과로 다시 그린다
    if ($null -ne $gMatch -and (Get-Command Update-GpuTotal -ErrorAction SilentlyContinue)) { Hook-MatchFoot $gMatch }
    if ($null -ne $infoPanel -and $null -ne $script:FnList -and (Get-Command Build-Detail -ErrorAction SilentlyContinue)) { Build-Detail }
}

# ---------------------------------------------------------------- 자체 시험
function WG-Fake($srcs) {
    $a, $b, $c = $srcs[0], $srcs[1], $srcs[2]
    $places = [ordered]@{
        function_params = [ordered]@{
            reid = [ordered]@{ scope = [ordered]@{ choices = @("source", "group"); default = "source"; labels = [ordered]@{ source = "소스 안"; group = "소스 묶음" } }
                               group_sources = [ordered]@{ min = 2 } }
            object_track = [ordered]@{ input = [ordered]@{ choices = @("object_detect", "zeroshot_detect", "targeted_detect"); default = "object_detect" } } }
        artifacts = [ordered]@{
            depth_estimate = @{ produces = @("depth"); consumes = @("depth") }
            targeted_detect = @{ produces = @("detections"); consumes = @("detections", "depth") }
            object_detect = @{ produces = @("detections"); consumes = @("detections") }
            single_cam_track = @{ produces = @("tracks"); consumes = @("detections") }
            reid = @{ produces = @("embeddings"); consumes = @("detections") } }
    }
    $plan = [ordered]@{
        gpu_units = @([ordered]@{ unit = "gpu:work"; node = "book-4vg1edpvi5"; models = @("yoloe", "da3_metric_l", "unidepth"); sources = @($a, $b)
                                  fps_cap = [ordered]@{ $a = 10.0; $b = 2.0 }; need_mb = 4400; need_basis = "estimate"; limit_mode = "none" })
        cpu_units = @(); refused = @(); degraded = @(); waiting = @(); mode = "k3s"; basis = "live"
        work = @(
            [ordered]@{ source = $a; node = "book-4vg1edpvi5"; artifact = "depth"; model = "da3_metric_l"; stage = "server"; fps = 2.0
                        consumers = @("depth_estimate", "targeted_detect"); shared = $true; gpu_mb = 2000; unit = "work:depth@$a" },
            [ordered]@{ source = $b; node = "book-4vg1edpvi5"; artifact = "depth"; model = "unidepth"; stage = "server"; fps = 2.0
                        consumers = @("depth_estimate"); shared = $false; gpu_mb = 1500; unit = "work:depth@$b" },
            [ordered]@{ source = $a; node = "book-4vg1edpvi5"; artifact = "detections"; model = "yoloe"; stage = "server"; fps = 10.0
                        consumers = @("object_detect", "single_cam_track", "reid"); shared = $true; gpu_mb = 900; unit = "work:det@$a" })
        work_warnings = @(
            [ordered]@{ source = $a; artifact = "depth"; models = @("da3_metric_l", "depth"); functions = @("depth_estimate", "targeted_detect")
                        save_mb = 2048; code = "dup_kind"; reason = "같은 소스의 거리를 두 모델이 따로 낸다" })
        nodes = @([ordered]@{ name = "book-4vg1edpvi5"; role = "gpu"; ready = $true; stale = $false; source = $null })
    }
    return @{ places = ($places | ConvertTo-Json -Depth 8 | ConvertFrom-Json); plan = ($plan | ConvertTo-Json -Depth 8 | ConvertFrom-Json) }
}
function PartTest-WorkGraph {
    $r = [ordered]@{}
    $keep = @{ places = $script:WgPlaces; plan = $script:WgPlan; fp = $script:FnParams; fnSrc = $script:FnSrc.Clone()
               fnModel = $script:FnModel.Clone(); cuePick = $script:CuePick.Clone(); sel = $script:SelFn; vis = $visOn.Checked
               checks = @{} }
    foreach ($k in $srcChecks.Keys) { $keep.checks[$k] = $srcChecks[$k].Checked }
    $script:WgNoReplan = $true
    try {
        # 데이터가 없으면 아무것도 그리지 않는다
        $script:WgPlaces = $null; $script:WgPlan = $null
        $savedPlaces = $script:Places; $script:Places = $null
        $hp = @{}; Hook-ProfileExtra $hp
        $r.hidden_without_spec = ($hp.Count -eq 0 -and (Hook-OrchArgs $PreviewWsl) -eq "")
        $srcs = @("pi12", "robot1", "pi13")
        foreach ($k in $srcChecks.Keys) { $srcChecks[$k].Checked = ($srcs -contains $k) }
        $visOn.Checked = $true
        $fk = WG-Fake $srcs
        $script:WgPlaces = $fk.places
        $script:FnParams = @{ reid = @{ scope = "group" } }
        $script:FnSrc = @{ depth_estimate = @($srcs[0], $srcs[1]); targeted_detect = @($srcs[0]); object_detect = @($srcs[0])
                           zeroshot_detect = @($srcs[0]); single_cam_track = @($srcs[0]); reid = @($srcs) }
        $script:FnModel["depth_estimate"] = "unidepth"
        $script:CuePick["depth"] = "auto"
        $script:SelFn = "reid"
        Build-Match; Build-CueTable; Build-CueBoxes
        # 미리보기 결과가 오면 카드에 공유 줄 · 기능 단계에 알림
        $hostP = New-Object System.Windows.Forms.Panel
        $hostP.Font = $font
        $hostP.Size = New-Object System.Drawing.Size(616, 360)
        $cnt = Hook-PreviewShow $hostP $fk.plan
        $pvBody = $hostP.Controls["pvBox"].Controls["pvBody"]
        $cardL = @(All-Ctls $pvBody | Where-Object { $_.Tag -is [hashtable] -and $_.Tag.wgCard })
        $r.card_line = @($cardL | ForEach-Object { "{0}: {1}" -f $_.Tag.wgCard, $_.Text })
        $r.card_tip = $(if ($cardL.Count) { $modelTip.GetToolTip($cardL[0]) } else { "" })
        if ($env:DS_LAUNCHER_TESTSHOT) {
            $bmp = New-Object System.Drawing.Bitmap $hostP.Width, $hostP.Height
            $hostP.DrawToBitmap($bmp, (New-Object System.Drawing.Rectangle 0, 0, $hostP.Width, $hostP.Height))
            $bmp.Save((Join-Path $env:DS_LAUNCHER_TESTSHOT "wg_preview.png"), [System.Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
        }
        $hostP.Dispose()
        # 중복 알림
        $warn = @($gMatch.Controls["wgWarn"].Controls)
        $r.warn_lines = @($warn | Where-Object { $_.Tag -is [hashtable] -and $_.Tag.wgWarn } | ForEach-Object { $_.Text })
        $r.warn_link = @($warn | Where-Object { $_ -is [System.Windows.Forms.LinkLabel] } | ForEach-Object { "{0} enabled={1}" -f $_.Text, $_.Enabled })
        $r.match_above_warn = ($matchPanel.Bottom -le $gMatch.Controls["wgWarn"].Top)
        # 재식별 — 소스 묶음, 소스 셋
        Select-Fn "reid"
        $wd = $detailBody.Controls["wgDetail"]
        $r.reid_radios = @(All-Ctls $wd | Where-Object { $_ -is [System.Windows.Forms.RadioButton] } | ForEach-Object { "{0}={1}" -f $_.Text, $_.Checked })
        $gcs = @(All-Ctls $wd | Where-Object { $_ -is [System.Windows.Forms.CheckBox] -and $_.Tag -is [hashtable] -and $_.Tag.wgGroup })
        $r.reid_group = @($gcs | ForEach-Object { "{0}={1}" -f $_.Text, $_.Checked })
        $r.reid_note_hidden_3 = -not (@(All-Ctls $wd | Where-Object { $_.Name -eq "wgGroupNote" })[0]).Visible
        $r.detail_below = ($wd.Top -ge $tgtSection.Bottom)
        if ($env:DS_LAUNCHER_TESTSHOT) {
            $form.StartPosition = "Manual"
            $form.Location = New-Object System.Drawing.Point(-20000, -20000)
            $form.Show()
            $tabs.SelectedIndex = 1
            [System.Windows.Forms.Application]::DoEvents()
            $detailBody.AutoScrollPosition = New-Object System.Drawing.Point(0, 10000)
            Save-Shot "wg_reid.png"
        }
        $gcs[1].Checked = $false; $gcs[2].Checked = $false
        $r.reid_note_shown_1 = (@(All-Ctls $wd | Where-Object { $_.Name -eq "wgGroupNote" })[0]).Visible
        $gcs[1].Checked = $true
        $prof = Build-Profile
        $r.profile_fp = ($prof.function_params | ConvertTo-Json -Compress -Depth 6)
        $r.profile_json_has = ((($prof | ConvertTo-Json -Depth 8) -match '"function_params"') -and $prof.function_params.reid.scope -eq "group")
        $r.plan_args_fp = (Orch-Args (Current-Config) $PreviewWsl) -replace '^.*(--function-params \S+).*$', '$1'
        $r.run_args_fp = (Orch-Args (Current-Config)) -replace '^.*(--function-params \S+).*$', '$1'
        # 물체 추적 — 입력
        $tf = WG-TrackFn
        $r.track_fn = $tf
        Select-Fn $tf
        $ic = @(All-Ctls $wd | Where-Object { $_.Name -eq "wgInput" })[0]
        $r.track_input_items = @($ic.Items)
        $r.track_input = $ic.Text
        $r.track_detail_below = ($wd.Top -ge $infoPanel.Bottom)
        if ($env:DS_LAUNCHER_TESTSHOT) { [System.Windows.Forms.Application]::DoEvents(); Save-Shot "wg_track.png" }
        $ic.SelectedIndex = 1
        WG-OnInput $ic
        $r.track_profile = ((Build-Profile).function_params.object_track | ConvertTo-Json -Compress)
        # 함께 쓰는 작업 — 거리 추정
        Select-Fn "depth_estimate"
        $r.work_rows = @(All-Ctls $wd | Where-Object { $_.Tag -is [hashtable] -and $_.Tag.wgWork } | ForEach-Object { "{0}{1}" -f $(if ($_.Tag.shared) { "[공유] " } else { "" }), $_.Text })
        $r.work_icons = @(All-Ctls $wd | Where-Object { $_ -is [System.Windows.Forms.PictureBox] }).Count
        if ($env:DS_LAUNCHER_TESTSHOT) { [System.Windows.Forms.Application]::DoEvents(); Save-Shot "wg_work.png" }
        # 맞추기 → 지정 물체 탐지의 거리 슬롯이 첫 기능의 모델로, 알림이 사라진다
        $w0 = @(WG-Warnings)[0]
        $r.align_ok = WG-Align $w0
        $r.align_cuepick_depth = $script:CuePick["depth"]
        $r.warn_after_align = @($gMatch.Controls["wgWarn"].Controls).Count
        if ($env:DS_LAUNCHER_TESTSHOT) { $form.Hide() }
        $script:Places = $savedPlaces
    } finally {
        $script:WgNoReplan = $false
        $script:WgPlaces = $keep.places; $script:WgPlan = $keep.plan; $script:FnParams = $keep.fp
        $script:FnSrc = $keep.fnSrc; $script:FnModel = $keep.fnModel; $script:CuePick = $keep.cuePick; $script:SelFn = $keep.sel
        foreach ($k in $keep.checks.Keys) { $srcChecks[$k].Checked = $keep.checks[$k] }
        $visOn.Checked = $keep.vis
        Build-Match; Build-CueTable; Build-CueBoxes
    }
    return $r
}
