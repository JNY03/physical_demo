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
#:   { "distro": "Ubuntu-26.04", "repoDir": "/home/<사용자>/data_stream", "profileWsl": "/mnt/c/Users/<사용자>/DataStreamLauncher/profile.json",
#:     "pythonWsl": "./venv/bin/python" }
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
#: 저장소 안에서 자식을 띄울 Python (컨테이너 · 다른 기기는 venv 가 없을 수 있다 — ds_resources.settings 의 DS_PYTHON 과 같은 뜻)
$PyWsl        = Pick-Setting "DS_WSL_PYTHON" "pythonWsl" { "./venv/bin/python" }
$ProfileWsl   = Pick-Setting "DS_WSL_PROFILE" "profileWsl" {
    # 이 폴더가 Windows 드라이브면 /mnt/<드라이브>/…, \\wsl.localhost\<배포판>\… 면 그 뒤가 곧 리눅스 자리다
    $m = [regex]::Match($ProfileFile, '^\\\\wsl(?:\.localhost|\$)\\[^\\]+(\\.*)$')
    if ($m.Success) { ($m.Groups[1].Value -replace '\\', '/') }
    else { "/mnt/" + $ProfileFile.Substring(0, 1).ToLower() + ($ProfileFile.Substring(2) -replace '\\', '/') }
}
# 「배치 보기」 전용 — 비전이 지켜보는 profile.json 은 「실행」(Start-All · Apply-Changes) 때만 바뀐다 (M11 R4)
$PreviewFile  = Join-Path $Here "profile_preview.json"
$PreviewWsl   = $ProfileWsl -replace "profile\.json$", "profile_preview.json"
#: 비전 GPU 메모리 어림(MB)은 vision/model_registry.yaml 하나에서 온다 (카탈로그 gpu_mb_base · gpu_mb_option · gpu_mb_targeted)

function From-Json($txt) {
    # 도구가 --json 과 함께 로그를 한 줄 찍는 경우가 있다 — 첫 `{` 부터 마지막 `}` 까지만 떼어 읽는다.
    # 못 읽으면 $null (부르는 쪽이 원문을 사람에게 보여 준다)
    if (-not $txt) { return $null }
    $s = [string]$txt
    $i = $s.IndexOf("{")
    $j = $s.LastIndexOf("}")
    if ($i -lt 0 -or $j -le $i) { return $null }
    try { return ($s.Substring($i, $j - $i + 1) | ConvertFrom-Json) } catch { return $null }
}
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
    [void][System.Windows.Forms.MessageBox]::Show("선택지를 못 읽었습니다. / Could not read the options.", "Data Stream")
    exit 1
}
$Models = @($cat.models | ForEach-Object { $_.key })
$ModelShown = @{}; foreach ($m in $cat.models) { $ModelShown[$m.key] = $m.shown }
$TargetDb = @{}; foreach ($t in $cat.targets) { $TargetDb[$t.name] = $t }

# ---- 화면 글 (한국어 · 영어, 2026-11-26)
# 화면 글만 옮긴다. 모델 이름 · 근거 이름 · 대상 이름은 카탈로그(vision/cue_catalog.py)가 주는 그대로 쓴다 —
# 거기까지 옮기려면 비전 쪽에 영문 칸을 더해야 하고, 가시화가 읽는 이름과 어긋날 수 있다.
$Str = @{
    d_title      = @{ ko = "데이터 스트림 실행"; en = "Data Stream" }
    step_src     = @{ ko = "소스"; en = "Sources" }
    step_func    = @{ ko = "기능"; en = "Capabilities" }
    step_target  = @{ ko = "지정 클래스"; en = "Target classes" }
    step_over    = @{ ko = "오버레이"; en = "Overlay" }
    step_place   = @{ ko = "배치"; en = "Placement" }
    g_stream     = @{ ko = "스트림"; en = "Stream" }
    g_vision     = @{ ko = "인지"; en = "Perception" }
    g_cues       = @{ ko = "근거"; en = "Evidence" }
    g_match      = @{ ko = "기능 소스별 배치"; en = "Capabilities per source" }
    g_place      = @{ ko = "배치"; en = "Placement" }
    g_target     = @{ ko = "대상"; en = "Targets" }
    l_sources    = @{ ko = "소스"; en = "Sources" }
    l_robotcam   = @{ ko = "로봇 캠"; en = "Robot cam" }
    l_slice      = @{ ko = "슬라이스"; en = "Slice" }
    l_on         = @{ ko = "켜기"; en = "On" }
    l_capability = @{ ko = "기능"; en = "Capability" }
    l_extra      = @{ ko = "확장"; en = "Extras" }
    l_node       = @{ ko = "노드"; en = "Nodes" }
    l_task       = @{ ko = "작업"; en = "Tasks" }
    l_lang       = @{ ko = "한국어"; en = "English" }
    b_plan       = @{ ko = "배치 미리보기"; en = "Preview the plan" }
    b_run        = @{ ko = "실행"; en = "Run" }
    b_prev       = @{ ko = "이전"; en = "Back" }
    b_next       = @{ ko = "다음"; en = "Next" }
    b_reset      = @{ ko = "선택 지우기"; en = "Clear picks" }
    b_load       = @{ ko = "노드 다시 읽기"; en = "Reload nodes" }
    b_register   = @{ ko = "등록"; en = "Register" }
    b_add        = @{ ko = "추가"; en = "Add" }
    c_orch       = @{ ko = "자원 배분"; en = "Resource-aware" }
    c_distance   = @{ ko = "거리 측정"; en = "Distance" }
    r_auto       = @{ ko = "자동"; en = "Automatic" }
    r_manual     = @{ ko = "수동"; en = "Manual" }
    r_default    = @{ ko = "기본값"; en = "Default" }
    r_custom     = @{ ko = "직접 고르기"; en = "Choose" }
    h_capability = @{ ko = "기능"; en = "Capability" }
    h_cue        = @{ ko = "근거"; en = "Evidence" }
    h_model      = @{ ko = "모델"; en = "Model" }
    h_gpu        = @{ ko = "예상 GPU"; en = "Est. GPU" }
    h_device     = @{ ko = "기기"; en = "Device" }
    h_role       = @{ ko = "역할"; en = "Role" }
    h_state      = @{ ko = "상태"; en = "State" }
    h_limit      = @{ ko = "한도"; en = "Limit" }
    h_where      = @{ ko = "허용"; en = "Allowed" }
    h_slot       = @{ ko = "자리"; en = "Node" }
    v_auto       = @{ ko = "자동"; en = "Auto" }
    v_none       = @{ ko = "제외"; en = "Off" }
    v_novision   = @{ ko = "없음"; en = "none" }
    v_usable     = @{ ko = "정상"; en = "ready" }
    v_notready   = @{ ko = "불가"; en = "down" }
    v_limited    = @{ ko = "있음"; en = "yes" }
    v_unlimited  = @{ ko = "없음"; en = "no" }
    v_waived     = @{ ko = "면제"; en = "waived" }
    v_rule       = @{ ko = "규칙 (모델 없음)"; en = "rule (no model)" }
    v_missing    = @{ ko = "없음"; en = "missing" }
    n_match      = @{ ko = "기능을 고르고 올릴 기기에 체크하세요."
                      en = "Pick a capability, then tick the sources to run it on." }
    n_cues       = @{ ko = "근거는 기능이 정한다. 지정 클래스의 근거는 3단계에서 대상마다 고른다."
                      en = "Evidence comes with the capability. For target classes, choose it per target in step 3." }
    n_manual     = @{ ko = "「불러오기」를 누르세요."; en = 'Press the "Load nodes / tasks" button.' }
    n_novision   = @{ ko = "인지를 켜면 고를 수 있습니다."; en = "Turn perception on to choose." }
    n_nomodel    = @{ ko = "기능 탭에서 기능을 먼저 고르세요."; en = "Choose a capability in the Capabilities tab." }
    n_nosource   = @{ ko = "소스 탭에서 소스를 먼저 고르세요."; en = "Choose a source in the Sources tab." }
    n_total      = @{ ko = "프로세스 하나 기본 {0} MB + 고른 기능 {1} MB = 어림 {2} MB"
                      en = "{0} MB per process + {1} MB for chosen capabilities = about {2} MB" }
    s_running    = @{ ko = "● 실행 중"; en = "● Running" }
    s_starting   = @{ ko = "띄우는 중"; en = "Starting" }
    s_stopping   = @{ ko = "끄는 중"; en = "Stopping" }
    s_restarting_server = @{ ko = "서버를 다시 띄우는 중"; en = "Restarting the server" }
    s_restarting_vision = @{ ko = "인지를 다시 띄우는 중"; en = "Restarting perception" }
    s_plan_calc  = @{ ko = "배치 계산 중"; en = "Calculating the plan" }
    s_loading    = @{ ko = "노드와 작업 읽는 중"; en = "Loading nodes and tasks" }
    m_need_source = @{ ko = "소스를 하나 이상 고르세요."; en = "Choose at least one source." }
    m_need_model  = @{ ko = "기능 탭에서 기능을 하나 이상 고르세요 (기기 칸에 체크)."
                      en = "Choose at least one capability (tick a device column)." }
    m_need_target = @{ ko = "지정 클래스 탐지에는 대상이 하나 이상 필요합니다."; en = "Target-class detection needs at least one target." }
    m_need_cam360 = @{ ko = "360 슬라이스는 cam360 소스도 켜야 합니다."; en = "The 360 slice needs the cam360 source too." }
    m_slice_u     = @{ ko = "슬라이스 u 는 숫자로 적으세요."; en = "Slice u must be a number." }
    m_match_orch  = @{ ko = "기기별 기능 매칭은 자원 배분을 켠 채로 씁니다."; en = "Per-device capabilities need resource-aware placement." }
    m_manual_orch = @{ ko = "수동 배치는 자원 배분을 켠 채로 씁니다."; en = "Manual placement needs resource-aware placement." }
    m_match_empty = @{ ko = "기기별 기능에서 모든 기기를 비웠습니다 — 하나 이상은 맞추세요."
                        en = "Every device was cleared — leave at least one with a capability." }
    m_stop        = @{ ko = "데이터 스트림을 끕니다."; en = "This stops the data stream." }
    m_no_plan     = @{ ko = "배치를 계산하지 못했습니다 (WSL, venv 확인)."; en = "Could not compute the plan (check WSL, venv)." }
    m_no_places   = @{ ko = "노드와 작업을 읽지 못했습니다 (WSL, venv 확인)."; en = "Could not read nodes and tasks (check WSL, venv)." }
    m_pick_both   = @{ ko = "소스와 기능을 고르세요."; en = "Choose a source and a capability." }
    m_name_en     = @{ ko = "영문 이름으로 적으세요 (예: traffic cone)."; en = "Use an English name (e.g. traffic cone)." }
    m_size_mm     = @{ ko = "크기는 가로x세로x높이 mm (예: 300x300x700)"; en = "Size is WxDxH in mm (e.g. 300x300x700)" }
    m_saved       = @{ ko = "저장했습니다. 다음 실행부터 목록에 나옵니다."; en = "Saved. It appears in the list from the next run." }
    m_save_fail   = @{ ko = "저장 실패"; en = "Could not save" }
    n_cue_tip     = @{ ko = "근거마다 쓰는 모델"; en = "Model used by each piece of evidence" }
    n_gpu_est     = @{ ko = "GPU 어림: +{0} MB"; en = "Est. GPU: +{0} MB" }
    n_gpu_sum     = @{ ko = "GPU 어림: 고른 단서의 모델 합"; en = "Est. GPU: sum of the chosen evidence models" }
    h_size        = @{ ko = "크기"; en = "Size" }
    h_colour      = @{ ko = "색"; en = "Colour" }
    h_cards       = @{ ko = "카드"; en = "Cards" }
    lbl_name      = @{ ko = "클래스명"; en = "Class" }
    lbl_conf      = @{ ko = "신뢰도"; en = "Confidence" }
    lbl_id        = @{ ko = "트랙 번호"; en = "Track id" }
    lbl_dist      = @{ ko = "거리"; en = "Distance" }
    lbl_gps       = @{ ko = "GPS"; en = "GPS" }
    flg_count     = @{ ko = "탐지 개수"; en = "Detection count" }
    flg_roi       = @{ ko = "탐지 영역"; en = "Region" }
    flg_nonsus    = @{ ko = "판정 아닌 검출"; en = "Non-suspect detections" }
    flg_rejected  = @{ ko = "관문에 떨어진 검출"; en = "Gated-out detections" }
    flg_title     = @{ ko = "제목 띠"; en = "Title bar" }
    flg_pose      = @{ ko = "드론 위치 띠"; en = "Drone pose bar" }
    s_not_applied = @{ ko = "반영 안 됨"; en = "not applied" }
    w_server      = @{ ko = "서버"; en = "server" }
    w_ondevice    = @{ ko = "온디바이스"; en = "on-device" }
    w_sensor      = @{ ko = "센서 노드"; en = "sensor node" }
    w_small       = @{ ko = "보조 노드"; en = "small node" }
    h_role2       = @{ ko = "역할"; en = "Role" }
    h_purpose     = @{ ko = "작업"; en = "Task" }
    h_res_free    = @{ ko = "여유"; en = "Free" }
    h_res_need    = @{ ko = "필요"; en = "Needs" }
    w_cores       = @{ ko = "코어"; en = "cores" }
    t_orch        = @{ ko = "자원에 맞춰 배치한다 — GPU 여유 확인, fps 상한, k3s 가 있으면 파이까지"
                      en = "Place by available resources — GPU room, fps cap, and the Pi nodes when k3s is up" }
    t_auto        = @{ ko = "배치 계산기가 자리를 고른다"; en = "The planner picks the node" }
    t_manual      = @{ ko = "작업마다 기기를 고른다. 고른 자리는 한도를 면제하지만 센서와 가속기 조건은 지킨다"
                      en = "Pick a node per task. A picked node waives limits but still needs the right sensors" }
    t_plan        = @{ ko = "띄우지 않고 배치 결과만 미리 본다"; en = "Show the plan without starting anything" }
    t_slice       = @{ ko = ("360 파노라마(cam360)의 한 방향을 잘라 보통 화면처럼 쓰는 기능입니다. " +
                             "잘린 화면은 cam360_slice 라는 소스로 따로 나오고, 왜곡이 적어 인지가 훨씬 잘 찾습니다. " +
                             "u 는 좌우 각도(0 이 파노라마 가운데, + 가 오른쪽), v 는 위아래로 -30 고정입니다. " +
                             "cam360 소스도 함께 켜야 합니다.")
                      en = ("Cuts one direction out of the 360 panorama (cam360) so it looks like a normal camera. " +
                            "The cut view shows up as its own source, cam360_slice, and vision reads it far better. " +
                            "u is the left-right angle (0 is the panorama centre), v is fixed at -30. " +
                            "The cam360 source must be on too.") }
    b_weights     = @{ ko = "모델 추가"; en = "Add models" }
    b_put         = @{ ko = "파일에서 넣기"; en = "From a file" }
    b_add_model   = @{ ko = "파일에서 추가"; en = "Add from a file" }
    h_perf        = @{ ko = "성능"; en = "Speed" }
    h_prompt      = @{ ko = "프롬프트"; en = "Prompt" }
    s_testing     = @{ ko = "적재 시험 중"; en = "Testing the load" }
    s_probe       = @{ ko = "소스 확인 중"; en = "Checking sources" }
    b_recheck     = @{ ko = "연결 재확인"; en = "Recheck links" }
    b_src_add     = @{ ko = "소스 추가"; en = "Add a source" }
    d_src_add     = @{ ko = "소스 추가"; en = "Add a source" }
    l_src_name    = @{ ko = "이름"; en = "Name" }
    l_src_host    = @{ ko = "주소"; en = "Address" }
    l_src_cam     = @{ ko = "카메라 포트"; en = "Camera port" }
    l_src_path    = @{ ko = "경로"; en = "Path" }
    l_src_note    = @{ ko = "메모"; en = "Note" }
    b_src_test    = @{ ko = "연결 시험"; en = "Test the link" }
    b_src_save    = @{ ko = "넣기"; en = "Add" }
    l_src_force   = @{ ko = "기기를 아직 켜지 않았습니다 (시험 없이 넣기)"
                       en = "The device is not on yet (add without a test)" }
    s_src_test    = @{ ko = "연결 시험 중"; en = "Testing the link" }
    m_src_ok      = @{ ko = "장 하나를 받았습니다: {0}x{1}, {2} 바이트 ({3} ms)"
                       en = "Got one frame: {0}x{1}, {2} bytes ({3} ms)" }
    m_src_bad     = @{ ko = "안 됩니다: {0}"; en = "No luck: {0}" }
    m_src_saved   = @{ ko = "넣었습니다: {0} (데이터 서버 포트 {1})"
                       en = "Added: {0} (data server port {1})" }
    n_src_add     = @{ ko = ("이름은 영문 소문자로 시작합니다. 주소는 그 기기의 IP 나 이름, " +
                             "카메라 포트는 영상이 나오는 포트입니다 (라즈베리파이와 go1 은 8090, ep-001 은 8091). " +
                             "먼저 연결 시험을 눌러 장이 오는지 보고 넣으십시오.")
                       en = ("The name starts with a lowercase letter. The address is the device IP or host name, " +
                             "and the camera port is where its video comes out (8090 on the Raspberry Pi and go1, " +
                             "8091 on ep-001). Press the test first to see whether a frame arrives.") }
    n_src_restart = @{ ko = ("넣은 소스는 데이터 스트림을 다시 시작해야 들어옵니다. 포트가 하나 늘어나니 " +
                             "가시화 쪽에도 그 포트를 알려 주십시오.")
                       en = ("A new source arrives only after the data stream restarts. It opens one more port, " +
                             "so tell the visualization side about it.") }
    b_detail      = @{ ko = "세부 설정"; en = "Settings" }
    b_limit       = @{ ko = "자원 한도"; en = "Resource limits" }
    b_preview     = @{ ko = "탐지 시험"; en = "Test detection" }
    b_preview_overlay = @{ ko = "오버레이 미리보기"; en = "Preview the overlay" }
    n_over_pick   = @{ ko = ("아래 「오버레이 미리보기」로 그림을 한 번 고르면 여기에 보입니다. " +
                             "그 뒤로는 오버레이를 고칠 때마다 다시 그립니다.")
                       en = ("Pick an image once with the button below and it shows here. " +
                             "After that it is redrawn whenever you change the overlay.") }
    s_over_draw   = @{ ko = "오버레이 다시 그리는 중"; en = "Redrawing the overlay" }
    b_browse      = @{ ko = "파일 고르기"; en = "Choose a file" }
    b_run_preview = @{ ko = "시험 시작"; en = "Start" }
    l_image       = @{ ko = "이미지 경로"; en = "Image path" }
    l_compare     = @{ ko = "지금 돌고 있는 설정과 비교"; en = "Compare with what is running" }
    t_compare     = @{ ko = ("켜면 같은 그림에 둘을 돌려 나란히 보여 줍니다 — 지금 고른 설정과, " +
                             "실행 중인 설정(profile.json). 무엇이 늘고 줄었는지 글로도 적힙니다.")
                      en = ("Runs both on the same image side by side: what you picked now, and what is running. " +
                            "The caption says what changed.") }
    n_image       = @{ ko = ("데이터 서버가 저장한 장을 쓰거나(stream_data/<소스>/<실행>/frames/…) " +
                             "원하는 사진을 골라도 됩니다. 폴더를 적으면 그 안에서 고릅니다.")
                      en = ("Use a frame the data server saved (stream_data/<source>/<run>/frames/...) " +
                            "or pick any photo. A folder works too.") }
    s_preview     = @{ ko = "이미지로 돌려 보는 중"; en = "Running on the image" }
    m_preview_fail = @{ ko = "이미지로 돌려 보지 못했습니다."; en = "Could not run on that image." }
    m_need_image  = @{ ko = "이미지 경로를 적거나 파일을 고르세요."; en = "Type an image path or pick a file." }
    m_no_image    = @{ ko = "그 자리에 파일이 없습니다: {0}"; en = "No such file: {0}" }
    m_no_preview_tool = @{ ko = ("미리보기 도구가 아직 없습니다 (tools/preview_apply.py). " +
                                 "그 도구가 들어오면 이 단추가 바로 됩니다.")
                          en = ("The preview tool is not in place yet (tools/preview_apply.py). " +
                                "This button works as soon as it lands.") }
    b_save        = @{ ko = "저장"; en = "Save" }
    h_lim_cpu     = @{ ko = "CPU 여유 %"; en = "CPU headroom %" }
    h_lim_mem     = @{ ko = "메모리 여유 %"; en = "Memory headroom %" }
    h_lim_cores   = @{ ko = "비울 코어"; en = "Reserved cores" }
    h_lim_temp    = @{ ko = "온도 한계 ℃"; en = "Temp limit C" }
    h_lim_gpu     = @{ ko = "GPU 예비 MB"; en = "GPU spare MB" }
    t_lim_cpu     = @{ ko = "그 기기에서 비워 둘 CPU 비율"; en = "Share of CPU to keep free" }
    t_lim_mem     = @{ ko = "비워 둘 메모리 비율"; en = "Share of memory to keep free" }
    t_lim_cores   = @{ ko = "그 기기가 제 일(제어, 센서)에 쓸 코어 수"; en = "Cores left for the device's own work" }
    t_lim_temp    = @{ ko = "이 온도를 넘으면 새 일을 두지 않는다"; en = "No new work above this temperature" }
    t_lim_gpu     = @{ ko = "GPU 에 남겨 둘 메모리"; en = "GPU memory to leave free" }
    t_limit       = @{ ko = "이 노드에 걸 한도를 고친다 (서버도 걸 수 있다)"; en = "Set the limits for this node (the server too)" }
    n_limit       = @{ ko = ("빈 칸은 「없음」입니다. 서버는 기본이 없음이고, 파이는 CPU 여유 20% 가 기본입니다. " +
                             "반드시 그 기기에서 해야 하는 일(안전, 기기 고정 임무)은 한도를 걸어도 면제입니다.")
                      en = ("An empty box means no limit. The server has none by default; a Pi keeps 20% CPU free. " +
                            "Work that must stay on the device (safety, pinned mission) is waived even so.") }
    m_limit_num   = @{ ko = "{0} 는 숫자로 적으세요."; en = "{0} must be a number." }
    l_slice_on    = @{ ko = "슬라이스 켜기"; en = "Slice on" }
    l_cam         = @{ ko = "cam"; en = "cam" }
    b_cue_detail  = @{ ko = "글과 사진 보기"; en = "Text and photos" }
    b_check_prompt = @{ ko = "글 검사"; en = "Check" }
    b_add_refs    = @{ ko = "기준 사진 넣기"; en = "Add photos" }
    b_add_feat    = @{ ko = "추가"; en = "Add" }
    n_features    = @{ ko = "모델과 따로 켜는 기능입니다. 고른 기능들 위에 함께 돕니다."
                      en = "Extras that run alongside the capabilities you picked." }
    t_feat_reid   = @{ ko = "아까 본 그 물체인지 가린다 (지정 클래스 탐지와 함께)"
                      en = "Tells whether it is the same object as before" }
    t_feat_track  = @{ ko = "장마다 바뀌는 탐지를 번호로 이어 안정시킨다 (실측: 라벨 바뀜 31%에서 2%로)"
                      en = "Keeps ids across frames so detections stop flickering (measured: 31% to 2%)" }
    m_feat_missing = @{ ko = "필요한 가중치가 없어 쓸 수 없습니다."; en = "Not usable: the weights are missing." }
    h_prompt_now  = @{ ko = "지금 쓰는 글"; en = "Text in use" }
    h_refs_now    = @{ ko = "기준 사진"; en = "Reference photos" }
    v_none_refs   = @{ ko = "없음"; en = "none" }
    n_no_refs     = @{ ko = "넣은 사진이 없습니다."; en = "No photos yet." }
    s_refs        = @{ ko = "저장하는 중"; en = "Saving" }
    m_prompt_ok   = @{ ko = "쓸 수 있는 글입니다."; en = "The text is usable." }
    m_prompt_empty = @{ ko = "글을 한 줄 이상 적으세요."; en = "Type at least one line." }
    m_prompt_fail = @{ ko = "글을 검사하지 못했습니다 (WSL, venv 확인)."; en = "Could not check the text." }
    m_refs_fail   = @{ ko = "사진을 넣지 못했습니다 (WSL, venv 확인)."; en = "Could not add the photos." }
    m_refs_ok     = @{ ko = "사진 {0}장 넣음 ({1})"; en = "{0} photos added ({1})" }
    n_src_live    = @{ ko = "영상 들어오는 중: {0}"; en = "Streaming now: {0}" }
    n_src_down    = @{ ko = "안 닿음(고를 수 없음): {0}"; en = "Unreachable (cannot pick): {0}" }
    m_probe_fail  = @{ ko = "소스 상태를 읽지 못했습니다 (WSL, venv 확인)."
                      en = "Could not read source status (check WSL, venv)." }
    v_opened      = @{ ko = "열림"; en = "opened" }
    v_not_opened  = @{ ko = "못 엶"; en = "failed" }
    m_test_ok     = @{ ko = "모두 열립니다. 다음 실행부터 이 모델을 씁니다."
                      en = "All files open. The next run uses this model." }
    m_test_bad    = @{ ko = "못 여는 파일이 있습니다. 위 까닭을 보고 다른 파일로 다시 넣으세요."
                      en = "Some files do not open. Check the reason above and try another file." }
    m_test_fail   = @{ ko = "적재 시험을 돌리지 못했습니다 (WSL, venv 확인)."
                      en = "Could not run the load test (check WSL, venv)." }
    n_models      = @{ ko = "성능은 중간 해상도 기준 장당 시간, 자원은 GPU 어림입니다."
                      en = "Speed is per-frame at medium resolution; resource is the GPU estimate." }
    b_get         = @{ ko = "모델 받기"; en = "Download" }
    t_get         = @{ ko = "그 모델을 인터넷에서 받는다 (창 하나가 열려 진행을 보여 준다)"
                      en = "Download that model (a window shows the progress)" }
    b_close       = @{ ko = "닫기"; en = "Close" }
    h_file        = @{ ko = "파일"; en = "File" }
    v_have        = @{ ko = "있음"; en = "ready" }
    t_weights     = @{ ko = "기능마다 지금 쓰는 모델과 역할을 보고, 없는 모델을 받거나 파일로 더한다"
                      en = "See the model and role each capability uses, then download or supply a different one" }
    s_weights     = @{ ko = "가중치 넣는 중"; en = "Placing weights" }
    m_weights_fail = @{ ko = "가중치를 넣지 못했습니다 (WSL, venv 확인)."; en = "Could not place the weights (check WSL / venv)." }
    m_weights_none = @{ ko = "넣을 것이 없습니다."; en = "Nothing to place." }
    m_get_started = @{ ko = "{0} 를 받는 중입니다. 창에서 진행을 보고, 끝나면 「모델 추가」를 다시 열어 확인하세요."
                      en = "Downloading {0}. Watch the window, then reopen Add models to check." }
    w_gb          = @{ ko = "GB"; en = "GB" }
    w_mem         = @{ ko = "메모리"; en = "Memory" }
    h_using       = @{ ko = "올린 작업"; en = "Placed" }
    h_headroom    = @{ ko = "여유"; en = "Headroom" }
    h_reserve     = @{ ko = "비울 코어"; en = "Reserved" }
    t_headroom    = @{ ko = "그 기기에서 비워 둘 비율. 0 이면 한도 없이 꽉 채운다"
                      en = "Share of the device to keep free. 0 fills it up with no limit" }
    t_reserve     = @{ ko = "그 기기가 제 일(제어 등)에 쓰도록 비워 둘 코어 수"
                      en = "Cores to leave for the device's own work (control, sensors)" }
    lg_used       = @{ ko = "쓰는 중"; en = "in use" }
    lg_none       = @{ ko = "올린 작업 없음"; en = "nothing placed" }
    lg_over       = @{ ko = "넘침"; en = "over" }
    lg_auto       = @{ ko = "자동 예정"; en = "planned" }
}
# 작업(기능) 이름 — vision/model_registry.yaml 의 functions 키 → 사람이 읽는 이름과 「무엇을 위한 것인가」.
# 모르는 키는 키 그대로 보여 준다(계약에 기능이 늘어도 화면은 깨지지 않는다).
$FnName = @{
    object_detect      = @{ ko = "물체 탐지";            en = "Object detection";        why_ko = "영상에서 물체를 찾는다";                 why_en = "find objects in the video" }
    targeted_detect    = @{ ko = "지정 물체 탐지";       en = "Target-class detection";  why_ko = "정한 대상을 근거로 찾아 판정한다";      why_en = "find the chosen targets and judge them by cues" }
    river_detect       = @{ ko = "하천 부유물 탐지";     en = "River debris detection";  why_ko = "물에 떠내려가는 물체를 학습 없이 찾는다"; why_en = "find floating debris without training" }
    zeroshot_detect    = @{ ko = "처음 보는 물체 탐지";  en = "Zero-shot detection";      why_ko = "배우지 않은 물체도 말로 찾는다";        why_en = "find unseen objects from a prompt" }
    segmentation       = @{ ko = "영역 분할";            en = "Segmentation";             why_ko = "물체의 윤곽을 딴다";                    why_en = "cut out the object outline" }
    single_cam_track   = @{ ko = "추적 (한 카메라)";     en = "Tracking (one camera)";    why_ko = "같은 물체에 번호를 달아 따라간다";      why_en = "keep an id on the same object" }
    multi_cam_track    = @{ ko = "추적 (여러 카메라)";   en = "Tracking (multi-camera)";  why_ko = "카메라를 넘나드는 같은 물체를 잇는다";  why_en = "link the same object across cameras" }
    reid               = @{ ko = "재식별";               en = "Re-identification";        why_ko = "아까 본 그 물체인지 가린다";            why_en = "tell if it is the same object as before" }
    motion_forecast    = @{ ko = "이동 예측";            en = "Motion forecast";          why_ko = "다음에 어디로 갈지 본다";               why_en = "predict where it moves next" }
    anomaly_detect     = @{ ko = "이상 탐지";            en = "Anomaly detection";        why_ko = "평소와 다른 장면을 찾는다";             why_en = "spot what is unusual" }
    depth_estimate     = @{ ko = "거리 추정";            en = "Depth estimation";         why_ko = "얼마나 먼지 잰다";                      why_en = "measure how far things are" }
    video_preprocess   = @{ ko = "영상 전처리";          en = "Video preprocessing";      why_ko = "원본을 줄이고 압축한다";                why_en = "shrink and compress the frames" }
    cam360_slice       = @{ ko = "360 슬라이스";         en = "360 slice";                why_ko = "파노라마를 보통 화면으로 자른다";       why_en = "cut the panorama into a normal view" }
    slam               = @{ ko = "SLAM (지도와 위치)";   en = "SLAM (map and pose)";        why_ko = "지도를 만들며 제 위치를 안다";          why_en = "build a map and know where it is" }
    local_avoidance    = @{ ko = "로컬 회피";            en = "Local avoidance";          why_ko = "코앞의 장애물을 피한다";                why_en = "dodge what is right in front" }
    global_path_plan   = @{ ko = "전역 경로 계획";       en = "Global path planning";     why_ko = "목적지까지 긴 경로를 짠다";             why_en = "plan the long route" }
    motor_control      = @{ ko = "모터 제어";            en = "Motor control";            why_ko = "바퀴와 관절을 돌린다";                  why_en = "drive the wheels and joints" }
    emergency_stop     = @{ ko = "긴급정지";             en = "Emergency stop";           why_ko = "부딪히기 전에 멈춘다";                  why_en = "stop before it hits" }
    sensor_collect     = @{ ko = "센서 수집";            en = "Sensor collection";        why_ko = "센서 값을 읽어 보낸다";                 why_en = "read and send sensor values" }
    resource_status    = @{ ko = "자원 상태";            en = "Resource status";          why_ko = "기기가 얼마나 바쁜지 알린다";           why_en = "report how busy each device is" }
    multi_robot_assign = @{ ko = "로봇 임무 배정";       en = "Robot task assignment";    why_ko = "어느 로봇이 무엇을 할지 나눈다";        why_en = "split the work between robots" }
    digital_twin       = @{ ko = "디지털 트윈";          en = "Digital twin";             why_ko = "현장을 화면 속 모형으로 맞춘다";        why_en = "mirror the site in a model" }
    grounding_explain  = @{ ko = "근거 설명";            en = "Grounding explanation";    why_ko = "왜 그렇게 판정했는지 글로 적는다";      why_en = "write down why it decided so" }
    vlm                = @{ ko = "상황 분석 (VLM)";      en = "Scene analysis (VLM)";     why_ko = "장면 전체를 말로 풀어 본다";            why_en = "describe the whole scene" }
    camera_calibration = @{ ko = "카메라 보정";          en = "Camera calibration";       why_ko = "렌즈와 자세 값을 맞춘다";               why_en = "fix the lens and pose values" }
    model_train        = @{ ko = "모델 학습";            en = "Model training";           why_ko = "새 가중치를 만든다";                    why_en = "train new weights" }
    model_update       = @{ ko = "모델 배포";            en = "Model update";             why_ko = "만든 가중치를 기기에 내려보낸다";       why_en = "ship the weights to the devices" }
}
# 패턴 · 단계 → 그 단계가 맡는 몫 (「무엇을 위한 것」 다음에 「이 조각은 무슨 일을 하나」)
$StageRole = @{
    "cascade.device"     = @{ ko = "기기 1차 거르기"; en = "device first pass" }
    "cascade.server"     = @{ ko = "서버 정밀 확인"; en = "server verify" }
    "split.device"       = @{ ko = "기기 값 뽑기";   en = "device extract" }
    "split.server"       = @{ ko = "서버 합치기";    en = "server combine" }
    "device_only.device" = @{ ko = "기기 전용";      en = "device only" }
    "device_only.server" = @{ ko = "서버 보조";      en = "server helper" }
    "server_only.server" = @{ ko = "서버 전용";      en = "server only" }
    "server_only.device" = @{ ko = "기기 보조";      en = "device helper" }
    "resident.device"    = @{ ko = "기기 상주";      en = "device resident" }
}
#: 확장 기능 이름 — 카탈로그가 주는 이름(한국어) 대신 화면 글로 쓴다. 모르는 키는 카탈로그 이름 그대로
$FeatName = @{
    reid  = @{ ko = "재식별"; en = "Re-id" }
    track = @{ ko = "추적";   en = "Tracking" }
}
function Feat-Name($f) {
    $e = $FeatName[[string]$f.key]
    if ($null -eq $e) { return [string]$f.shown }
    $v = $e[$script:Lang]
    if ($v) { return $v }
    return $e["ko"]
}
#: 근거마다 하는 일 — 「형태 비교」처럼 무엇을 견주는지. 레지스트리의 role(score · gate · veto)은 **쓰는 방법**이고
#: 이것은 **하는 일**이다. 모르는 근거는 역할 칸을 비워 둔다(근거 이름은 카탈로그가 준다)
$CueRole = @{
    class         = @{ ko = "이름으로 찾기";        en = "find by name" }
    free_arm      = @{ ko = "후보 넓히기";          en = "widen the candidates" }
    semantic      = @{ ko = "글 설명과 대조";       en = "match the text" }
    dict_features = @{ ko = "특징 사전과 대조";     en = "match the feature list" }
    shape         = @{ ko = "형태 비교";            en = "compare the shape" }
    colour        = @{ ko = "색 비교";              en = "compare the colour" }
    ovd_pair_iou  = @{ ko = "다른 탐지기와 교차 확인"; en = "cross-check with another detector" }
    clip_colour   = @{ ko = "색 대조 (CLIP)";       en = "colour via CLIP" }
    clip_shape    = @{ ko = "모양 대조 (CLIP)";     en = "shape via CLIP" }
    saturation    = @{ ko = "흐린 것 거르기";       en = "drop washed-out ones" }
    parts         = @{ ko = "부분 찾기";            en = "find the parts" }
    hue           = @{ ko = "색상 각도로 거르기";   en = "filter by hue" }
    aspect        = @{ ko = "가로세로 비율";        en = "aspect ratio" }
    ground        = @{ ko = "바닥에 닿았나";        en = "is it on the ground" }
    height        = @{ ko = "높이 재기";            en = "measure the height" }
    calibration   = @{ ko = "거리가 잴 범위 안인가"; en = "is the range measurable" }
    size_veto     = @{ ko = "크기가 안 맞으면 버리기"; en = "reject by size" }
    lookalike     = @{ ko = "닮은꼴 감점";          en = "penalise look-alikes" }
    ref_match     = @{ ko = "기준 사진과 비교";     en = "compare with the photos" }
}
function Cue-Role($key) {
    $e = $CueRole[[string]$key]
    if ($null -eq $e) { return "" }
    $v = $e[$script:Lang]
    if ($v) { return $v }
    return $e["ko"]
}
function Fn-Text($key, $field) {
    $e = $FnName[$key]
    if ($null -eq $e) { return "" }
    $v = $e[($field + "_" + $script:Lang)]
    if ($v) { return $v }
    return $e[($field + "_ko")]
}
function Unit-Name($u) {
    $e = $FnName[$u.function]
    $n = $u.unit
    if ($null -ne $e) {
        $n = $e[$script:Lang]
        if (-not $n) { $n = $e["ko"] }
    }
    if ($u.source) { return "{0} ({1})" -f $n, $u.source }   # 어느 소스의 일인지 — 소스마다 따로 돈다
    return $n
}
function Unit-Role($u) {
    $e = $StageRole[("{0}.{1}" -f $u.pattern, $u.stage)]
    if ($null -eq $e) { return $u.stage }
    $v = $e[$script:Lang]
    if ($v) { return $v }
    return $e["ko"]
}
function Unit-Auto($u) {
    # 자동일 때 어디로 갈지 — places 가 같은 계산을 한 번 돌려 알려 준다
    if ($u.auto_node) { return "{0} {1}" -f (L "v_auto"), (Short-Of $u.auto_node) }
    if ($u.auto_code) { return "{0} ({1})" -f (L "v_auto"), $u.auto_code }
    return (L "v_auto")
}
function Short-Of($name) {
    if ($null -eq $script:Places) { return $name }
    foreach ($n in @($script:Places.nodes)) { if ($n.name -eq $name) { return (Node-Short $n) } }
    return $name
}
function Node-Auto($name, $kind) {
    # 자동으로 그 노드에 갈 예정인 작업들의 합 (사람이 고른 것은 Node-Load 가 센다)
    $sum = 0
    if ($null -eq $script:Places -or $kind -eq "gpu") { return 0 }
    foreach ($u in @($script:Places.units)) {
        if ($script:ManualPlace[$u.unit]) { continue }
        if ($u.auto_node -ne $name) { continue }
        $sum += [double]$(if ($kind -eq "cpu") { $u.cpu_m } else { $u.mem_mb })
    }
    return $sum
}
function Node-Short($n) {
    # 좁은 자리용 — 그 기기가 맡은 이름(data_stream · ep-001 · realsense). 없으면 노드 이름 그대로
    $src = [string]$n.source
    if ($src) { return $src }
    return [string]$n.name
}
function Node-Label($n) {
    # 「ep-001 온디바이스 (pi1)」 처럼 — 사람은 노드 이름(pi1)보다 그 기기가 맡은 것(ep-001)으로 알아본다
    $role = switch ([string]$n.role) {
        "gpu"    { L "w_server" }
        "robot"  { L "w_ondevice" }
        "sensor" { L "w_sensor" }
        "small"  { L "w_small" }
        default  { [string]$n.role }
    }
    $src = [string]$n.source
    if ($src -and $src -ne $n.name) { return "{0} {1} ({2})" -f $src, $role, $n.name }
    return "{0} ({1})" -f $role, $n.name
}
function L($k) {
    $e = $Str[$k]
    if ($null -eq $e) { return $k }
    $v = $e[$script:Lang]
    if ($v) { return $v }
    return $e["ko"]
}
# 글이 붙은 컨트롤을 적어 두었다가 언어를 바꿀 때 한 번에 다시 쓴다 (다시 그리는 패널은 L 을 그때 부른다)
$script:Texts = New-Object System.Collections.ArrayList
function Reg($ctl, $key) { [void]$script:Texts.Add(@{ ctl = $ctl; key = $key }); $ctl.Text = (L $key); return $ctl }

# ---- 마지막 선택
$s = [pscustomobject]@{ sources = $Sources; robotCam = 1; vision = $false; models = @("yoloe"); orchestrate = $true;
                        slice = $false; sliceU = ""; targets = $null; direct = @(); distance = $cat.distance_default;
                        cueMode = $null; cues = $null; overlay = $null; features = $null;
                        assign = $null; manual = $false; manualPlace = $null; lang = "ko"; nodeLimit = $null;
                        fnModel = $null; fnSrc = $null }
if (Test-Path $SettingsFile) {
    try {
        $saved = Get-Content $SettingsFile -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($k in "sources", "robotCam", "vision", "orchestrate", "models", "slice", "sliceU", "targets", "direct", "distance",
                       "cueMode", "cues", "overlay", "features", "assign", "manual", "manualPlace", "lang", "nodeLimit", "fnModel", "fnSrc") {
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
$script:Lang     = $(if ($s.lang -eq "en") { "en" } else { "ko" })   # 화면 글 (한국어 · 영어)
$script:Assign   = @{}                                   # 기기 → 그 기기에 올릴 모델 (2026-11-26, 모델 매칭 탭)
foreach ($kv in (To-Hash $s.assign).GetEnumerator()) { $script:Assign[$kv.Key] = @($kv.Value) }
$script:ManualPlace = To-Hash $s.manualPlace             # 작업 단위 → 노드 | "none" (수동 배치 탭)
# 고른 기능 — @{기능 → 모델} · @{기능 → @(소스)}. 목록과 선택지는 계약(cli places)에서 읽는다
$script:FnModel  = To-Hash $s.fnModel
$script:FnSrc    = @{}
foreach ($kv in (To-Hash $s.fnSrc).GetEnumerator()) { $script:FnSrc[$kv.Key] = @($kv.Value) }
$script:FnList   = @()
$script:FnOpts   = $null
$script:NodeLimit = To-Hash $s.nodeLimit        # 노드 → @{headroom, cores} (런처에서 조절하는 한도)
$script:PlacesKey = $null                                # 노드 · 작업을 어떤 기능 묶음으로 읽었나
$script:PreviewCompare = $false                          # 미리보기에서 적용 전과 견줄까
$script:PreviewPath = ""                                 # 미리보기에 쓸 그림 (마지막에 쓴 것)
$script:Places   = $null                                 # cli places --json 결과 (노드와 작업)
$script:TargetsTouched = ($null -ne $s.targets)
$script:Running  = $false
$script:Applied  = $null                                 # 반영된 실행 구성(비교용)
$script:Building = $false

# ---- 근거 · 자원 셈 (화면보다 먼저 — Option-Tip 이 창을 만들 때 부른다)
function Cue-Models($row) {
    # 그 근거가 쓰는 모델들 — 파일이 없으면 「(없음)」을 붙인다 (카탈로그가 available 로 알려 준다)
    return @($row.models | ForEach-Object { if ($_.available) { $_.key } else { "{0} ({1})" -f $_.key, (L "v_missing") } })
}
function Cue-MB($row) {
    # 그 근거의 예상 GPU — 지정 클래스 단서는 모델마다 어림이 있다(gpu_mb_targeted). 없으면 $null
    $sum = 0
    $known = $false
    foreach ($x in @($row.models)) {
        $v = $cat.gpu_mb_targeted.($x.key)
        if ($null -ne $v) { $sum += [int]$v; $known = $true }
    }
    if ($known) { return $sum }
    return $null
}
function Chosen-MB {
    # 고른 기능의 GPU 어림 합 (프로세스 기본값 제외) — Start-Vision 의 셈과 같다
    $mb = 0
    foreach ($m in (Chosen-Models)) {
        if ($m -eq "targeted") { $mb += (Targeted-MB) } else { $mb += [int]$cat.gpu_mb_option.$m }
    }
    return $mb
}

# ---- 창
$font = New-Object System.Drawing.Font("Malgun Gothic", 10)
$form = New-Object System.Windows.Forms.Form
$form.Text = (L "d_title")
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
# 단계 화면 — 고르는 순서 그대로다: 소스 → 기능(근거 · 모델 · 자원) → 지정 클래스 → 오버레이 → 배치.
# 탭을 눌러 바로 갈 수도 있고 아래 「이전 · 다음」으로 한 단계씩 갈 수도 있다 (2026-11-26)
$tSrc    = New-Tab (L "step_src")
$tFunc   = New-Tab (L "step_func")
$tTarget = New-Tab (L "step_target")
$tOver   = New-Tab (L "step_over")
$tPlace  = New-Tab (L "step_place")
$StepKeys = @("step_src", "step_func", "step_target", "step_over", "step_place")

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

# ================================================================ 1단계 — 소스
$gStream = New-Group $tSrc "" 8 8 596 170
[void](Reg $gStream "g_stream")
[void](Reg (New-Label $gStream "" 14 30) "l_sources")
# 소스를 줄줄이 놓는다(흐름 배치) — 이름 길이가 제각각이고, 캠을 고를 수 있는 소스 옆에는 상자가 하나 더 붙는다
$srcChecks = @{}
$script:SrcAfter = @{}                                   # 소스 → 그 옆에 붙일 컨트롤의 자리
$sx = 80
$sy = 28
foreach ($n in $Sources) {
    $c = New-Check $gStream $n $sx $sy ($s.sources -contains $n)
    $srcChecks[$n] = $c
    $sx += $c.Width + 10
    if ($n -eq "robot1") {
        $script:SrcAfter[$n] = @{ x = $sx; y = $sy }     # 「cam [번호]」 가 여기에 들어간다
        $sx += 108                                       # 다음 소스(ep-001)와 떼어 둔다 — robot1 의 설정으로 보이게
    }
    if ($n -eq "cam360") {
        $script:SrcAfter[$n] = @{ x = $sx; y = $sy }     # 「세부 설정」은 cam360 만의 것이라 그 옆에
        $sx += 112                                       # 다음 소스와 떼어 둔다
    }
    if ($sx -gt 470) { $sx = 80; $sy += 26 }
}
# 로봇 캠은 **여러 대를 고를 수 있는 소스 옆에만** 둔다 (robot1 — 나머지는 카메라가 하나다)
$camAt = $script:SrcAfter["robot1"]
[void](New-Label $gStream (L "l_cam") $camAt.x ($camAt.y + 2))
$cam = New-Object System.Windows.Forms.ComboBox
$cam.DropDownStyle = "DropDownList"
$cam.Location = New-Object System.Drawing.Point(($camAt.x + 34), ($camAt.y - 3))
$cam.Width = 46
1..5 | ForEach-Object { [void]$cam.Items.Add($_) }
$cam.SelectedItem = [int]$s.robotCam
if ($cam.SelectedIndex -lt 0) { $cam.SelectedIndex = 0 }
$gStream.Controls.Add($cam)
$srcState = New-Label $gStream "" 14 ($sy + 30)
$srcState.AutoSize = $false
$srcState.Size = New-Object System.Drawing.Size(450, 34)
$srcCheck = New-Object System.Windows.Forms.Button
$srcCheck.Location = New-Object System.Drawing.Point(472, ($sy + 26))
$srcCheck.Width = 114
[void](Reg $srcCheck "b_recheck")
$gStream.Controls.Add($srcCheck)
# 사람이 기기를 하나 더 붙였을 때 — 코드를 고치지 않고 여기서 넣는다 (config/sources.json)
$srcAdd = New-Object System.Windows.Forms.Button
$srcAdd.Location = New-Object System.Drawing.Point(472, ($sy + 52))
$srcAdd.Width = 114
[void](Reg $srcAdd "b_src_add")
$gStream.Controls.Add($srcAdd)
$script:SrcX = $sx                                       # 더한 소스를 이어 놓을 자리
$script:SrcY = $sy

$sliceAt = $script:SrcAfter["cam360"]
$srcTip = New-Object System.Windows.Forms.ToolTip
$srcTip.AutoPopDelay = 20000
$sliceTip = New-Object System.Windows.Forms.ToolTip
$sliceTip.AutoPopDelay = 30000
# 슬라이스 값(켜기 · u)은 **cam360 옆 「세부 설정」 창**에서 고친다. 아래 두 컨트롤은 그 값을 담는 자리로만 둔다
# (Current-Config · Build-Profile 같은 기존 길이 그대로 이것들을 읽는다 — 화면에는 보이지 않는다)
$sliceOn = New-Check $gStream "" 0 0 $s.slice
$sliceOn.Visible = $false
$sliceU = New-Object System.Windows.Forms.TextBox
$sliceU.Location = New-Object System.Drawing.Point(0, 0)
$sliceU.Width = 44
$sliceU.Text = [string]$s.sliceU
$sliceU.Visible = $false
$gStream.Controls.Add($sliceU)
$sliceBtn = New-Object System.Windows.Forms.Button
$sliceBtn.Location = New-Object System.Drawing.Point($sliceAt.x, ($sliceAt.y - 3))
$sliceBtn.Width = 100
[void](Reg $sliceBtn "b_detail")
$sliceTip.SetToolTip($sliceBtn, (L "t_slice"))
$gStream.Controls.Add($sliceBtn)

function Pick-Image {
    # 미리보기에 쓸 그림 — 경로를 직접 적거나 파일을 고른다. 기본값은 데이터 서버가 저장한 마지막 실행의 장들
    $d = New-Object System.Windows.Forms.Form
    $d.Text = (L "b_preview")
    $d.Font = $font
    $d.StartPosition = "CenterParent"
    $d.FormBorderStyle = "FixedDialog"
    $d.MaximizeBox = $false
    $d.MinimizeBox = $false
    $d.ClientSize = New-Object System.Drawing.Size(640, 176)
    [void](New-Label $d (L "l_image") 12 16)
    $tb = New-Object System.Windows.Forms.TextBox
    $tb.Location = New-Object System.Drawing.Point(12, 42)
    $tb.Width = 500
    $tb.Text = $script:PreviewPath
    $d.Controls.Add($tb)
    $br = New-Object System.Windows.Forms.Button
    $br.Text = (L "b_browse")
    $br.Location = New-Object System.Drawing.Point(520, 40)
    $br.Width = 100
    $br.Add_Click({
        $fd = New-Object System.Windows.Forms.OpenFileDialog
        $fd.Filter = "images|*.jpg;*.jpeg;*.png|all|*.*"
        if ($fd.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $tb.Text = $fd.FileName }
    })
    $d.Controls.Add($br)
    $cmp = New-Check $d (L "l_compare") 12 72 $script:PreviewCompare
    $manualTip.SetToolTip($cmp, (L "t_compare"))
    $hint = New-Label $d (L "n_image") 12 98
    $hint.AutoSize = $false
    $hint.Size = New-Object System.Drawing.Size(610, 30)
    $ok = New-Object System.Windows.Forms.Button
    $ok.Text = (L "b_run_preview")
    $ok.Location = New-Object System.Drawing.Point(420, 138)
    $ok.Width = 100
    $cancel = New-Object System.Windows.Forms.Button
    $cancel.Text = (L "b_close")
    $cancel.Location = New-Object System.Drawing.Point(528, 138)
    $cancel.Width = 92
    $go = @{ v = $false }
    $ok.Add_Click({ $go.v = $true; $d.Close() })
    $cancel.Add_Click({ $d.Close() })
    $d.Controls.Add($ok)
    $d.Controls.Add($cancel)
    $d.AcceptButton = $ok
    [void]$d.ShowDialog($form)
    if (-not $go.v) { return $null }
    $script:PreviewPath = $tb.Text.Trim()
    $script:PreviewCompare = $cmp.Checked
    return $script:PreviewPath
}
function Wsl-Win($repoRelative) {
    # 저장소 안의 자리를 Windows 가 읽는 경로로 (\\wsl.localhost\<배포판>\<저장소>\...)
    return "\\wsl.localhost\" + $Distro + (($RepoDir + $repoRelative) -replace "/", "\")
}
function To-WslPath($p) {
    if ($p -match "^[A-Za-z]:") { return "/mnt/" + $p.Substring(0, 1).ToLower() + ($p.Substring(2) -replace "\\", "/") }
    # \\wsl.localhost\<배포판>\home\... 는 이미 WSL 안의 자리다 — 그 뒤만 떼어 쓴다 (설치하지 않고
    # 저장소 폴더에서 바로 띄운 경우. 전에는 그대로 넘겨 /mnt/\\wsl.localhost/... 같은 자리가 나왔다)
    $m = [regex]::Match($p, '^\\\\wsl(?:\.localhost|\$)\\[^\\]+(\\.*)$')
    if ($m.Success) { return ($m.Groups[1].Value -replace "\\", "/") }
    return $p
}
function Show-Preview {
    # 저장된 그림에 **지금 고른 설정**을 그대로 돌려 결과를 보여 준다 (모델 · 파라미터 · 프롬프트 · 오버레이를
    # 바꿨을 때 실제로 무엇이 달라지는지). tools/preview_apply.py 가 비전을 한 번만 돌린다
    $img = Pick-Image
    if (-not $img) { return }
    if ($img -eq "") {
        [void][System.Windows.Forms.MessageBox]::Show((L "m_need_image"), (L "d_title"))
        return
    }
    # 도구가 있는지는 **Windows 쪽 파일 검사**로 본다 — wsl.exe 로 쉘 한 줄을 넘기면 따옴표가 깨져
    # 늘 빈 답이 와서, 도구가 있어도 막아 버렸다 (2026-11-26 실측). 그림이 없는 경우는 도구가 까닭을 돌려준다
    if (-not (Test-Path (Wsl-Win "/tools/preview_apply.py"))) {
        [void][System.Windows.Forms.MessageBox]::Show((L "m_no_preview_tool"), (L "d_title"))
        return
    }
    Write-PreviewProfile
    Set-Status (L "s_preview") $true
    $out = "/tmp/ds_preview"
    $argv = @($PyWsl, "tools/preview_apply.py", "--json", "--image", (To-WslPath $img),
              "--profile", $PreviewWsl)
    # 「비교」를 켜면 **지금 돌고 있는 프로필**(profile.json)을 B 로 두어 바뀐 것이 눈에 보이게 한다
    if ($script:PreviewCompare -and (Test-Path $ProfileFile)) { $argv += @("--profile-b", $ProfileWsl) }
    $argv += @("--out", $out, "--models") + @(Chosen-Models)
    $txt = Invoke-Wsl $argv
    Set-Status $(if ($script:Running) { L "s_running" } else { "" })
    $r = $null
    $r = From-Json $txt
    if ($null -eq $r -or -not $r.ok) {
        $why = if ($null -eq $r) { $txt } else { [string]$r.why }
        [void][System.Windows.Forms.MessageBox]::Show((L "m_preview_fail") + "`n" + $why, (L "d_title"))
        return
    }
    Set-OverShot ([string]$r.overlay) ([string]$r.summary)
    Show-Image ([string]$r.overlay) ([string]$r.summary)
}
function Show-Image($wslPath, $summary) {
    # 결과 그림을 창으로 — WSL 경로를 Windows 에서 읽는다 (\\wsl.localhost\<배포판>\...)
    $win = "\\wsl.localhost\" + $Distro + ($wslPath -replace "/", "\")
    $d = New-Object System.Windows.Forms.Form
    $d.Text = (L "b_preview")
    $d.Font = $font
    $d.StartPosition = "CenterParent"
    $d.ClientSize = New-Object System.Drawing.Size(900, 620)
    $pic = New-Object System.Windows.Forms.PictureBox
    $pic.Location = New-Object System.Drawing.Point(8, 8)
    $pic.Size = New-Object System.Drawing.Size(884, 540)
    $pic.SizeMode = "Zoom"
    try { $pic.Image = [System.Drawing.Image]::FromFile($win) } catch { }
    $d.Controls.Add($pic)
    $lab = New-Label $d ([string]$summary) 8 556
    $lab.AutoSize = $false
    $lab.Size = New-Object System.Drawing.Size(884, 36)
    $close = New-Object System.Windows.Forms.Button
    $close.Text = (L "b_close")
    $close.Location = New-Object System.Drawing.Point(796, 586)
    $close.Width = 96
    $close.Add_Click({ $d.Close() })
    $d.Controls.Add($close)
    [void]$d.ShowDialog($form)
    if ($null -ne $pic.Image) { $pic.Image.Dispose() }
}
function Target-Info($name) {
    # 카탈로그가 대상마다 돌려주는 글(prompt)과 기준 사진(refs)
    foreach ($x in @($cat.targets)) { if ($x.name -eq $name) { return $x } }
    return $null
}
function Show-Cue-Detail($name) {
    # 그 대상이 지금 쓰는 **글 단서와 기준 사진**을 보여 주고 그 자리에서 바꾼다.
    # 글은 cue_catalog --check-prompt 로 바로 검사하고, 저장은 --save-target · 사진은 --attach-refs 가 한다
    $info = Target-Info $name
    $d = New-Object System.Windows.Forms.Form
    $d.Text = ((L "b_cue_detail") + " — " + $name)
    $d.Font = $font
    $d.StartPosition = "CenterParent"
    $d.FormBorderStyle = "FixedDialog"
    $d.MaximizeBox = $false
    $d.MinimizeBox = $false
    $d.ClientSize = New-Object System.Drawing.Size(700, 470)
    [void](New-Label $d (L "h_prompt_now") 12 12)
    $tb = New-Object System.Windows.Forms.TextBox
    $tb.Location = New-Object System.Drawing.Point(12, 38)
    $tb.Size = New-Object System.Drawing.Size(560, 60)
    $tb.Multiline = $true
    $tb.ScrollBars = "Vertical"
    $tb.Text = (@($info.prompt) -join "`r`n")
    $d.Controls.Add($tb)
    $chk = New-Object System.Windows.Forms.Button
    $chk.Text = (L "b_check_prompt")
    $chk.Location = New-Object System.Drawing.Point(580, 38)
    $chk.Width = 106
    $msg = New-Label $d "" 12 104
    $msg.AutoSize = $false
    $msg.Size = New-Object System.Drawing.Size(674, 34)
    $chk.Add_Click({
        $lines = @($tb.Text -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
        if ($lines.Count -eq 0) { $msg.Text = (L "m_prompt_empty"); return }
        $argv = @($PyWsl, "vision/cue_catalog.py", "--check-prompt")
        foreach ($l in $lines) { $argv += @("--prompt", $l) }
        $r = From-Json (Invoke-Wsl $argv)
        if ($null -eq $r) { $msg.Text = (L "m_prompt_fail") }
        elseif ($r.ok) { $msg.Text = (L "m_prompt_ok") }
        else { $msg.Text = [string]$r.error }
    })
    $d.Controls.Add($chk)
    # 적는 법 (카탈로그가 주는 그대로)
    $g = $cat.prompt_guide
    $guide = New-Label $d (@($g.lines) -join "`r`n") 12 142
    $guide.AutoSize = $false
    $guide.Size = New-Object System.Drawing.Size(674, 92)
    # 기준 사진
    [void](New-Label $d (L "h_refs_now") 12 240)
    $refTxt = (L "v_none_refs")
    if ($null -ne $info.refs) {
        $refTxt = ("{0} {1}장" -f [string]$info.refs.state, [int]$info.refs.count)
        if ($info.refs.why) { $refTxt += (" ({0})" -f [string]$info.refs.why) }
    }
    [void](New-Label $d $refTxt 120 240)
    $pics = New-Object System.Windows.Forms.Panel
    $pics.Location = New-Object System.Drawing.Point(12, 266)
    $pics.Size = New-Object System.Drawing.Size(674, 120)
    $pics.AutoScroll = $true
    $d.Controls.Add($pics)
    $fill = {
        $pics.Controls.Clear()
        $i2 = 0
        $inf = Target-Info $name
        if ($null -ne $inf.refs -and $inf.refs.dir) {
            foreach ($f in @($inf.refs.files)) {
                $win = Wsl-Win ("/vision/assets/targets/" + [string]$inf.refs.dir + "/" + [string]$f)
                $pb = New-Object System.Windows.Forms.PictureBox
                $pb.Location = New-Object System.Drawing.Point((4 + $i2 * 116), 4)
                $pb.Size = New-Object System.Drawing.Size(110, 104)
                $pb.SizeMode = "Zoom"
                try { $pb.Image = [System.Drawing.Image]::FromFile($win) } catch { }
                $pics.Controls.Add($pb)
                $i2++
            }
        }
        if ($i2 -eq 0) { [void](New-Label $pics (L "n_no_refs") 4 8) }
    }
    & $fill
    $addRef = New-Object System.Windows.Forms.Button
    $addRef.Text = (L "b_add_refs")
    $addRef.Location = New-Object System.Drawing.Point(12, 394)
    $addRef.Width = 150
    $addRef.Add_Click({
        $fd = New-Object System.Windows.Forms.OpenFileDialog
        $fd.Multiselect = $true
        $fd.Filter = "images or zip|*.zip;*.png;*.jpg;*.jpeg;*.bmp;*.webp|all|*.*"
        if ($fd.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { return }
        $argv = @($PyWsl, "vision/cue_catalog.py", "--attach-refs", $name)
        foreach ($f in $fd.FileNames) {
            $w = To-WslPath $f
            if ($f.ToLower().EndsWith(".zip")) { $argv += @("--ref-zip", $w) } else { $argv += @("--ref-image", $w) }
        }
        Set-Status (L "s_refs") $true
        $r = From-Json (Invoke-Wsl $argv)
        Set-Status ""
        if ($null -eq $r) { $msg.Text = (L "m_refs_fail") }
        else {
            $msg.Text = $(if ($r.ok) { ((L "m_refs_ok") -f [int]$r.images, [string]$r.state) } else { [string]$r.error })
            Reload-Catalog
            & $fill
        }
    })
    $d.Controls.Add($addRef)
    $save = New-Object System.Windows.Forms.Button
    $save.Text = (L "b_save")
    $save.Location = New-Object System.Drawing.Point(480, 394)
    $save.Width = 96
    $save.Add_Click({
        $lines = @($tb.Text -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
        $argv = @($PyWsl, "vision/cue_catalog.py", "--save-target", $name)
        foreach ($l in $lines) { $argv += @("--prompt", $l) }
        Set-Status (L "s_refs") $true
        $r = From-Json (Invoke-Wsl $argv)
        Set-Status ""
        if ($null -eq $r) { $msg.Text = (L "m_prompt_fail") }
        elseif ($r.ok) { $msg.Text = (L "m_saved"); Reload-Catalog; Build-CueBoxes }
        else { $msg.Text = [string]$r.error }
    })
    $d.Controls.Add($save)
    $close = New-Object System.Windows.Forms.Button
    $close.Text = (L "b_close")
    $close.Location = New-Object System.Drawing.Point(590, 394)
    $close.Width = 96
    $close.Add_Click({ $d.Close() })
    $d.Controls.Add($close)
    [void]$d.ShowDialog($form)
}
function Show-Features {
    # 확장 기능 — 모델과 따로 켜는 것(재식별, 추적). 쓸 수 없는 것은 끄고 까닭을 풍선말로
    $d = New-Object System.Windows.Forms.Form
    $d.Text = (L "b_add_feat")
    $d.Font = $font
    $d.StartPosition = "CenterParent"
    $d.FormBorderStyle = "FixedDialog"
    $d.MaximizeBox = $false
    $d.MinimizeBox = $false
    $d.ClientSize = New-Object System.Drawing.Size(520, 240)
    [void](New-Label $d (L "n_features") 12 12)
    $y = 46
    $tip = New-Object System.Windows.Forms.ToolTip
    foreach ($f in @($cat.features)) {
        $c = New-Check $d (Feat-Name $f) 16 $y $featChecks[$f.key].Checked
        $c.Enabled = [bool]$f.available
        $c.Tag = @{ key = $f.key }
        $c.Add_CheckedChanged({ $featChecks[$this.Tag.key].Checked = $this.Checked; Changed })
        $why = $(if ($f.available) { (L "t_feat_" + $f.key) } else { (L "m_feat_missing") })
        $tip.SetToolTip($c, $why)
        [void](New-Label $d $why 160 ($y + 3))
        $y += 30
    }
    $close = New-Object System.Windows.Forms.Button
    $close.Text = (L "b_close")
    $close.Location = New-Object System.Drawing.Point(412, 196)
    $close.Width = 96
    $close.Add_Click({ $d.Close() })
    $d.Controls.Add($close)
    [void]$d.ShowDialog($form)
}
function Show-Slice {
    # cam360 세부 설정 — 슬라이스 켜기와 u (v 는 -30 고정). 값은 숨은 컨트롤에 담아 기존 길이 그대로 읽게 한다
    $d = New-Object System.Windows.Forms.Form
    $d.Text = (L "b_detail")
    $d.Font = $font
    $d.StartPosition = "CenterParent"
    $d.FormBorderStyle = "FixedDialog"
    $d.MaximizeBox = $false
    $d.MinimizeBox = $false
    $d.ClientSize = New-Object System.Drawing.Size(520, 215)
    [void](New-Label $d "cam360" 12 12)
    $on = New-Check $d (L "l_slice_on") 12 40 $sliceOn.Checked
    [void](New-Label $d "u" 12 78)
    $ub = New-Object System.Windows.Forms.TextBox
    $ub.Location = New-Object System.Drawing.Point(32, 74)
    $ub.Width = 60
    $ub.Text = $sliceU.Text
    $d.Controls.Add($ub)
    [void](New-Label $d ("v {0}" -f $SliceV) 104 78)
    $why = New-Label $d (L "t_slice") 12 106
    $why.AutoSize = $false
    $why.Size = New-Object System.Drawing.Size(496, 66)
    $ok = New-Object System.Windows.Forms.Button
    $ok.Text = (L "b_close")
    $ok.Location = New-Object System.Drawing.Point(412, 178)
    $ok.Width = 96
    $ok.Add_Click({ $d.Close() })
    $d.Controls.Add($ok)
    $d.AcceptButton = $ok
    [void]$d.ShowDialog($form)
    $changed = ($sliceOn.Checked -ne $on.Checked) -or ($sliceU.Text -ne $ub.Text.Trim())
    $sliceOn.Checked = $on.Checked
    $sliceU.Text = $ub.Text.Trim()
    if ($changed) { Build-Match; Changed }
}

# ================================================================ 2단계 — 기능 (그 소스에 무엇을 시킬 것인가)
$gVision = New-Group $tFunc "" 8 8 596 56
[void](Reg $gVision "g_vision")
$visOn = Reg (New-Check $gVision "" 14 26 $s.vision) "l_on"
# 기능은 아래 「기능 × 소스」 격자에서 고른다. 모델 체크는 없앴다 — 모델은 기능마다 고른다 (2026-11-26)
$modelChecks = @{}
# 선택지마다 「단서 → 모델」 — vision/model_registry.yaml (카탈로그 options)
$manualTip = New-Object System.Windows.Forms.ToolTip     # 수동 배치 — 작업의 진짜 이름 · 쓰임새 · 둘 수 있는 곳
$manualTip.AutoPopDelay = 30000
$modelTip = New-Object System.Windows.Forms.ToolTip
$modelTip.AutoPopDelay = 30000
function Option-Tip($m) {
    $lines = @(L "n_cue_tip")
    foreach ($r in @($cat.options.$m)) {
        $ms = @(Cue-Models $r)
        $lines += ("{0}: {1}" -f $r.label, $(if ($ms.Count) { $ms -join ", " } else { L "v_rule" }))
    }
    $add = [int]$cat.gpu_mb_option.$m
    if ($m -eq "targeted") { $lines += (L "n_gpu_sum") } else { $lines += ((L "n_gpu_est") -f $add) }
    return ($lines -join "`n")
}

# 추가 기능 — 모델과 따로 켜는 것(카탈로그 features). 지금은 객체 재식별

$featChecks = @{}
# 확장 기능(재식별, 추적)은 기능 소스별 배치 아래 「추가」 창에서 고른다. 체크는 여기 두되 화면에서는 뺀다
# (Build-Profile · Save-Settings 가 이것들을 그대로 읽는다)
foreach ($f in @($cat.features)) {
    $on = $false
    if ($null -ne $s.features -and $null -ne $s.features.($f.key)) { $on = [bool]$s.features.($f.key) }
    $featChecks[$f.key] = New-Check $gVision (Feat-Name $f) 0 0 $on
    $featChecks[$f.key].Visible = $false
    if (-not $f.available) { $featChecks[$f.key].Checked = $false }
}

$previewBtn = New-Object System.Windows.Forms.Button
$previewBtn.Location = New-Object System.Drawing.Point(320, 18)
$previewBtn.Width = 110
[void](Reg $previewBtn "b_preview")
$gVision.Controls.Add($previewBtn)
$weightsBtn = New-Object System.Windows.Forms.Button
$weightsBtn.Location = New-Object System.Drawing.Point(440, 18)
$weightsBtn.Width = 130
[void](Reg $weightsBtn "b_weights")
$gVision.Controls.Add($weightsBtn)

# ---- 근거 · 모델 · 예상 자원 (고른 기능마다) — 카탈로그 options · gpu_mb_* 를 그대로 읽어 적는다
$gCues = New-Group $tFunc "" 8 362 596 174
[void](Reg $gCues "g_cues")
$cueTable = New-Object System.Windows.Forms.Panel
$cueTable.Location = New-Object System.Drawing.Point(10, 20)
$cueTable.Size = New-Object System.Drawing.Size(576, 144)
$cueTable.AutoScroll = $true
$gCues.Controls.Add($cueTable)

# ---- 기기별 기능 (모델 매칭) — 소스마다 올릴 기능
$gMatch = New-Group $tFunc "" 8 70 596 286
[void](Reg $gMatch "g_match")
$matchPanel = New-Object System.Windows.Forms.Panel
$matchPanel.Location = New-Object System.Drawing.Point(10, 20)
$matchPanel.Size = New-Object System.Drawing.Size(576, 226)
$matchPanel.AutoScroll = $true
$gMatch.Controls.Add($matchPanel)
$matchReset = New-Object System.Windows.Forms.Button
$addFeat = New-Object System.Windows.Forms.Button
$addFeat.Location = New-Object System.Drawing.Point(10, 254)
$addFeat.Width = 100
[void](Reg $addFeat "b_add_feat")
$gMatch.Controls.Add($addFeat)
$matchReset.Location = New-Object System.Drawing.Point(120, 254)
$matchReset.Width = 130
[void](Reg $matchReset "b_reset")
$gMatch.Controls.Add($matchReset)
$matchNote = New-Object System.Windows.Forms.Label
$matchNote.AutoSize = $false
$matchNote.Location = New-Object System.Drawing.Point(260, 258)
$matchNote.Size = New-Object System.Drawing.Size(326, 26)
$gMatch.Controls.Add($matchNote)

# ================================================================ 지정 클래스 탭
$gT = New-Group $tTarget "" 8 8 596 118
[void](Reg $gT "g_target")
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
[void](Reg $directAdd "b_add")
$directAdd.Location = New-Object System.Drawing.Point(218, 82)
$directAdd.Size = New-Object System.Drawing.Size(60, 26)
$gT.Controls.Add($directAdd)
$distOn = Reg (New-Check $gT "" 300 85 $s.distance) "c_distance"
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
        $det = New-Object System.Windows.Forms.Button
        $det.Text = (L "b_cue_detail")
        $det.Location = New-Object System.Drawing.Point(430, 18)
        $det.Width = 130
        $det.Tag = @{ name = $name }
        $det.Add_Click({ Show-Cue-Detail $this.Tag.name })
        $g.Controls.Add($det)
        $mode = $script:CueMode[$name]; if (-not $mode) { $mode = "default" }
        foreach ($rm in @(@("default", (L "r_default"), 12), @("custom", (L "r_custom"), 110))) {
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
            [void](New-Label $g (L "h_size") 12 ($ly + 3))
            $sz = New-Object System.Windows.Forms.TextBox
            $sz.Location = New-Object System.Drawing.Point(52, $ly); $sz.Width = 110; $sz.Text = $d.size; $sz.Tag = $name
            $sz.Add_Leave({ On-DirectSize $this })
            $g.Controls.Add($sz)
            [void](New-Label $g (L "h_colour") 176 ($ly + 3))
            $col = New-Object System.Windows.Forms.ComboBox
            $col.DropDownStyle = "DropDownList"; $col.Location = New-Object System.Drawing.Point(200, $ly); $col.Width = 100
            [void]$col.Items.Add("")
            foreach ($cn in $cat.colours) { [void]$col.Items.Add($cn) }
            $col.SelectedItem = $d.colour; $col.Tag = $name
            $col.Add_SelectedIndexChanged({ On-DirectColour $this })
            $g.Controls.Add($col)
            $reg = New-Object System.Windows.Forms.Button
            $reg.Text = (L "b_register"); $reg.Location = New-Object System.Drawing.Point(420, ($ly - 2)); $reg.Size = New-Object System.Drawing.Size(130, 26)
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
        [void][System.Windows.Forms.MessageBox]::Show((L "m_size_mm"), (L "d_title")); return
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
    try { $r = (Invoke-Wsl $argv) | ConvertFrom-Json } catch { $r = [pscustomobject]@{ ok = $false; error = (L "m_save_fail") } }
    if ($r.ok) { [void][System.Windows.Forms.MessageBox]::Show((L "m_saved"), (L "d_title")) }
    else { [void][System.Windows.Forms.MessageBox]::Show([string]$r.error, (L "d_title")) }
}

function Add-Direct($text) {
    $n = $text.Trim().ToLower()
    if ($n -notmatch '^[a-z][a-z0-9 _\-]{0,40}$') {
        [void][System.Windows.Forms.MessageBox]::Show((L "m_name_en"), (L "d_title")); return
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
$overPanel.Size = New-Object System.Drawing.Size(606, 286)
$overPanel.AutoScroll = $true
$tOver.Controls.Add($overPanel)
$overPreview = New-Object System.Windows.Forms.Button
$overPreview.Location = New-Object System.Drawing.Point(4, 512)
$overPreview.Width = 150
[void](Reg $overPreview "b_preview_overlay")
$tOver.Controls.Add($overPreview)
# 고른 오버레이가 실제로 어떻게 보이는지 — 저장된 장에 그려 바로 아래에 둔다 (고칠 때마다 다시 그린다)
$overPic = New-Object System.Windows.Forms.PictureBox
$overPic.Location = New-Object System.Drawing.Point(4, 296)
$overPic.Size = New-Object System.Drawing.Size(606, 188)
$overPic.SizeMode = "Zoom"
$overPic.BorderStyle = "FixedSingle"
$tOver.Controls.Add($overPic)
$overCap = New-Object System.Windows.Forms.Label
$overCap.AutoSize = $false
$overCap.Location = New-Object System.Drawing.Point(4, 488)
$overCap.Size = New-Object System.Drawing.Size(606, 20)
$tOver.Controls.Add($overCap)
# 라벨 · 깃발 글은 번역표에서 (키는 그대로 — 프로필에 들어가는 것은 키다)
function Label-Items { return @(@("name", (L "lbl_name")), @("conf", (L "lbl_conf")), @("id", (L "lbl_id")),
                                @("dist", (L "lbl_dist")), @("gps", (L "lbl_gps"))) }

# 고른 기능 — @{기능 → @(소스...)} 와 @{기능 → 모델}. 모델은 그 기능이 쓸 수 있는 것 중에서 고른다
function Fn-Sources($fn) {
    $v = $script:FnSrc[$fn]
    if ($null -eq $v) { return @() }
    return @(@(On-All) | Where-Object { $v -contains $_ })      # 지금 켠 소스만
}
function Fn-On($fn) { return (Fn-Sources $fn).Count -gt 0 }
function Fn-Usable($fn) {
    # 이 기기에서 지금 되는 기능인가 (cli preflight 가 본 것 — 주로 모델 가중치가 있나)
    $e = $script:FnWhy[$fn]
    if ($null -eq $e) { return $true }                 # 모르면 막지 않는다
    return [bool]$e.usable
}
function Fn-Why($fn) {
    $e = $script:FnWhy[$fn]
    if ($null -eq $e) { return "" }
    return [string]$e.why
}
function Fn-Options($fn) {
    if ($null -eq $script:FnOpts) { return @() }
    return @($script:FnOpts.$fn)
}
function Option-Domain($o) {
    # 도메인 전용 파이프라인인가 (레지스트리 options.<키>.domain) — 그러면 모델이 아니라 기능이다
    if ($null -eq $script:ModelInfo) { return $false }
    $i = $script:ModelInfo.$o
    return ($null -ne $i -and [bool]$i.domain)
}
function Option-ModelsText($o) {
    # 도메인 전용 기능이 실제로 쓰는 모델 — 근거마다 고른 모델의 모음. 모르면 「근거 모델」
    $i = $(if ($null -ne $script:ModelInfo) { $script:ModelInfo.$o } else { $null })
    $ms = @(if ($null -ne $i) { @($i.models) } else { @() })
    if ($ms.Count) { return ($ms -join ", ") }
    return $(if ($script:Lang -eq "en") { "cue models" } else { "근거 모델" })
}
function Fn-Model($fn) {
    $opts = @(Fn-Options $fn)
    if ($opts.Count -eq 0) { return $null }
    $m = $script:FnModel[$fn]
    if ($m -and ($opts -contains $m)) { return $m }
    return $opts[0]
}
function Chosen-Functions { return @($script:FnList | Where-Object { Fn-On $_ }) }
#: 비전 프로세스가 올릴 모델 — 고른 기능들이 쓰는 모델의 모음 (카탈로그 순서)
function Chosen-Models {
    $want = @{}
    foreach ($fn in (Chosen-Functions)) {
        $m = Fn-Model $fn
        if ($m) { $want[$m] = $true }
    }
    return @($Models | Where-Object { $want.ContainsKey($_) })
}
#: {소스: [기능...]} — 오케스트레이터에 그대로 넘긴다
function Functions-Map {
    $out = @{}
    foreach ($fn in (Chosen-Functions)) {
        foreach ($s2 in (Fn-Sources $fn)) {
            if ($null -eq $out[$s2]) { $out[$s2] = @() }
            $out[$s2] += $fn
        }
    }
    return $out
}
function Functions-Args {
    $m = Functions-Map
    if ($m.Count -eq 0) { return "" }
    $parts = @($m.Keys | Sort-Object | ForEach-Object { "{0}={1}" -f $_, (($m[$_] | Sort-Object) -join ",") })
    return " --functions " + ($parts -join " ")
}

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
function Set-OverShot($wslPath, $summary) {
    # 결과 그림을 탭 안에 건다. 파일을 잡아 두면 다음 미리보기가 같은 이름에 못 쓰니 바이트로 읽어 들인다
    $win = "\\wsl.localhost\" + $Distro + ($wslPath -replace "/", "\")
    if ($null -ne $overPic.Image) { $overPic.Image.Dispose(); $overPic.Image = $null }
    try {
        $bytes = [System.IO.File]::ReadAllBytes($win)
        $overPic.Image = [System.Drawing.Image]::FromStream((New-Object System.IO.MemoryStream(, $bytes)))
    } catch { }
    $overCap.Text = [string]$summary
}
function Run-OverShot {
    # 지금 고른 오버레이를 저장된 장 한 장에 그려 본다 (비교 없이, 한 장만 — 빨라야 하니)
    if (-not $script:PreviewPath) { $overCap.Text = (L "n_over_pick"); return }
    if (-not (Test-Path (Wsl-Win "/tools/preview_apply.py"))) { return }
    Write-PreviewProfile
    Set-Status (L "s_over_draw") $true
    $argv = @($PyWsl, "tools/preview_apply.py", "--json", "--image", (To-WslPath $script:PreviewPath),
              "--profile", $PreviewWsl, "--frames", "1", "--out", "/tmp/ds_preview_tab", "--models") + @(Chosen-Models)
    $r = From-Json (Invoke-Wsl $argv)
    Set-Status $(if ($script:Running) { L "s_running" } else { "" })
    if ($null -eq $r -or -not $r.ok) {
        $overCap.Text = ((L "m_preview_fail") + " " + $(if ($null -eq $r) { "" } else { [string]$r.why }))
        return
    }
    Set-OverShot ([string]$r.overlay) ([string]$r.summary)
}
function Queue-OverShot {
    # 오버레이를 고칠 때마다 다시 그린다 — 연달아 누르면 마지막 것만 (한 번 그리는 데 몇 초 걸린다)
    if ($script:Building -or $null -eq $script:OverShot -or -not $script:PreviewPath) { return }
    $script:OverShot.Stop()
    $script:OverShot.Start()
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
        foreach ($it in (Label-Items)) {
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
        foreach ($k in @(@("count", (L "flg_count")), @("roi", (L "flg_roi")), @("show_non_suspect", (L "flg_nonsus")),
                         @("show_rejected", (L "flg_rejected")), @("title", (L "flg_title")), @("pose", (L "flg_pose")))) {
            if ($cat.overlay.$m.($k[0]) -ne $true) { continue }
            if ($k[0] -eq "show_rejected" -and -not ((Checked-Targets) -contains "demo_box")) { continue }
            if ($x -gt 450) { $x = 12; $fy += 26 }
            $cb = New-Check $g $k[1] $x $fy ([bool](Overlay-Value $m $k[0]))
            $cb.Tag = @{ model = $m; kind = "flag"; key = $k[0] }
            $cb.Add_CheckedChanged({ if ($script:Building) { return }; Mark-Overlay $this.Tag.model $this.Tag.key $this.Checked; Changed; Queue-OverShot })
            $x += 130
        }
        if ($hasCards) {
            [void](New-Label $g (L "h_cards") 12 114)
            $cx = 60
            $cur = Overlay-Value $m "cards"
            foreach ($cm in $cat.card_modes) {
                $r = New-Object System.Windows.Forms.RadioButton
                $r.Text = $cm.shown; $r.AutoSize = $true; $r.Location = New-Object System.Drawing.Point($cx, 112)
                $r.Checked = ($cur -eq $cm.key); $r.Tag = @{ model = $m; mode = $cm.key }
                $r.Add_CheckedChanged({ if ($script:Building -or -not $this.Checked) { return }; Mark-Overlay $this.Tag.model "cards" $this.Tag.mode; Changed; Queue-OverShot })
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
    Queue-OverShot
}

# ================================================================ 2단계 함수 — 근거 표 · 기기별 기능 (2026-11-26)
# 기기마다 올릴 기능을 고른다. 적지 않은 기기는 고른 기능 전부다(지금까지의 동작). 모두 끄면 그 기기에는 비전이 없다.
# 기능 조합이 같은 기기끼리 한 프로세스로 묶인다 — 조합이 갈리면 프로세스와 GPU 메모리도 갈린다.
function On-All {
    $on = @($Sources | Where-Object { $srcChecks[$_].Checked })
    return @($on) + @(if ($sliceOn.Checked) { "cam360_slice" })
}
function Assign-For($s) {
    # 그 소스에 올릴 모델 — 그 소스에 고른 기능들의 모델이다 (기능 선택 하나에서 나온다)
    $want = @{}
    foreach ($fn in (Chosen-Functions)) {
        if ((Fn-Sources $fn) -contains $s) {
            $m = Fn-Model $fn
            if ($m) { $want[$m] = $true }
        }
    }
    return @($Models | Where-Object { $want.ContainsKey($_) })
}
function Assign-Map {
    # 기본값(고른 모델 전부)과 같은 소스는 적지 않는다 — 적어 두면 소스를 켜고 끌 때마다 비전이 괜히 다시 뜬다
    $chosen = @(Chosen-Models)
    $out = @{}
    foreach ($s in (On-All)) {
        $keep = @(Assign-For $s)
        if (($keep -join ",") -ne ($chosen -join ",")) { $out[$s] = $keep }
    }
    return $out
}
function On-MatchCheck($cb) {
    if ($script:Building) { return }
    $s = $cb.Tag.src
    $now = @(Assign-For $s)
    $script:Assign[$s] = @(Chosen-Models | Where-Object { if ($_ -eq $cb.Tag.model) { $cb.Checked } else { $now -contains $_ } })
    Build-Match
    Changed
}
function Build-CueTable {
    # 고른 **기능**마다 그 기능이 쓰는 근거를 펼친다. 첫 칸은 기능 이름(지정 물체 탐지 …)이고,
    # 「역할」은 그 근거가 하는 일(형태 비교 …), 「모델」은 그 일에 쓰이는 가중치다 (2026-11-26)
    $script:Building = $true
    $cueTable.Controls.Clear()
    $fns = @(Chosen-Functions | Where-Object { @(Fn-Options $_).Count -gt 0 })
    if (-not $visOn.Checked -or $fns.Count -eq 0) {
        [void](New-Label $cueTable $(if ($visOn.Checked) { L "n_nomodel" } else { L "n_novision" }) 4 6)
        $script:Building = $false
        return
    }
    $cols = @(4, 150, 280, 400, 500)
    $hdr = @("h_capability", "h_cue", "h_role2", "h_model", "h_gpu")
    for ($i = 0; $i -lt $hdr.Count; $i++) { [void](New-Label $cueTable (L $hdr[$i]) $cols[$i] 4) }
    $y = 26
    foreach ($fn in $fns) {
        $opt = Fn-Model $fn
        $rows = @($cat.options.$opt)
        $first = $true
        if ($rows.Count -eq 0) {
            [void](New-Label $cueTable (Fn-Text2 $fn) $cols[0] $y)
            [void](New-Label $cueTable (L "v_rule") $cols[1] $y)
            $y += 22
            continue
        }
        foreach ($r in $rows) {
            [void](New-Label $cueTable $(if ($first) { Fn-Text2 $fn } else { "" }) $cols[0] $y)
            [void](New-Label $cueTable $r.label $cols[1] $y)
            [void](New-Label $cueTable (Cue-Role $r.key) $cols[2] $y)
            $ms = @(Cue-Models $r)
            [void](New-Label $cueTable $(if ($ms.Count) { $ms -join ", " } else { L "v_rule" }) $cols[3] $y)
            $mb = Cue-MB $r
            if ($null -eq $mb -and $first -and $opt -ne "targeted") { $mb = [int]$cat.gpu_mb_option.$opt }
            [void](New-Label $cueTable $(if ($null -ne $mb) { "{0} MB" -f $mb } else { "-" }) $cols[4] $y)
            $y += 22
            $first = $false
        }
    }
    $base = [int]$cat.gpu_mb_base
    $add = Chosen-MB
    [void](New-Label $cueTable ((L "n_total") -f $base, $add, ($base + $add)) $cols[0] ($y + 8))
    $script:Building = $false
}
function Build-Match {
    # 기능 × 소스 격자 — 세로는 기능(물체 탐지 · 추적 …), 가로는 1단계에서 켠 소스. 기능마다 쓸 모델을 고른다.
    # 여기서 고른 것이 그대로 오케스트레이터의 --functions(소스별) 와 비전의 모델이 된다 (2026-11-26)
    $script:Building = $true
    $matchPanel.Controls.Clear()
    $srcs = @(On-All)
    if (-not $visOn.Checked) { $matchNote.Text = (L "n_novision") }
    elseif ($srcs.Count -eq 0) { $matchNote.Text = (L "n_nosource") }
    else { $matchNote.Text = (L "n_match") }
    if ($visOn.Checked -and $srcs.Count -gt 0 -and $null -ne $script:FnList) {
        $x0 = 170                                    # 모델 콤보
        $x1 = $x0 + 118                              # 소스 칸들
        [void](New-Label $matchPanel (L "h_capability") 4 6)
        [void](New-Label $matchPanel (L "h_model") $x0 6)
        $i = 0
        foreach ($s2 in $srcs) {
            [void](New-Label $matchPanel $s2 ($x1 + $i * 62) 6)
            $i++
        }
        $y = 30
        foreach ($fn in $script:FnList) {
            $usable = Fn-Usable $fn
            # 기능을 쓸지 고르는 상자 — 켜면 켠 소스 전부에 붙이고, 끄면 전부 뗀다
            $pick = New-Check $matchPanel (Fn-Text2 $fn) 4 $y (Fn-On $fn)
            $pick.Enabled = $usable
            $pick.Tag = @{ fn = $fn }
            $pick.Add_CheckedChanged({ On-FnPick $this })
            $modelTip.SetToolTip($pick, (Fn-Tip $fn))
            $opts = @(Fn-Options $fn)
            $dom = @($opts | Where-Object { Option-Domain $_ })
            if ($opts.Count -gt 0 -and $dom.Count -eq $opts.Count) {
                # 도메인 전용 기능(지정 물체 탐지 · 하천 부유물 탐지 …) — 고를 모델이 아니다. 실제로 쓰는 모델을 적는다
                $lb = New-Label $matchPanel (Option-ModelsText $opts[0]) $x0 ($y + 3)
                $lb.AutoSize = $false
                $lb.Width = 112
                $lb.AutoEllipsis = $true
                $modelTip.SetToolTip($lb, (Option-Tip $opts[0]))
            } elseif ($opts.Count -gt 0) {
                $cb = New-Object System.Windows.Forms.ComboBox
                $cb.DropDownStyle = "DropDownList"
                $cb.Location = New-Object System.Drawing.Point($x0, $y)
                $cb.Width = 112
                foreach ($o in $opts) { [void]$cb.Items.Add($ModelShown[$o]) }
                $cur = Fn-Model $fn
                $cb.SelectedIndex = [Math]::Max(0, [Array]::IndexOf($opts, $cur))
                $cb.Tag = @{ fn = $fn; opts = $opts }
                $cb.Add_SelectionChangeCommitted({ On-FnModel $this })
                $modelTip.SetToolTip($cb, (Option-Tip (Fn-Model $fn)))
                $cb.Enabled = $usable
                $matchPanel.Controls.Add($cb)
            } else {
                [void](New-Label $matchPanel "-" $x0 ($y + 3))
            }
            $i = 0
            foreach ($s2 in $srcs) {
                $on = (Fn-Sources $fn) -contains $s2
                $chk = New-Check $matchPanel "" ($x1 + $i * 62) $y $on
                $chk.Tag = @{ fn = $fn; src = $s2 }
                $chk.Enabled = $usable
                $chk.Add_CheckedChanged({ On-FnSource $this })
                if (-not $usable) { $modelTip.SetToolTip($chk, (Fn-Why $fn)) }
                $i++
            }
            if (-not $usable) {
                $modelTip.SetToolTip($pick, ((Fn-Tip $fn) + "`n" + (Fn-Why $fn)))
                # 왜 안 되는지 + 바로 고치는 길 (모델 받기)
                $fx2 = $x1 + $srcs.Count * 62 + 6
                $gb = New-Object System.Windows.Forms.Button
                $gb.Location = New-Object System.Drawing.Point($fx2, $y)
                $gb.Width = 70
                $gb.Text = (L "b_get")
                $gb.Tag = @{ opt = (Fn-Model $fn) }
                $gb.Add_Click({ Get-Model $this.Tag.opt })
                $modelTip.SetToolTip($gb, (Fn-Why $fn))
                $matchPanel.Controls.Add($gb)
            }
            $y += 26
        }
    }
    $script:Building = $false
}
function Fn-Text2($fn) {
    $e = $FnName[$fn]
    if ($null -eq $e) { return $fn }
    $n = $e[$script:Lang]
    if (-not $n) { $n = $e["ko"] }
    return $n
}
function Fn-Tip($fn) {
    $why = Fn-Text $fn "why"
    $lines = @($fn)
    if ($why) { $lines += $why }
    $opts = @(Fn-Options $fn)
    if ($opts.Count) { $lines += ((L "h_model") + ": " + ($opts -join ", ")) }
    return ($lines -join "`n")
}
function On-FnModel($cb) {
    if ($script:Building) { return }
    $script:FnModel[$cb.Tag.fn] = @($cb.Tag.opts)[$cb.SelectedIndex]
    Build-CueTable
    Build-Overlay
    Changed
}
function On-FnPick($chk) {
    # 기능 선택 — 켜면 지금 켠 소스 전부에, 끄면 전부 뗀다 (소스마다 따로 고르는 것은 오른쪽 칸)
    if ($script:Building) { return }
    $fn = $chk.Tag.fn
    $script:FnSrc[$fn] = $(if ($chk.Checked) { @(On-All) } else { @() })
    Build-Match
    Build-CueTable
    Build-Overlay
    Sync-Enabled
    Update-Nav
    Changed
}
function On-FnSource($chk) {
    if ($script:Building) { return }
    $fn = $chk.Tag.fn
    $src = $chk.Tag.src
    $now = @($script:FnSrc[$fn])
    if ($chk.Checked) {
        if ($now -notcontains $src) { $now += $src }
    } else {
        $now = @($now | Where-Object { $_ -ne $src })
    }
    $script:FnSrc[$fn] = @($now)
    Build-Match
    Build-CueTable
    Build-Overlay
    Update-Nav
    Changed
}

# ================================================================ 5단계 — 배치 (2026-11-26)
# 자동이면 배치 계산기가 자리를 고른다. 수동으로 고른 자리는 **한도를 면제**하지만 하드웨어 조건(센서 · 가속기 · arch)은
# 그대로 지킨다 — 카메라 없는 노드에 카메라 작업을 둘 수는 없다. 「올리지 않음」은 그 작업을 빼는 것이다.
$gPlace = New-Group $tPlace "" 8 8 596 112
[void](Reg $gPlace "g_place")
# 자원 배분 — 켜면 비전을 오케스트레이터(python -m ds_resources.cli run)로 띄운다: 자원에 맞춰 배치 · GPU 여유 확인 ·
# fps 상한 · k3s 가 있으면 파이까지. 끄면 예전처럼 창을 직접 나눠 띄운다 (2026-11-24)
$orchOn = Reg (New-Check $gPlace "" 14 24 $(if ($null -ne $s.orchestrate) { [bool]$s.orchestrate } else { $true })) "c_orch"
$planBtn = New-Object System.Windows.Forms.Button
$planBtn.Location = New-Object System.Drawing.Point(440, 20)
$planBtn.Width = 140
[void](Reg $planBtn "b_plan")
$gPlace.Controls.Add($planBtn)
$autoRadio = New-Object System.Windows.Forms.RadioButton
$autoRadio.AutoSize = $true
$autoRadio.Location = New-Object System.Drawing.Point(14, 52)
$autoRadio.Checked = -not [bool]$s.manual
[void](Reg $autoRadio "r_auto")
$gPlace.Controls.Add($autoRadio)
$manualOn = New-Object System.Windows.Forms.RadioButton
$manualOn.AutoSize = $true
$manualOn.Location = New-Object System.Drawing.Point(14, 78)
$manualOn.Checked = [bool]$s.manual
[void](Reg $manualOn "r_manual")
$gPlace.Controls.Add($manualOn)
$manualLoad = New-Object System.Windows.Forms.Button
$manualLoad.Location = New-Object System.Drawing.Point(440, 74)
$manualLoad.Width = 140
[void](Reg $manualLoad "b_load")
$gPlace.Controls.Add($manualLoad)
$manualNote = New-Object System.Windows.Forms.Label
$manualNote.AutoSize = $false
$manualNote.Location = New-Object System.Drawing.Point(8, 124)
$manualNote.Size = New-Object System.Drawing.Size(596, 20)
$tPlace.Controls.Add($manualNote)
$manualPanel = New-Object System.Windows.Forms.Panel
$manualPanel.Location = New-Object System.Drawing.Point(8, 146)
$manualPanel.Size = New-Object System.Drawing.Size(596, 390)
$manualPanel.AutoScroll = $true
$tPlace.Controls.Add($manualPanel)

function Node-Shown($n) {
    # 1단계 소스에서 켠 기기만 보여 준다 — 안 켠 기기에는 올릴 작업이 없다 (2026-11-26).
    # 어떤 기기도 맡지 않은 노드와 서버(소스 칸에 없는 이름)는 늘 보인다
    $src = [string]$n.source
    if (-not $src -or -not $srcChecks.ContainsKey($src)) { return $true }
    return [bool]$srcChecks[$src].Checked
}
function Nodes-Shown {
    if ($null -eq $script:Places) { return @() }
    return @($script:Places.nodes | Where-Object { Node-Shown $_ })
}
function Node-Choices($u) {
    if ($null -eq $script:Places) { return @() }
    return @((Nodes-Shown) | Where-Object {
        if ($u.pin -eq "device" -and -not $_.limited) { return $false }
        if ($u.pin -eq "server" -and $_.limited) { return $false }
        return ((@($u.roles).Count -eq 0) -or (@($u.roles) -contains $_.role))
    })
}
function On-ManualPick($cb) {
    if ($script:Building) { return }
    $unit = $cb.Tag.unit
    $v = [string]$cb.SelectedItem
    if ($cb.SelectedIndex -eq 0) { $script:ManualPlace.Remove($unit) }   # 첫 항목이 「자동」이다
    elseif ($v -eq (L "v_none")) { $script:ManualPlace[$unit] = "none" }
    else {
        # 화면에는 「ep-001 온디바이스 (pi1)」로 보이지만 넘기는 것은 진짜 노드 이름(pi1)이다
        $real = $cb.Tag.byLabel[$v]
        $script:ManualPlace[$unit] = $(if ($real) { $real } else { $v })
    }
    Update-Bars                                     # 고른 자리에 따라 막대가 차고 빈다
    Changed
}
function Limit-Of($name, $key, $default) {
    $e = $script:NodeLimit[$name]
    if ($null -ne $e -and $null -ne $e[$key]) { return $e[$key] }
    return $default
}
function Show-Limit($node) {
    # 한도는 한 가지가 아니다 — 비울 비율(CPU · 메모리) · 비울 코어 · 온도 한계 · GPU 예비.
    # 빈 칸은 「없음(기본값)」이다. 서버도 걸 수 있다 — 걸면 배치가 서버에서도 그 한도를 지킨다
    $d = New-Object System.Windows.Forms.Form
    $d.Text = ((L "b_limit") + " — " + $node)
    $d.Font = $font
    $d.StartPosition = "CenterParent"
    $d.FormBorderStyle = "FixedDialog"
    $d.MaximizeBox = $false
    $d.MinimizeBox = $false
    $d.ClientSize = New-Object System.Drawing.Size(560, 290)
    $boxes = @{}
    $i = 0
    foreach ($row in @(@("headroom_cpu", (L "h_lim_cpu"), (L "t_lim_cpu")),
                       @("headroom_mem", (L "h_lim_mem"), (L "t_lim_mem")),
                       @("cores_reserved", (L "h_lim_cores"), (L "t_lim_cores")),
                       @("temp_max", (L "h_lim_temp"), (L "t_lim_temp")),
                       @("gpu_spare_mb", (L "h_lim_gpu"), (L "t_lim_gpu")))) {
        $yy = 16 + $i * 34
        [void](New-Label $d $row[1] 14 ($yy + 3))
        $tb = New-Object System.Windows.Forms.TextBox
        $tb.Location = New-Object System.Drawing.Point(170, $yy)
        $tb.Width = 70
        $v = Limit-Of $node $row[0] $null
        $tb.Text = $(if ($null -eq $v) { "" } else { [string]$v })
        $d.Controls.Add($tb)
        $boxes[$row[0]] = $tb
        $hint = New-Label $d $row[2] 250 ($yy + 3)
        $hint.AutoSize = $false
        $hint.Size = New-Object System.Drawing.Size(296, 30)
        $i++
    }
    $note = New-Label $d (L "n_limit") 14 196
    $note.AutoSize = $false
    $note.Size = New-Object System.Drawing.Size(532, 44)
    $ok = New-Object System.Windows.Forms.Button
    $ok.Text = (L "b_save")
    $ok.Location = New-Object System.Drawing.Point(340, 250)
    $ok.Width = 96
    $cancel = New-Object System.Windows.Forms.Button
    $cancel.Text = (L "b_close")
    $cancel.Location = New-Object System.Drawing.Point(446, 250)
    $cancel.Width = 96
    $saved = @{ v = $false }
    $ok.Add_Click({ $saved.v = $true; $d.Close() })
    $cancel.Add_Click({ $d.Close() })
    $d.Controls.Add($ok)
    $d.Controls.Add($cancel)
    $d.AcceptButton = $ok
    [void]$d.ShowDialog($form)
    if (-not $saved.v) { return }
    $one = @{}
    foreach ($k in $boxes.Keys) {
        $txt = $boxes[$k].Text.Trim()
        if ($txt -eq "") { continue }
        $num = [double]0
        if (-not [double]::TryParse($txt, [System.Globalization.NumberStyles]::Float,
                                    [System.Globalization.CultureInfo]::InvariantCulture, [ref]$num)) {
            [void][System.Windows.Forms.MessageBox]::Show((L "m_limit_num") -f $k, (L "d_title"))
            return
        }
        # 비율은 사람이 % 로 적는다 (20 → 0.20). 1 이하면 비율 그대로 받는다
        if (($k -eq "headroom_cpu" -or $k -eq "headroom_mem") -and $num -gt 1) { $num = $num / 100.0 }
        $one[$k] = $num
    }
    if ($one.Count -eq 0) { $script:NodeLimit.Remove($node) } else { $script:NodeLimit[$node] = $one }
    Load-Places
    Changed
}
function Limit-Args {
    $parts = @()
    foreach ($k in ($script:NodeLimit.Keys | Sort-Object)) {
        $v = $script:NodeLimit[$k]
        $bits = @()
        foreach ($kv in @(@("headroom_cpu", "cpu"), @("headroom_mem", "mem"), @("cores_reserved", "cores"),
                          @("temp_max", "temp"), @("gpu_spare_mb", "gpu"))) {
            if ($null -ne $v[$kv[0]]) { $bits += ("{0}:{1}" -f $kv[1], $v[$kv[0]]) }
        }
        if ($bits.Count) { $parts += ("{0}={1}" -f $k, ($bits -join ",")) }
    }
    if ($parts.Count -eq 0) { return "" }
    return " --limit " + ($parts -join " ")
}
function Draw-Legend($s, $e) {
    # 막대의 색이 무엇인지 — 회색(쓰는 중), 파랑(올린 작업), 빨강(넘침)
    $g = $e.Graphics
    $x = 0
    $node = $s.Tag.node
    $all = @(Node-Funcs $node)
    $mine = @($all | Where-Object { -not $_.auto } | ForEach-Object { $_.text })
    $autos = @($all | Where-Object { $_.auto } | ForEach-Object { $_.text })
    $over = $false
    foreach ($kind in @("cpu", "mem")) {
        $b = $script:Bars[("{0}|{1}" -f $node, $kind)]
        if ($null -ne $b -and $null -ne $b.bar) {
            if ((Node-Load $node $kind) -gt [double]$b.bar.Tag.room) { $over = $true }
        }
    }
    # @(색, 글) 로 담으면 PowerShell 이 배열을 펴 버린다 — 객체로 담는다 (2026-11-26 실측: FillRectangle 형변환 오류)
    $items = @([pscustomobject]@{ b = [System.Drawing.Brushes]::Silver; t = (L "lg_used") })
    if ($mine.Count) { $items += [pscustomobject]@{ b = [System.Drawing.Brushes]::SteelBlue; t = ($mine -join ", ") } }
    if ($autos.Count) {
        $items += [pscustomobject]@{ b = [System.Drawing.Brushes]::LightSteelBlue; t = ($autos -join ", ") }
    }
    if (-not $mine.Count -and -not $autos.Count) {
        $items += [pscustomobject]@{ b = [System.Drawing.Brushes]::SteelBlue; t = (L "lg_none") }
    }
    if ($over) { $items += [pscustomobject]@{ b = [System.Drawing.Brushes]::IndianRed; t = (L "lg_over") } }
    foreach ($it in $items) {
        $g.FillRectangle($it.b, $x, 5, 10, 10)
        $g.DrawRectangle([System.Drawing.Pens]::Gray, $x, 5, 10, 10)
        $x += 14
        $txt = [string]$it.t
        $g.DrawString($txt, $s.Font, [System.Drawing.Brushes]::Black, $x, 2)
        $x += [int]$g.MeasureString($txt, $s.Font).Width + 12
        if ($x -gt $s.ClientSize.Width - 40) { break }
    }
}
function Node-Load($name, $kind) {
    # 내가 이 노드에 붙인 작업들의 합 (수동으로 고른 것만 — 자동은 배치 계산기가 정한다)
    $sum = 0
    if ($null -eq $script:Places) { return 0 }
    if ($kind -eq "gpu") {
        # GPU 작업은 고른 기능(모델)이다 — 2단계의 어림과 같은 셈. 지금은 서버에서만 돈다
        foreach ($n in @($script:Places.nodes)) {
            if ($n.name -eq $name -and -not $n.limited -and $visOn.Checked) {
                return [double]([int]$cat.gpu_mb_base + (Chosen-MB))
            }
        }
        return 0
    }
    foreach ($u in @($script:Places.units)) {
        if ($script:ManualPlace[$u.unit] -ne $name) { continue }
        $sum += [double]$(if ($kind -eq "cpu") { $u.cpu_m } else { $u.mem_mb })
    }
    return $sum
}
function Node-Funcs($name) {
    # 그 노드에 올린 작업 — 이름(소스)과 그 작업이 쓰는 양까지. 수동으로 고른 것과 자동 예정을 나눠 돌려준다
    if ($null -eq $script:Places) { return @() }
    $out = @()
    foreach ($u in @($script:Places.units)) {
        $pick = $script:ManualPlace[$u.unit]
        $auto = (-not $pick) -and ($u.auto_node -eq $name)
        if ($pick -ne $name -and -not $auto) { continue }
        $out += [pscustomobject]@{
            text = "{0} {1:n1}{2} {3:n0}MB" -f (Unit-Name $u), ($u.cpu_m / 1000), (L "w_cores"), $u.mem_mb
            auto = $auto
        }
    }
    return @($out)
}
function Draw-Bar($s, $e) {
    # 회색 = 이미 쓰거나 비워 둔 몫 · 파랑 = 내가 붙인 작업 · 빨강 = 넘긴 만큼
    $t = $s.Tag
    $alloc = [double]$t.alloc
    if ($alloc -le 0) { return }
    $room = [double]$t.room
    $mine = Node-Load $t.node $t.kind
    $w = $s.ClientSize.Width
    $h = $s.ClientSize.Height
    $g = $e.Graphics
    $used = [Math]::Max(0.0, $alloc - $room)
    $px = { param($v) [int][Math]::Round([Math]::Max(0.0, [Math]::Min($v, $alloc)) / $alloc * $w) }
    $uw = & $px $used
    $mw = & $px ([Math]::Min($mine, $room))
    $ow = & $px ([Math]::Max(0.0, $mine - $room))
    $g.FillRectangle([System.Drawing.Brushes]::WhiteSmoke, 0, 0, $w, $h)
    if ($uw -gt 0) { $g.FillRectangle([System.Drawing.Brushes]::Silver, 0, 0, $uw, $h) }
    if ($mw -gt 0) { $g.FillRectangle([System.Drawing.Brushes]::SteelBlue, $uw, 0, $mw, $h) }
    if ($ow -gt 0) { $g.FillRectangle([System.Drawing.Brushes]::IndianRed, ($uw + $mw), 0, $ow, $h) }
    $g.DrawRectangle([System.Drawing.Pens]::Gray, 0, 0, ($w - 1), ($h - 1))
}
function Update-Bars {
    if ($null -eq $script:Bars) { return }
    foreach ($n in @($script:Places.nodes)) {
        foreach ($kind in @("cpu", "mem", "gpu")) {
            $e = $script:Bars[("{0}|{1}" -f $n.name, $kind)]
            if ($null -eq $e) { continue }
            $mine = Node-Load $n.name $kind
            # 막대 옆에는 「지금 쓰는 양 / 전체」만 — 무엇이 얼마나 쓰는지는 아래 범례가 기능마다 적는다
            if ($kind -eq "gpu") {
                $e.text.Text = "GPU {0:n1} / {1:n1} {2}" -f ($n.gpu.used_mb / 1024), ($n.gpu.total_mb / 1024),
                               (L "w_gb")
            } elseif ($kind -eq "cpu") {
                $e.text.Text = "CPU {0:n1} / {1:n1} {2}" -f (($n.alloc.cpu_m - $n.room.cpu_m) / 1000),
                               ($n.alloc.cpu_m / 1000), (L "w_cores")
            } else {
                $e.text.Text = "{0} {1:n1} / {2:n1} {3}" -f (L "w_mem"),
                               (($n.alloc.mem_mb - $n.room.mem_mb) / 1024), ($n.alloc.mem_mb / 1024), (L "w_gb")
            }
            $e.bar.Invalidate()
        }
        $f = $script:Bars[("{0}|funcs" -f $n.name)]
        if ($null -ne $f -and $null -ne $f.legend) { $f.legend.Invalidate() }
    }
}
function Build-Manual {
    # 작업마다 「무엇을 위한 것 · 이 조각의 역할 · 어느 기기」를 사람 말로 적는다. 진짜 이름(object_detect.device)과
    # 둘 수 있는 곳 · 한도 면제 여부는 풍선말에 둔다 — 줄이 길어지면 읽히지 않는다 (2026-11-26)
    $script:Building = $true
    $manualPanel.Controls.Clear()
    if ($null -eq $script:Places) {
        $manualNote.Text = (L "n_manual")
        $script:Building = $false
        return
    }
    $manualNote.Text = $script:Places.mode
    $y = 4
    [void](New-Label $manualPanel (L "l_node") 4 $y)
    $y += 22
    $script:Bars = @{}
    foreach ($n in @(Nodes-Shown)) {
        $state = if ($n.ready -and -not $n.stale) { L "v_usable" } elseif ($n.stale) { "stale" } else { L "v_notready" }
        $limit = if ($n.limited) { L "v_limited" } else { L "v_unlimited" }
        $lab = New-Label $manualPanel ("{0}  —  {1},  {2} {3}" -f (Node-Label $n), $state, (L "h_limit"), $limit) 12 $y
        $caps = @($n.caps.sensors) -join " "
        $manualTip.SetToolTip($lab, ("{0}, {1}{2}" -f $n.name, $n.role, $(if ($caps) { ", " + $caps } else { "" })))
        $y += 20
        # 노드별 한도 — 창 하나에서 고친다 (서버에도 걸 수 있다. 기본은 「없음」)
        $lb = New-Object System.Windows.Forms.Button
        $lb.Location = New-Object System.Drawing.Point(466, ($y - 22))
        $lb.Width = 96
        $lb.Text = (L "b_limit")
        $lb.Tag = @{ node = $n.name }
        $lb.Add_Click({ Show-Limit $this.Tag.node })
        $manualTip.SetToolTip($lb, (L "t_limit"))
        $manualPanel.Controls.Add($lb)
        if ($null -ne $n.room) {
            $kinds = @("cpu", "mem")
            if ($null -ne $n.gpu) { $kinds += "gpu" }
            foreach ($kind in $kinds) {
                $bar = New-Object System.Windows.Forms.Panel
                $bar.Location = New-Object System.Drawing.Point(28, $y)
                $bar.Size = New-Object System.Drawing.Size(240, 14)
                $alloc = switch ($kind) { "cpu" { $n.alloc.cpu_m } "mem" { $n.alloc.mem_mb } "gpu" { $n.gpu.total_mb } }
                $room = switch ($kind) {
                    "cpu" { $n.room.cpu_m }
                    "mem" { $n.room.mem_mb }
                    "gpu" { [Math]::Max(0, $n.gpu.total_mb - $n.gpu.used_mb) }
                }
                $bar.Tag = @{ node = $n.name; kind = $kind; alloc = $alloc; room = $room }
                $bar.Add_Paint({ param($s, $e) Draw-Bar $s $e })
                $manualPanel.Controls.Add($bar)
                $txt = New-Label $manualPanel "" 276 ($y - 1)
                $txt.AutoSize = $false
                $txt.Size = New-Object System.Drawing.Size(280, 16)
                $script:Bars[("{0}|{1}" -f $n.name, $kind)] = @{ bar = $bar; text = $txt }
                $y += 17
            }
        }
        $fn = New-Object System.Windows.Forms.Panel
        $fn.Location = New-Object System.Drawing.Point(28, $y)
        $fn.Size = New-Object System.Drawing.Size(540, 20)
        $fn.Tag = @{ node = $n.name }
        $fn.Add_Paint({ param($s, $e) Draw-Legend $s $e })
        $manualPanel.Controls.Add($fn)
        $script:Bars[("{0}|funcs" -f $n.name)] = @{ legend = $fn }
        $y += 24
    }
    Update-Bars
    $y += 10
    [void](New-Label $manualPanel (L "l_task") 4 $y)
    $y += 22
    [void](New-Label $manualPanel (L "h_purpose") 12 $y)
    [void](New-Label $manualPanel (L "h_role2") 180 $y)
    [void](New-Label $manualPanel (L "h_res_need") 300 $y)
    [void](New-Label $manualPanel (L "h_slot") 440 $y)
    $y += 20
    foreach ($u in @($script:Places.units)) {
        $name = New-Label $manualPanel (Unit-Name $u) 12 ($y + 4)
        $why = Fn-Text $u.function "why"
        $where = if ($u.pin -ne "none") { "pin " + $u.pin } else { (@($u.roles) -join ", ") }
        if ($u.must_device) { $where += ", " + (L "v_waived") }
        $manualTip.SetToolTip($name, ("{0}`n{1}`n{2} {3}" -f $u.unit, $why, (L "h_where"), $where))
        $role = New-Label $manualPanel (Unit-Role $u) 180 ($y + 4)
        $manualTip.SetToolTip($role, $why)
        # 예상 사용 자원 — 계약에 적힌 선언값 (실측이 쌓이면 배치가 그 값으로 다시 잡는다)
        $need = if ($null -ne $u.cpu_m) { "{0:n1} {1} + {2:n0} MB" -f ($u.cpu_m / 1000), (L "w_cores"), $u.mem_mb } else { "-" }
        [void](New-Label $manualPanel $need 300 ($y + 4))
        $cb = New-Object System.Windows.Forms.ComboBox
        $cb.DropDownStyle = "DropDownList"
        $cb.Location = New-Object System.Drawing.Point(440, $y)
        $cb.Width = 134
        $byLabel = @{}
        [void]$cb.Items.Add((Unit-Auto $u))
        foreach ($n in (Node-Choices $u)) {
            $lab = Node-Short $n                      # 칸이 좁다 — 콤보에는 그 기기가 맡은 이름만 (전체는 노드 목록에)
            $byLabel[$lab] = $n.name
            [void]$cb.Items.Add($lab)
        }
        [void]$cb.Items.Add((L "v_none"))
        $cur = $script:ManualPlace[$u.unit]
        if (-not $cur) {
            $cb.SelectedIndex = 0                        # 자동 — 첫 항목이 예정 노드까지 적고 있다
        } elseif ($cur -eq "none") {
            $cb.SelectedItem = (L "v_none")
        } else {
            $hit = @($byLabel.Keys | Where-Object { $byLabel[$_] -eq $cur } | Select-Object -First 1)
            $cb.SelectedItem = $(if ($hit.Count -gt 0) { $hit[0] } else { $cur })
        }
        if ($null -eq $cb.SelectedItem) { $cb.SelectedIndex = 0 }
        $cb.Enabled = $manualOn.Checked
        $cb.Tag = @{ unit = $u.unit; byLabel = $byLabel }
        # SelectionChangeCommitted 는 **사람이 고를 때만** 온다. SelectedIndexChanged 로 받으면 WinForms 가
        # 콤보 핸들을 만들 때 다시 고르면서 이벤트를 쏘아, 고르지도 않은 자리가 수동 배치에 적힌다 (2026-11-26 실측)
        $cb.Add_SelectionChangeCommitted({ On-ManualPick $this })
        $manualPanel.Controls.Add($cb)
        $y += 28
    }
    $script:Building = $false
}
function Reload-Catalog {
    # 가중치를 넣은 뒤 「(없음)」이 사라지도록 선택지를 다시 읽는다 (모델 목록은 그대로다 — available 만 바뀐다)
    try {
        $txt = Invoke-Wsl @($PyWsl, "vision/cue_catalog.py", "--json")
        $new = From-Json $txt
        if ($null -ne $new -and $null -ne $new.models) { $script:cat = $new }
    } catch { }
    Build-CueTable
    Build-CueBoxes
    Build-Overlay
    Sync-Enabled
}
function Weight-Rows {
    # 기능 → 지금 쓰는 모델(선택지) → 역할마다 쓰는 모델과 파일 상태 · 성능 · 자원 · 프롬프트.
    # 상태는 tools/fetch_models.py --check 에서, 성능과 자원은 레지스트리(cli places 의 model_info)에서 온다
    $stat = @{}
    $txt = Invoke-Wsl @($PyWsl, "tools/fetch_models.py", "--all", "--check", "--json")
    $chk = From-Json $txt
    foreach ($it in @($chk.items)) {
        if ($null -eq $stat[$it.model]) { $stat[$it.model] = @() }
        $stat[$it.model] += [pscustomobject]@{ status = $it.status; file = (Split-Path -Leaf $it.path) }
    }
    $rows = @()
    foreach ($fn in $script:FnList) {
        $opt = Fn-Model $fn
        if (-not $opt) { continue }
        $info = $script:ModelInfo.$opt
        foreach ($r in @($cat.options.$opt)) {
            foreach ($m in @($r.models)) {
                $st = @($stat[$m.key])
                $bad = @($st | Where-Object { $_.status -ne "ok" })
                $rows += [pscustomobject]@{
                    fn = $fn; shown = (Fn-Text2 $fn); option = $opt; role = $r.label; model = $m.key
                    ok = ($st.Count -gt 0 -and $bad.Count -eq 0)
                    files = @($st | ForEach-Object { $_.file })
                    why = (($bad | ForEach-Object { "{0}({1})" -f $_.file, $_.status }) -join ", ")
                    note = [string]$m.note
                    ms = $(if ($null -ne $info) { [int]$info.p90_ms.m } else { 0 })
                    mb = $(if ($null -ne $info) { [int]$info.gpu_mb } else { 0 })
                    prompt = $(if ($null -ne $info -and $info.prompt) { [string]$info.prompt } else { "-" })
                }
            }
        }
    }
    return @($rows)
}
function Show-Weights {
    # 기능마다 지금 쓰는 모델과 역할 · 성능 · 자원 · 프롬프트를 보여 주고, 바꾸고 싶은 모델을 넣거나 받는다
    Set-Status (L "s_weights") $true
    $rows = @(Weight-Rows)
    Set-Status $(if ($script:Running) { L "s_running" } else { "" })
    $dlg = New-Object System.Windows.Forms.Form
    $dlg.Text = (L "b_weights")
    $dlg.Font = $font
    $dlg.StartPosition = "CenterParent"
    $dlg.FormBorderStyle = "FixedDialog"
    $dlg.MaximizeBox = $false
    $dlg.MinimizeBox = $false
    $dlg.ClientSize = New-Object System.Drawing.Size(880, 470)
    $panel = New-Object System.Windows.Forms.Panel
    $panel.Location = New-Object System.Drawing.Point(8, 8)
    $panel.Size = New-Object System.Drawing.Size(864, 390)
    $panel.AutoScroll = $true
    $dlg.Controls.Add($panel)
    $addBtn = New-Object System.Windows.Forms.Button
    $addBtn.Text = (L "b_add_model")
    $addBtn.Location = New-Object System.Drawing.Point(8, 406)
    $addBtn.Width = 150
    $addBtn.Add_Click({ Put-Weight $null })
    $dlg.Controls.Add($addBtn)
    $note = New-Label $dlg (L "n_models") 168 410
    $close = New-Object System.Windows.Forms.Button
    $close.Text = (L "b_close")
    $close.Location = New-Object System.Drawing.Point(776, 406)
    $close.Width = 96
    $close.Add_Click({ $dlg.Close() })
    $dlg.Controls.Add($close)
    $cols = @(4, 140, 250, 380, 450, 530, 610, 700)
    $heads = @("h_capability", "h_role2", "h_model", "h_state", "h_perf", "h_res_need", "h_prompt", "h_file")
    $y = 4
    for ($i = 0; $i -lt $heads.Count; $i++) { [void](New-Label $panel (L $heads[$i]) $cols[$i] $y) }
    $y += 22
    $last = ""
    foreach ($r in $rows) {
        $name = if ($r.shown -eq $last) { "" } else { $r.shown }
        $last = $r.shown
        [void](New-Label $panel $name $cols[0] ($y + 3))
        [void](New-Label $panel ([string]$r.role) $cols[1] ($y + 3))
        $ml = New-Label $panel ([string]$r.model) $cols[2] ($y + 3)
        if ($r.note) { $manualTip.SetToolTip($ml, [string]$r.note) }
        $stateTxt = if ($r.ok) { L "v_have" } else { L "v_missing" }
        $lab = New-Label $panel $stateTxt $cols[3] ($y + 3)
        if (-not $r.ok) { $manualTip.SetToolTip($lab, [string]$r.why) }
        [void](New-Label $panel $(if ($r.ms) { "{0} ms" -f $r.ms } else { "-" }) $cols[4] ($y + 3))
        [void](New-Label $panel $(if ($r.mb) { "{0} MB" -f $r.mb } else { "-" }) $cols[5] ($y + 3))
        [void](New-Label $panel ([string]$r.prompt) $cols[6] ($y + 3))
        [void](New-Label $panel ((@($r.files) -join ", ")) $cols[7] ($y + 3))
        $y += 24
    }
    [void]$dlg.ShowDialog($form)
    Reload-Catalog
}
function Put-Weight($model) {
    # 파일을 고르면: 자리에 넣고 → 「로딩 중」 → **실제로 열리는지 자동 시험**(tools/model_check.py) → 결과를 알린다.
    # 이름과 sha256 이 레지스트리와 맞아야 들어간다 — 틀리면 그 이유를 그대로 보여 준다
    $fd = New-Object System.Windows.Forms.OpenFileDialog
    $fd.Multiselect = $true
    $fd.Filter = "weights|*.pt;*.onnx;*.json;*.zip|all|*.*"
    if ($fd.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { return }
    $wsl = @($fd.FileNames | ForEach-Object { "/mnt/" + $_.Substring(0, 1).ToLower() + ($_.Substring(2) -replace "\\", "/") })
    Set-Status (L "s_weights") $true
    $txt = Invoke-Wsl (@($PyWsl, "tools/fetch_models.py", "--place") + $wsl + @("--json"))
    $r = $null
    $r = From-Json $txt
    if ($null -eq $r) {
        Set-Status ""
        [void][System.Windows.Forms.MessageBox]::Show((L "m_weights_fail") + "`n" + $txt, (L "d_title"))
        return
    }
    $lines = @()
    foreach ($x in @($r.placed)) { $lines += ("+ {0}: {1}" -f $x.model, $x.path) }
    foreach ($x in @($r.skipped)) { $lines += ("= {0}: {1}" -f $x.path, $x.why) }
    foreach ($x in @($r.errors)) { $lines += ("x {0}: {1}" -f $x.file, $x.why) }
    if ($lines.Count -eq 0) { $lines = @((L "m_weights_none")) }
    # 넣은 것이 있으면 실제로 열리는지 본다 (적재 시험 — 몇 초 걸린다)
    if (@($r.placed).Count -gt 0) {
        Set-Status (L "s_testing") $true
        $keys = @($r.placed | ForEach-Object { $_.model } | Sort-Object -Unique)
        $t2 = Invoke-Wsl (@($PyWsl, "tools/model_check.py", "--json", "--models") + $keys)
        $c = $null
        $c = From-Json $t2
        $lines += ""
        if ($null -eq $c) {
            $lines += (L "m_test_fail")
        } else {
            foreach ($f in @($c.files)) {
                $lines += ("{0} {1}: {2}" -f $(if ($f.ok) { L "v_opened" } else { L "v_not_opened" }), $f.file,
                           $(if ($f.why) { $f.why } else { "{0} ms" -f $f.ms }))
            }
            $lines += $(if ($c.ok) { L "m_test_ok" } else { L "m_test_bad" })
        }
    }
    Set-Status $(if ($script:Running) { L "s_running" } else { "" })
    [void][System.Windows.Forms.MessageBox]::Show(($lines -join "`n"), (L "d_title"))
    Reload-Catalog
}
function Set-SrcState {
    # 마지막으로 확인한 결과를 지금 언어로 다시 적는다 (언어를 바꿔도 남은 글이 어긋나지 않게)
    if ($null -eq $script:SrcLive) { return }
    $live = @($script:SrcLive)
    $down = @($script:SrcDown)
    $srcState.Text = ((L "n_src_live") -f $(if ($live.Count) { $live -join ", " } else { "-" })) +
                     $(if ($down.Count) { "   " + ((L "n_src_down") -f ($down -join ", ")) } else { "" })
}
function Move-SrcFoot {
    # 소스 줄이 한 줄 늘면 아래 글과 단추도 함께 내린다
    $y = $script:SrcY + 30
    $srcState.Location = New-Object System.Drawing.Point(14, $y)
    $srcCheck.Location = New-Object System.Drawing.Point(472, ($y - 4))
    $srcAdd.Location = New-Object System.Drawing.Point(472, ($y + 22))
    $gStream.Height = [Math]::Max(170, $y + 60)
}
function New-SourceCheck($n) {
    # 사람이 더한 소스를 소스 줄 뒤에 하나 붙인다 (config/sources.json 에 적힌 것)
    if ($srcChecks.ContainsKey($n)) { return }
    $c = New-Check $gStream $n $script:SrcX $script:SrcY ($s.sources -contains $n)
    $srcChecks[$n] = $c
    $c.Add_CheckedChanged({ Build-Match; Build-CueTable; Build-Manual; Update-Nav; Changed })
    $script:SrcX += $c.Width + 10
    if ($script:SrcX -gt 470) { $script:SrcX = 80; $script:SrcY += 26 }
    $script:Sources = @($script:Sources) + $n
    Move-SrcFoot
}
function Load-UserSources {
    # 더해 둔 소스를 목록에 들인다 — 런처를 다시 켜도 그대로 보이게
    $r = From-Json (Invoke-Wsl @($PyWsl, "tools/source_add.py", "--list", "--json"))
    if ($null -eq $r -or $null -eq $r.sources) { return }
    foreach ($pr in @($r.sources.PSObject.Properties)) { New-SourceCheck $pr.Name }
}
function Show-SourceAdd($shot) {
    # 소스 추가 — 적을 것을 적고, 먼저 연결을 시험해 보고, 되는 것만 넣는다 (tools/source_add.py)
    $d = New-Object System.Windows.Forms.Form
    $d.Text = (L "d_src_add")
    $d.Font = $font
    $d.StartPosition = "CenterParent"
    $d.FormBorderStyle = "FixedDialog"
    $d.MaximizeBox = $false
    $d.MinimizeBox = $false
    $d.ClientSize = New-Object System.Drawing.Size(560, 356)
    $rows = @(@((L "l_src_name"), ""), @((L "l_src_host"), ""), @((L "l_src_cam"), "8090"),
              @((L "l_src_path"), "/stream/1"), @((L "l_src_note"), ""))
    $boxes = @()
    $y = 14
    foreach ($row in $rows) {
        [void](New-Label $d $row[0] 14 ($y + 4))
        $b = New-Object System.Windows.Forms.TextBox
        $b.Location = New-Object System.Drawing.Point(110, $y)
        $b.Width = 300
        $b.Text = $row[1]
        $d.Controls.Add($b)
        $boxes += $b
        $y += 30
    }
    $force = New-Check $d (L "l_src_force") 110 ($y + 2) $false
    $hint = New-Label $d ((L "n_src_add") + "  " + (L "n_src_restart")) 14 ($y + 32)
    $hint.AutoSize = $false
    $hint.Size = New-Object System.Drawing.Size(532, 96)
    $out = New-Label $d "" 14 ($y + 132)
    $out.AutoSize = $false
    $out.Size = New-Object System.Drawing.Size(532, 36)
    $test = New-Object System.Windows.Forms.Button
    $test.Text = (L "b_src_test")
    $test.Location = New-Object System.Drawing.Point(110, ($y + 174))
    $test.Width = 110
    $d.Controls.Add($test)
    $save = New-Object System.Windows.Forms.Button
    $save.Text = (L "b_src_save")
    $save.Location = New-Object System.Drawing.Point(230, ($y + 174))
    $save.Width = 96
    $d.Controls.Add($save)
    $close = New-Object System.Windows.Forms.Button
    $close.Text = (L "b_close")
    $close.Location = New-Object System.Drawing.Point(450, ($y + 174))
    $close.Width = 96
    $close.Add_Click({ $d.Close() })
    $d.Controls.Add($close)
    $d.CancelButton = $close
    $argsOf = {
        $a = @($PyWsl, "tools/source_add.py")
        $a += @("--host", $boxes[1].Text.Trim())
        if ($boxes[2].Text.Trim()) { $a += @("--cam-port", $boxes[2].Text.Trim()) }
        if ($boxes[3].Text.Trim()) { $a += @("--path", $boxes[3].Text.Trim()) }
        $a
    }
    $show = {
        param($r)
        if ($null -eq $r) { $out.Text = (L "m_probe_fail"); return }
        $t2 = $(if ($null -ne $r.test) { $r.test } else { $r })
        if ($t2.ok) {
            $out.Text = ((L "m_src_ok") -f $t2.width, $t2.height, $t2.jpeg, $t2.ms)
        } else {
            $out.Text = ((L "m_src_bad") -f $(if ($t2.why) { $t2.why } else { $r.why }))
        }
    }
    $test.Add_Click({
        if (-not $boxes[1].Text.Trim()) { $out.Text = ((L "m_src_bad") -f (L "l_src_host")); return }
        $out.Text = (L "s_src_test")
        $d.Refresh()
        & $show (From-Json (Invoke-Wsl ((& $argsOf) + @("--test", "--json"))))
    })
    $save.Add_Click({
        $nm = $boxes[0].Text.Trim()
        if (-not $nm -or -not $boxes[1].Text.Trim()) { $out.Text = ((L "m_src_bad") -f (L "l_src_name")); return }
        $out.Text = (L "s_src_test")
        $d.Refresh()
        $a = (& $argsOf) + @("--save", "--name", $nm, "--json")
        if ($force.Checked) { $a += "--force" }
        $r = From-Json (Invoke-Wsl $a)
        if ($null -ne $r -and $r.ok) {
            $out.Text = ((L "m_src_saved") -f $r.source.name, $r.source.port)
            New-SourceCheck $r.source.name
            $srcChecks[$r.source.name].Checked = $true
            Load-Sources
        } else {
            & $show $r
        }
    })
    if ($shot) {
        # 확인용 — 창을 모달로 띄우지 않고 그려서 찍는다. DS_SRC_HOST 가 있으면 연결 시험까지 눌러 본다
        $d.Show()
        [System.Windows.Forms.Application]::DoEvents()
        if ($env:DS_SRC_HOST) {
            $boxes[0].Text = "shot-test"
            $boxes[1].Text = $env:DS_SRC_HOST
            $test.PerformClick()
        }
        [System.Windows.Forms.Application]::DoEvents()
        $bmp = New-Object System.Drawing.Bitmap $d.Width, $d.Height
        $d.DrawToBitmap($bmp, (New-Object System.Drawing.Rectangle 0, 0, $d.Width, $d.Height))
        $bmp.Save((Join-Path $shot "src_add.png"), [System.Drawing.Imaging.ImageFormat]::Png)
        $bmp.Dispose()
        $out.Text
        $d.Close()
        return
    }
    [void]$d.ShowDialog($form)
}
function Load-Sources {
    # 지금 닿는 소스만 고를 수 있게 한다 (tools/source_probe.py). 안 닿는 것은 끄고 체크도 푼다.
    # 주소를 모르는 소스(기기가 서버로 밀어 넣는 방식)는 막지 않는다 — 모름은 없음이 아니다
    Set-Status (L "s_probe") $true
    $txt = Invoke-Wsl @($PyWsl, "tools/source_probe.py", "--json")
    Set-Status $(if ($script:Running) { L "s_running" } else { "" })
    $r = From-Json $txt
    if ($null -eq $r) { $srcState.Text = (L "m_probe_fail"); return }
    $script:SrcProbe = @{}
    foreach ($x in @($r.sources)) { $script:SrcProbe[$x.source] = $x }
    $live = @()
    $down = @()
    $script:Building = $true
    foreach ($n in $Sources) {
        $x = $script:SrcProbe[$n]
        $c = $srcChecks[$n]
        if ($null -eq $x) { continue }
        $ok = $x.state -ne "down"
        $c.Enabled = $ok
        if (-not $ok -and $c.Checked) { $c.Checked = $false }
        $srcTip.SetToolTip($c, ("{0}: {1}" -f $x.state, $x.why))
        if ($x.state -eq "live") { $live += $n } elseif (-not $ok) { $down += $n }
    }
    $script:Building = $false
    $script:SrcLive = @($live)
    $script:SrcDown = @($down)
    Set-SrcState
    Build-Match
    Changed
}
function Load-Functions {
    # 기능 목록과 기능마다 쓸 수 있는 모델 — 계약(vision/model_registry.yaml)에서 온다. 기능이 늘면 여기도 는다.
    # 쓸 수 있는 기능을 먼저, 모델이 없어 못 쓰는 것은 아래로 (cli preflight 가 그 까닭까지 준다)
    if ($null -ne $script:FnOpts) { return }
    $txt = Invoke-Wsl @($PyWsl, "-m", "ds_resources.cli", "places", "--json")
    $r = From-Json $txt
    if ($null -eq $r -or $null -eq $r.function_options) { return }
    $script:FnOpts = $r.function_options
    $script:ModelInfo = $r.model_info
    $all = @($r.function_options.PSObject.Properties.Name)
    $pre = Invoke-Wsl @($PyWsl, "-m", "ds_resources.cli", "preflight", "--json")
    $script:FnWhy = @{}
    $pj = From-Json $pre
    foreach ($f in @($pj.features)) { $script:FnWhy[$f.key] = [pscustomobject]@{ usable = $f.usable; why = $f.why } }
    $script:FnList = @($all | Sort-Object @{ Expression = { -not (Fn-Usable $_) } }, @{ Expression = { $_ } })
    if ($script:FnSrc.Count -eq 0 -and @($s.models).Count -gt 0) {
        # 옛 설정(모델 체크)에서 옮겨 온다 — 고르던 모델을 물체 탐지에 얹고 켠 소스 전부에 붙인다
        # 도메인 전용(targeted · river)은 그 기능으로 (2026-11-26: 모델이 아니라 기능)
        foreach ($m in @($s.models)) {
            $fnOf = @($script:FnOpts.PSObject.Properties.Name | Where-Object { @(Fn-Options $_) -contains $m } |
                      Sort-Object @{ Expression = { $_ -ne "object_detect" } }, @{ Expression = { $_ } })[0]
            if ($m -eq "targeted") { $fnOf = "targeted_detect" } elseif ($m -eq "river") { $fnOf = "river_detect" }
            if ($fnOf -and -not $script:FnSrc[$fnOf]) { $script:FnModel[$fnOf] = $m; $script:FnSrc[$fnOf] = @(On-All) }
        }
    }
}
function Load-Places {
    Set-Status (L "s_loading") $true
    $args2 = @($PyWsl, "-m", "ds_resources.cli", "places", "--json") + @((Limit-Args) -split " " | Where-Object { $_ })
    $txt = Invoke-Wsl $args2
    $script:Places = From-Json $txt
    if ($null -eq $script:Places) { $manualNote.Text = (L "m_no_places") }
    Build-Manual
    $script:PlacesKey = (Functions-Args)                 # 이 기능 묶음으로 읽었다는 표시
    Set-Status $(if ($script:Running) { L "s_running" } else { "" })
}

# ================================================================ 아래
$status = New-Object System.Windows.Forms.Label
$status.AutoSize = $false
$status.Location = New-Object System.Drawing.Point(12, 592)
$status.Size = New-Object System.Drawing.Size(288, 40)
$form.Controls.Add($status)
$script:Busy = $false
# 언어 — 화면 글만 바꾼다 (모델 · 근거 · 대상 이름은 카탈로그가 주는 그대로)
$langBox = New-Object System.Windows.Forms.ComboBox
$langBox.DropDownStyle = "DropDownList"
$langBox.Location = New-Object System.Drawing.Point(304, 598)
$langBox.Width = 92
[void]$langBox.Items.Add("한국어")
[void]$langBox.Items.Add("English")
$langBox.SelectedIndex = $(if ($script:Lang -eq "en") { 1 } else { 0 })
$form.Controls.Add($langBox)
# 단계 이동 — 탭을 눌러 바로 가도 되고 여기로 한 단계씩 가도 된다
$prevBtn = New-Object System.Windows.Forms.Button
$prevBtn.Size = New-Object System.Drawing.Size(70, 34)
$prevBtn.Location = New-Object System.Drawing.Point(402, 592)
[void](Reg $prevBtn "b_prev")
$form.Controls.Add($prevBtn)
$nextBtn = New-Object System.Windows.Forms.Button
$nextBtn.Size = New-Object System.Drawing.Size(70, 34)
$nextBtn.Location = New-Object System.Drawing.Point(476, 592)
[void](Reg $nextBtn "b_next")
$form.Controls.Add($nextBtn)
$run = New-Object System.Windows.Forms.Button
[void](Reg $run "b_run")
$run.Size = New-Object System.Drawing.Size(80, 34)
$run.Location = New-Object System.Drawing.Point(550, 592)
$form.Controls.Add($run)
$form.AcceptButton = $run

function Step-Usable($i) {
    # 3단계(지정 클래스)는 그 기능을 골랐을 때만 — 「다음」이 건너뛴다
    if ($i -eq 2) { return ($visOn.Checked -and ((Chosen-Models) -contains "targeted")) }
    return $true
}
function Go-Step($i) {
    if ($i -lt 0 -or $i -ge $tabs.TabPages.Count) { return }
    $tabs.SelectedIndex = $i
    Update-Nav
    # 배치 탭은 「고른 기능」을 보여 준다 — 기능이 바뀐 채로 들어오면 다시 읽는다 (몇 초 걸린다)
    if ($i -eq 4 -and $visOn.Checked -and (Functions-Args) -ne $script:PlacesKey) { Load-Places }
}
function Step-Move($d) {
    $i = $tabs.SelectedIndex + $d
    while ($i -ge 0 -and $i -lt $tabs.TabPages.Count -and -not (Step-Usable $i)) { $i += $d }
    Go-Step $i
}
function Update-Nav {
    $i = $tabs.SelectedIndex
    $prev = $i - 1
    while ($prev -ge 0 -and -not (Step-Usable $prev)) { $prev-- }
    $next = $i + 1
    while ($next -lt $tabs.TabPages.Count -and -not (Step-Usable $next)) { $next++ }
    $prevBtn.Enabled = ($prev -ge 0)
    $nextBtn.Enabled = ($next -lt $tabs.TabPages.Count)
}
function Apply-Language {
    $script:Lang = $(if ($langBox.SelectedIndex -eq 1) { "en" } else { "ko" })
    $form.Text = (L "d_title")
    foreach ($x in $script:Texts) { $x.ctl.Text = (L $x.key) }
    for ($i = 0; $i -lt $StepKeys.Count; $i++) { $tabs.TabPages[$i].Text = (L $StepKeys[$i]) }
    foreach ($f in @($cat.features)) { if ($featChecks[$f.key]) { $featChecks[$f.key].Text = (Feat-Name $f) } }
    Set-SrcState
    Build-CueTable
    Build-Match
    Build-CueBoxes
    Build-Overlay
    Build-Manual
    Sync-Enabled
    Update-Nav
    if (-not $script:Busy) { Set-Status $(if ($script:Running) { L "s_running" } else { "" }) }
}

function Sync-Enabled {
    $tOn = $visOn.Checked -and ((Chosen-Models) -contains "targeted")
    foreach ($c in $tTarget.Controls) { $c.Enabled = $tOn }
    foreach ($k in $featChecks.Keys) { $featChecks[$k].Enabled = $visOn.Checked -and [bool](@($cat.features) | Where-Object { $_.key -eq $k -and $_.available }) }
    $overPanel.Enabled = $visOn.Checked
    $orchOn.Enabled = $visOn.Checked
    $planBtn.Enabled = $visOn.Checked
    $gCues.Enabled = $visOn.Checked
    $gMatch.Enabled = $visOn.Checked
    $autoRadio.Enabled = $visOn.Checked -and $orchOn.Checked
    $manualOn.Enabled = $visOn.Checked -and $orchOn.Checked
    $manualLoad.Enabled = $visOn.Checked -and $orchOn.Checked
    $manualPanel.Enabled = $visOn.Checked -and $orchOn.Checked -and $manualOn.Checked
    $sliceU.Enabled = $sliceOn.Checked
    $tabs.TabPages[2].Enabled = (Step-Usable 2)        # 지정 클래스는 그 기능을 골랐을 때만
}

# ---- 구성 · 프로필
function Current-Config {
    $on = @($Sources | Where-Object { $srcChecks[$_].Checked })
    $onAll = @($on) + @(if ($sliceOn.Checked) { "cam360_slice" })
    return [pscustomobject]@{
        server = (($on -join " ") + "|" + $cam.SelectedItem + "|" + $sliceOn.Checked)
        slice  = ($sliceOn.Checked.ToString() + "|" + $sliceU.Text.Trim())
        vision = ($visOn.Checked.ToString() + "|" + ((Chosen-Models) -join " ") + "|" + ($onAll -join " ") + "|" + (Vision-Groups-Key) + "|" + $orchOn.Checked + "|" + (Match-Key) + "|" + (Pin-Key))
        on = $on; onAll = $onAll
    }
}
# 매칭 · 수동 배치 열쇠 — 바뀌면 묶음(프로세스)과 자리가 바뀌니 비전을 다시 띄운다
function Match-Key {
    $m = Assign-Map
    return (($m.Keys | Sort-Object | ForEach-Object { "{0}={1}" -f $_, (($m[$_]) -join ",") }) -join ";")
}
function Pin-Key {
    if (-not $manualOn.Checked) { return "" }
    return (($script:ManualPlace.Keys | Sort-Object | ForEach-Object { "{0}={1}" -f $_, $script:ManualPlace[$_] }) -join ";")
}
function Vision-Groups-Key { if ((Chosen-Models) -contains "targeted") { return ((Checked-Targets) -join ",") + ($distOn.Checked) } return "" }

function Build-Profile {
    $pm = @{}
    $tm = @{}
    if (((Chosen-Models) -contains "targeted") -and $script:TargetsTouched) {
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
                       assign = $script:Assign; manual = $manualOn.Checked; manualPlace = $script:ManualPlace;
                       nodeLimit = $script:NodeLimit; fnModel = $script:FnModel; fnSrc = $script:FnSrc;
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

# $script:Busy 는 「띄우는 중 · 끄는 중」 같은 진행 글이 떠 있다는 표시다 — 비전 보고 폴링이 그 글을 덮지 않게.
# 예전에는 글이 「중」으로 끝나는지 봤는데, 영어로 바꾸면 그 규칙이 깨진다 (2026-11-26)
function Set-Status($text, $busy = $false) {
    $script:Busy = [bool]$busy
    $status.Text = $text
    [System.Windows.Forms.Application]::DoEvents()
}

function Check-Inputs($cfg) {
    if ($cfg.on.Count -eq 0) { return (L "m_need_source") }
    if ($visOn.Checked -and (Chosen-Functions).Count -eq 0) { return (L "m_need_model") }
    if ($visOn.Checked -and ((Chosen-Models) -contains "targeted") -and (Checked-Targets).Count -eq 0) { return (L "m_need_target") }
    if ($sliceOn.Checked -and ($cfg.on -notcontains "cam360")) { return (L "m_need_cam360") }
    if ($visOn.Checked -and -not $orchOn.Checked -and (Assign-Map).Count -gt 0) {
        return (L "m_match_orch")
    }
    if ($visOn.Checked -and $manualOn.Checked -and -not $orchOn.Checked) { return (L "m_manual_orch") }
    if ($visOn.Checked -and (Chosen-Models).Count -gt 0) {
        $live = @((On-All) | Where-Object { (Assign-For $_).Count -gt 0 })
        if ((On-All).Count -gt 0 -and $live.Count -eq 0) { return (L "m_match_empty") }
    }
    $u = $sliceU.Text.Trim()
    if ($sliceOn.Checked -and $u -ne "") {
        $num = [double]0
        if (-not [double]::TryParse($u, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$num)) { return (L "m_slice_u") }
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
function Assign-Args {
    $m = Assign-Map
    if ($m.Count -eq 0) { return "" }
    $parts = @($m.Keys | Sort-Object | ForEach-Object { "{0}={1}" -f $_, (($m[$_]) -join ",") })
    return " --assign " + ($parts -join " ")
}
function Pin-Args {
    if (-not $manualOn.Checked) { return "" }
    $parts = @($script:ManualPlace.Keys | Sort-Object | Where-Object { $script:ManualPlace[$_] } |
               ForEach-Object { "{0}={1}" -f $_, $script:ManualPlace[$_] })
    if ($parts.Count -eq 0) { return "" }
    return " --pin " + ($parts -join " ")
}
function Orch-Args($cfg, $profWsl = $ProfileWsl) {
    return ("--sources {0} --models {1} --profile {2}" -f ($cfg.onAll -join " "), ((Chosen-Models) -join " "), $profWsl) +
           (Functions-Args) + (Assign-Args) + (Pin-Args) + (Limit-Args)
}
function Show-Plan {
    $cfg = Current-Config
    if ($cfg.onAll.Count -eq 0 -or (Chosen-Models).Count -eq 0) { [void][System.Windows.Forms.MessageBox]::Show((L "m_pick_both"), (L "b_plan")); return }
    Write-PreviewProfile
    Set-Status (L "s_plan_calc") $true
    $argv = @("./venv/bin/python", "-m", "ds_resources.cli", "plan") + ((Orch-Args $cfg $PreviewWsl) -split " " | Where-Object { $_ })
    $txt = Invoke-Wsl $argv
    Set-Status $(if ($script:Running) { L "s_running" } else { "" })
    if (-not $txt) { $txt = (L "m_no_plan") }
    [void][System.Windows.Forms.MessageBox]::Show($txt, (L "b_plan"))
}
function Start-Vision($cfg) {
    $mods = Chosen-Models
    if ($orchOn.Checked) {
        $env = "HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
        Start-WslWindow "ds_vision_orch" "인지 오케스트레이터" ("./venv/bin/python -m ds_resources.cli run " + (Orch-Args $cfg)) $env
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
        $title = if ($groups -gt 1) { "인지 추론 ({0})" -f ($b -join " ") } else { "인지 추론" }
        Start-WslWindow ("ds_vision_{0}" -f $k) $title ("{0} vision/vision_infer.py --devices {1} --models {2} --profile {3}" -f $PyWsl, ($b -join " "), ($mods -join " "), $ProfileWsl) $env
        if ($groups -gt 1) { Start-Sleep -Seconds 3 }
    }
}
function Vision-Names { return @(Running-Names | Where-Object { $_ -like "ds_vision_*" }) }
function Mine-Names {
    return @(Running-Names | Where-Object { $_ -eq "ds_server" -or $_ -eq "ds_slice" -or $_ -like "ds_vision_*" })
}

function Start-All {
    $cfg = Current-Config
    $why = Check-Inputs $cfg
    if ($why) { [void][System.Windows.Forms.MessageBox]::Show($why, (L "d_title")); return }
    Save-Settings
    Write-Profile
    Remove-Item -Path (Join-Path $Here "status_*.json") -ErrorAction SilentlyContinue   # 지난 실행의 반영 결과
    Set-Status (L "s_starting") $true
    Start-Server $cfg
    $waited = 0
    if ($sliceOn.Checked) { Start-Sleep -Seconds $SliceDelay; $waited = $SliceDelay; Start-Slice }
    if ($visOn.Checked) { Start-Sleep -Seconds ([Math]::Max(0, $VisionDelay - $waited)); Start-Vision $cfg }
    $script:Applied = $cfg
    $script:Running = $true
    $run.Visible = $false
    Set-Status (L "s_running")
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
        Set-Status (L "s_restarting_server") $true
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
                Set-Status (L "s_restarting_vision") $true
                Stop-Windows (Vision-Names)
                if ($visOn.Checked) { Start-Vision $cfg }
            }
        }
    }
    $script:Applied = $cfg
    Set-Status (L "s_running")
}

# ---- 비전이 남긴 반영 결과(틀린 프로필이면 이유)를 상태 줄에
# 오버레이 탭 그림을 다시 그리는 타이머 — 연달아 고치면 마지막 것만 그린다
$script:OverShot = New-Object System.Windows.Forms.Timer
$script:OverShot.Interval = 1200
$script:OverShot.Add_Tick({ $script:OverShot.Stop(); Run-OverShot })
$poll = New-Object System.Windows.Forms.Timer
$poll.Interval = 2000
$poll.Add_Tick({
    if (-not $script:Running) { return }
    $f = Get-ChildItem -Path $Here -Filter "status_*.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $f) { return }
    try { $j = Get-Content $f.FullName -Raw -Encoding UTF8 | ConvertFrom-Json } catch { return }
    # 「…중」 같은 진행 글이 떠 있는 동안에는 덮지 않는다 (언어에 기대지 않게 깃발로 본다)
    if ($j.ok) { if (-not $script:Busy) { $status.Text = (L "s_running") } }
    else { $status.Text = (L "s_running") + ", " + (L "s_not_applied") + ": " + $j.message }
})
$poll.Start()

# ---- 연결
foreach ($c in $srcChecks.Values) { $c.Add_CheckedChanged({ Build-Match; Build-CueTable; Build-Manual; Update-Nav; Changed }) }
$tabs.Add_SelectedIndexChanged({
    Update-Nav
    if ($tabs.SelectedIndex -eq 4 -and $visOn.Checked -and (Functions-Args) -ne $script:PlacesKey) { Load-Places }
})
$srcCheck.Add_Click({ Load-Sources })
$srcAdd.Add_Click({ Show-SourceAdd })
$sliceBtn.Add_Click({ Show-Slice })
$addFeat.Add_Click({ Show-Features })
$previewBtn.Add_Click({ Show-Preview })
$overPreview.Add_Click({ Show-Preview })
$prevBtn.Add_Click({ Step-Move -1 })
$nextBtn.Add_Click({ Step-Move 1 })
$langBox.Add_SelectionChangeCommitted({ Apply-Language; Changed })
$cam.Add_SelectedIndexChanged({ Changed })
$sliceOn.Add_CheckedChanged({ Build-Match; Changed })
$sliceU.Add_Leave({ Changed })
$visOn.Add_CheckedChanged({ Build-Overlay; Build-CueTable; Build-Match; Update-Nav; Changed })
foreach ($c in $featChecks.Values) { $c.Add_CheckedChanged({ Changed }) }
$run.Add_Click({ Start-All })
$orchOn.Add_CheckedChanged({ Build-Manual; Sync-Enabled; Changed })
# 한 단어로 줄인 것들의 뜻은 풍선말에 둔다 (화면 글은 짧게, 뜻은 잃지 않게 — 2026-11-26)
$manualTip.SetToolTip($orchOn, (L "t_orch"))
$manualTip.SetToolTip($autoRadio, (L "t_auto"))
$manualTip.SetToolTip($manualOn, (L "t_manual"))
$manualTip.SetToolTip($planBtn, (L "t_plan"))
$manualTip.SetToolTip($weightsBtn, (L "t_weights"))
$weightsBtn.Add_Click({ Show-Weights })
$planBtn.Add_Click({ Show-Plan })
$matchReset.Add_Click({ $script:FnSrc = @{}; $script:FnModel = @{}; Build-Match; Build-CueTable; Build-Overlay; Sync-Enabled; Changed })
$manualLoad.Add_Click({ Load-Places })
$manualOn.Add_CheckedChanged({
    if ($manualOn.Checked -and $null -eq $script:Places) { Load-Places } else { Build-Manual }
    Sync-Enabled
    Changed
})
$autoRadio.Add_CheckedChanged({ if ($autoRadio.Checked) { Build-Manual; Sync-Enabled; Changed } })
$form.Add_FormClosing({
    param($sender, $e)
    if (-not $script:Running) { return }
    $r = [System.Windows.Forms.MessageBox]::Show((L "m_stop"), (L "d_title"),
                                                [System.Windows.Forms.MessageBoxButtons]::OKCancel)
    if ($r -ne [System.Windows.Forms.DialogResult]::OK) { $e.Cancel = $true; return }
    Save-Settings
    Set-Status (L "s_stopping") $true
    $poll.Stop(); $debounce.Stop()
    Stop-Windows (@(Vision-Names) + @("ds_slice", "ds_server"))
})

Move-SrcFoot
Load-UserSources
Load-Sources
Load-Functions
Build-TargetChecks
Build-CueBoxes
Build-Overlay
Build-CueTable
Build-Match
Build-Manual
Sync-Enabled
Update-Nav

# ---- 이어받기: 지난번 런처가 띄운 창이 아직 돌고 있으면 '실행 중' 으로
$alive = Mine-Names
if ($alive.Count -gt 0) {
    $script:Running = $true
    $script:Applied = Current-Config
    $run.Visible = $false
    $status.Text = (L "s_running")
}

# ---- 자체 시험 (창을 띄우지 않는다): $env:DS_LAUNCHER_SELFTEST 가 있으면 선택을 코드로 바꿔 보고 프로필을 찍는다
function Find-Ctl($panel, $pred) {
    foreach ($g in $panel.Controls) { foreach ($c in $g.Controls) { if (& $pred $c) { return $c } } }
    return $null
}
function Pick-Function($fn, $model, $sources) {
    # 시험 · 기본값 채우기용 — 기능 하나를 고르고 그 모델과 소스를 정한다
    $script:FnModel[$fn] = $model
    $script:FnSrc[$fn] = @($sources)
    Build-Match
    Build-CueTable
    Build-Overlay
    Sync-Enabled
}
function Self-Test {
    $r = [ordered]@{}
    # 시험 중에는 「실행 중 반영」이 돌면 안 된다 — 돌면 사람이 고른 settings.json 을 시험 상태로 덮어쓰고
    # 창까지 다시 띄운다 (2026-11-26 실측: 자체 시험 중에 Apply-Changes 가 돌아 settings.json 이 바뀌었다)
    $script:Running = $false
    $debounce.Stop()
    $poll.Stop()
    $script:OverShot.Stop()
    # 지난 실행이 남긴 선택에 기대지 않는다 — 기기별 기능 · 수동 자리는 시험이 직접 채운다
    $script:Assign = @{}
    $script:ManualPlace = @{}
    $manualOn.Checked = $false
    $r.models = $Models
    $r.initial_targets = @(Checked-Targets)
    $r.profile_untouched = Build-Profile
    $r.orchestrate = $orchOn.Checked
    $r.orch_args = Orch-Args (Current-Config)
    $visOn.Checked = $true
    Pick-Function "object_detect" "yoloe" @(@(On-All)[0])
    Pick-Function "targeted_detect" "targeted" @(@(On-All)[0])
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
    Pick-Function "targeted_detect" "targeted" @()
    $r.target_tab_enabled_after_off = $tTarget.Controls[0].Enabled
    # 「배치 보기」 는 profile.json · 프로필 버전에 손대지 않는다 (M11 R4)
    $snap = { if (Test-Path $ProfileFile) { [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($ProfileFile)) } else { "missing" } }
    $before, $verBefore = (& $snap), $script:ProfileVer
    Write-PreviewProfile
    $r.preview_keeps_profile = ((& $snap) -eq $before)
    $r.preview_keeps_version = ($script:ProfileVer -eq $verBefore)
    $r.preview_written = (Test-Path $PreviewFile)
    $r.plan_args = Orch-Args (Current-Config) $PreviewWsl
    # ---- 모델 매칭 · 수동 배치 (2026-11-26)
    foreach ($n in @("pi13", "pi11")) { $srcChecks[$n].Checked = $true }
    Pick-Function "object_detect" "yoloe" @("pi13", "pi11")
    Build-Match
    $script:Assign = @{ "pi13" = @("yoloe"); "pi11" = @() }      # pi13 은 yoloe 만 · pi11 은 비전 없음
    $r.assign_args = Assign-Args
    $r.assign_rows = @((Assign-Map).Keys | Sort-Object)
    $r.assign_pi11_empty = ((Assign-For "pi11").Count -eq 0)
    $script:Assign = @{}
    $r.assign_args_default = Assign-Args                          # 기본값(전부)이면 인자가 없다
    $script:ManualPlace = @{ "object_detect.device" = "pi1"; "video_preprocess.server" = "none" }
    $manualOn.Checked = $true
    $r.pin_args = Pin-Args
    $r.vision_key_sees_pins = ((Current-Config).vision -like "*object_detect.device=pi1*")
    $manualOn.Checked = $false
    $r.pin_args_when_auto = Pin-Args                              # 수동을 끄면 자리를 넘기지 않는다
    $r.step_count = $tabs.TabPages.Count
    # 화면 밖으로 나간 컨트롤이 없나 (단계마다 묶음 · 패널이 탭 안에 들어가는지)
    # 창을 띄우지 않으므로 ClientSize 는 아직 없다 — 탭 크기에서 테두리 · 탭 띠를 뺀 값으로 본다
    $innerW = $tabs.Width - 8
    $innerH = $tabs.Height - 28
    $over = @()
    foreach ($pg in $tabs.TabPages) {
        foreach ($c in $pg.Controls) {
            if ($c.Right -gt $innerW -or $c.Bottom -gt $innerH) {
                $over += ("{0}/{1} {2}x{3}" -f $pg.Text, $c.GetType().Name, $c.Right, $c.Bottom)
            }
        }
    }
    $r.off_screen = $over
    $r.src_state = $srcState.Text
    $r.match_rows = @($matchPanel.Controls | Where-Object { $_ -is [System.Windows.Forms.Label] } |
                      ForEach-Object { $_.Text }) -join " | "
    # ---- 소스 추가로 들어온 소스가 소스 줄과 기능 격자에 바로 보이는가 (화면 쪽 길만 — 파일에는 쓰지 않는다)
    New-SourceCheck "pi14"
    $r.added_source = (@($Sources) -contains "pi14")
    $srcChecks["pi14"].Checked = $true
    Build-Match
    $r.added_in_grid = (@($matchPanel.Controls | Where-Object { $_ -is [System.Windows.Forms.Label] } |
                          ForEach-Object { $_.Text }) -contains "pi14")
    $srcChecks["pi14"].Checked = $false
    # ---- 단계 이동 · 한/영 (2026-11-26)
    $r.steps_ko = @(0..($tabs.TabPages.Count - 1) | ForEach-Object { $tabs.TabPages[$_].Text })
    Pick-Function "targeted_detect" "targeted" @()
    Go-Step 1
    Step-Move 1
    $r.next_skips_targets = $tabs.SelectedIndex            # 지정 클래스를 안 골랐으면 3단계를 건너뛴다 (2 → 3)
    Pick-Function "targeted_detect" "targeted" @(@(On-All)[0])
    Go-Step 1
    Step-Move 1
    $r.next_into_targets = $tabs.SelectedIndex             # 골랐으면 3단계(인덱스 2)로
    $r.cue_rows_ko = @($cueTable.Controls | ForEach-Object { $_.Text }) -join " | "
    $langBox.SelectedIndex = 1
    Apply-Language
    $r.lang = $script:Lang
    $r.steps_en = @(0..($tabs.TabPages.Count - 1) | ForEach-Object { $tabs.TabPages[$_].Text })
    $r.run_en = $run.Text
    $r.form_en = $form.Text
    $r.place_en = @($autoRadio.Text, $manualOn.Text)
    $r.cue_rows_en = @($cueTable.Controls | ForEach-Object { $_.Text }) -join " | "
    $r.check_inputs_en = (Check-Inputs ([pscustomobject]@{ on = @(); onAll = @() }))
    $langBox.SelectedIndex = 0
    Apply-Language
    $r.back_to_ko = $run.Text
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
    Pick-Function "targeted_detect" "targeted" @(@(On-All)[0])
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
    Pick-Function "object_detect" "yoloe" @(@(On-All)[0])
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
# ---- 화면 찍기 (확인용): $env:DS_LAUNCHER_SHOT 에 폴더를 주면 단계마다 창을 그림으로 저장하고 끝낸다.
# 한국어 · 영어 두 벌을 찍는다. 창을 잠깐 띄웠다 바로 닫는다(사람이 쓰는 것을 건드리지 않게)
function Shot-Test($dir) {
    # 시험 중에는 「실행 중 반영」이 돌면 안 된다 — 돌면 사람이 고른 settings.json 을 시험 상태로 덮어쓰고
    # 창까지 다시 띄운다 (2026-11-26 실측: 자체 시험 중에 Apply-Changes 가 돌아 settings.json 이 바뀌었다)
    $script:Running = $false
    $debounce.Stop()
    $poll.Stop()
    $script:OverShot.Stop()
    $form.Show()
    [System.Windows.Forms.Application]::DoEvents()
    $manualOn.Checked = $true          # 5단계 표가 보이게 (확인용)
    Load-Places
    if ($env:DS_SHOT_IMAGE) {
        # 오버레이 탭 아래 그림까지 확인한다 — 기능 하나를 켜고 저장된 장에 실제로 그려 본다
        $visOn.Checked = $true
        Pick-Function "object_detect" "yoloe" @("pi12")
        $script:PreviewPath = $env:DS_SHOT_IMAGE
        Build-Overlay
        Run-OverShot
        [System.Windows.Forms.Application]::DoEvents()
    }
    foreach ($lang in @(0, 1)) {
        $langBox.SelectedIndex = $lang
        Apply-Language
        [System.Windows.Forms.Application]::DoEvents()
        for ($i = 0; $i -lt $tabs.TabPages.Count; $i++) {
            $tabs.SelectedIndex = $i
            Update-Nav
            [System.Windows.Forms.Application]::DoEvents()
            Start-Sleep -Milliseconds 250
            [System.Windows.Forms.Application]::DoEvents()
            $bmp = New-Object System.Drawing.Bitmap $form.Width, $form.Height
            $form.DrawToBitmap($bmp, (New-Object System.Drawing.Rectangle 0, 0, $form.Width, $form.Height))
            $name = "step{0}_{1}.png" -f ($i + 1), $(if ($lang -eq 1) { "en" } else { "ko" })
            $bmp.Save((Join-Path $dir $name), [System.Drawing.Imaging.ImageFormat]::Png)
            $bmp.Dispose()
            $name
        }
    }
    ("src_add: " + (Show-SourceAdd $dir))
    $form.Close()
}
if ($env:DS_LAUNCHER_PREVIEW) {
    # 창 없이 미리보기 경로만 확인한다 — 런처가 부르는 명령 · JSON 읽기 · 그림 자리까지 그대로
    $script:Running = $false
    $debounce.Stop(); $poll.Stop()
    $img = $env:DS_LAUNCHER_PREVIEW
    Write-PreviewProfile
    $probe = $(if (Test-Path (Wsl-Win "/tools/preview_apply.py")) { "tool" } else { "" })
    $argv = @($PyWsl, "tools/preview_apply.py", "--json", "--image", (To-WslPath $img),
              "--profile", $PreviewWsl, "--out", "/tmp/ds_preview", "--models") + @(Chosen-Models)
    $txt = Invoke-Wsl $argv
    $r = From-Json $txt
    $win = ""
    if ($null -ne $r -and $r.overlay) { $win = "\\wsl.localhost\" + $Distro + ([string]$r.overlay -replace "/", "\") }
    [pscustomobject]@{ probe = $probe; models = @(Chosen-Models); argv = ($argv -join " ")
                       ok = $(if ($null -eq $r) { $null } else { $r.ok })
                       summary = $(if ($null -eq $r) { $txt } else { [string]$r.summary })
                       overlay = $(if ($null -eq $r) { "" } else { [string]$r.overlay })
                       win_path = $win
                       win_readable = $(if ($win -and (Test-Path $win)) { $true } else { $false })
    } | ConvertTo-Json -Depth 5
    exit 0
}
if ($env:DS_LAUNCHER_SHOT) { Shot-Test $env:DS_LAUNCHER_SHOT; exit 0 }
if ($env:DS_LAUNCHER_SELFTEST) { Self-Test; exit 0 }
if ($env:DS_LAUNCHER_E2E) { E2E-Test; exit 0 }
[void]$form.ShowDialog()
