# 런처 부품 (tools/launcher/parts)

`launcher.ps1` 이 이 폴더의 `*.ps1` 을 이름순으로 dot-source 한다 (창을 만들기 전, `$Str` · `$FnName` 정의 뒤).

규칙
- 부품은 **함수와 화면 글만** 정의한다. 불러올 때 창 컨트롤을 만들지 않는다.
- 화면 글은 `$Str["키"] = @{ ko = "..."; en = "..." }` 로 더한다 (본문 키와 겹치지 않게 부품 이름을 앞에 붙인다).
- 본문이 부르는 자리: `Part-Hook "<이름>" @(인자…)` → 부품의 `Hook-<이름>` 함수. 없으면 아무 일도 없다.
- 자체 시험(`DS_LAUNCHER_SELFTEST=1`)은 `PartTest-*` 함수의 결과를 JSON 에 싣는다.
- 설치(`install_desktop.ps1`)가 이 폴더를 `%USERPROFILE%\DataStreamLauncher\parts\` 로 복사한다.
- 파일은 UTF-8 BOM 으로 저장한다 (Windows PowerShell 5.1 이 한글을 읽게).
