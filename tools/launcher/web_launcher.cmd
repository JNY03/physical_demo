@echo off
chcp 65001 >nul
rem 데이터 스트림 웹 런처 바로가기 (2026-11-25, Docker 계획 D7) - 브라우저만 연다. 실행 로직은 없다.
rem 웹 런처는 WSL/컨테이너 안에서 돈다: python -m ds_resources.weblauncher (기본 127.0.0.1:10080)
rem 주소: 첫 인자 -> 환경변수 DS_WEB_LAUNCHER_URL -> http://localhost:10080/
setlocal
set "URL=%~1"
if "%URL%"=="" set "URL=%DS_WEB_LAUNCHER_URL%"
if "%URL%"=="" set "URL=http://localhost:10080/"
start "" "%URL%"
endlocal
