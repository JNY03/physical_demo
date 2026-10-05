/**********************************************************************
 피지컬팀 mk2 — Go1 얼굴 라이트 브리지 (Go1 헤드 Nano 192.168.123.13 상주)
 ---------------------------------------------------------------------
 왜 필요한가:
   HighCmd.led 는 Go1 에서 동작하지 않는다(SDK 헤더에도 "reserve", 실측으로도
   빨강/초록/파랑 어느 것도 반응 없음). Go1 의 얼굴 LED 는 헤드 Nano 에서 도는
   faceLightServer(UDP 192.168.123.13:7800)가 잡고 있고, 그 패킷은 닫힌 바이너리
   libfaceLight_SDK_arm64.so 안에서 만들어진다. 그래서 라이트를 쓰려면 "그 .so 를
   부를 수 있는 자리" = 헤드 Nano 에 이 브리지를 두고, 밖(파이)에서는 텍스트로 색만
   보내는 게 가장 단순하고 깨지지 않는다.

 프로토콜(이 브리지가 계약이다):
   UDP <port, 기본 7801> 로 "R G B" (각 0~255, 공백 구분) 한 줄 -> 얼굴 LED 전체 색.
   예) echo -n "0 0 255" | nc -u -w1 192.168.123.13 7801     # 파랑
       echo -n "0 0 0"   | nc -u -w1 192.168.123.13 7801     # 끄기

 빌드(헤드 Nano 에서):
   SDK=~/Unitree/sdk/faceLightSDK_Nano
   g++ -O2 -std=c++14 -I $SDK/include go1_face_light_bridge.cpp \
       -L $SDK/lib -lfaceLight_SDK_arm64 -Wl,-rpath,$SDK/lib -o face_light_bridge
 실행:
   nohup ./face_light_bridge 7801 > ~/face_light_bridge.log 2>&1 &
***********************************************************************/

#include "FaceLightClient.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

int main(int argc, char** argv)
{
  int port = (argc > 1) ? std::atoi(argv[1]) : 7801;

  int s = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (s < 0) { std::perror("socket"); return 1; }

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);
  if (::bind(s, (sockaddr*)&addr, sizeof(addr)) < 0) { std::perror("bind"); return 1; }

  FaceLightClient client;
  std::printf("[faceLightBridge] udp/%d 대기 — 페이로드 \"R G B\"\n", port);
  std::fflush(stdout);

  char buf[128];
  while (true)
  {
    sockaddr_in from; socklen_t fromlen = sizeof(from);
    int n = ::recvfrom(s, buf, sizeof(buf) - 1, 0, (sockaddr*)&from, &fromlen);
    if (n <= 0) continue;
    buf[n] = '\0';

    int r = 0, g = 0, b = 0;
    if (std::sscanf(buf, "%d %d %d", &r, &g, &b) != 3) continue;
    if (r < 0) r = 0; if (r > 255) r = 255;
    if (g < 0) g = 0; if (g > 255) g = 255;
    if (b < 0) b = 0; if (b > 255) b = 255;

    const uint8_t rgb[3] = { (uint8_t)r, (uint8_t)g, (uint8_t)b };
    client.setAllLed(rgb);
    client.sendCmd();

    std::printf("[faceLightBridge] %s -> rgb(%d,%d,%d)\n",
                inet_ntoa(from.sin_addr), r, g, b);
    std::fflush(stdout);
  }
  return 0;
}
