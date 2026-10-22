# -*- coding: utf-8 -*-
"""
Unity 경로계획 -> UDP 송신 -> go1_sdk_pc.cpp 경로추종/DR -> 상태 송신 -> Unity 표시
전 구간을 수정 후 코드 그대로 재현해서, 화면의 가상 GO1 이 계획 경로와 같은 곳을
같은 각도로 가는지 확인한다.
"""
import math
D=math.degrees; R=math.radians
def wrap(a):
    while a> math.pi: a-=2*math.pi
    while a<-math.pi: a+=2*math.pi
    return a
def ndeg(d):
    while d> 180: d-=360
    while d<-180: d+=360
    return d

# ── Unity: 월드 -> 로컬 (yaw φ, 시계+, 전진=(sinφ,cosφ)) ──────────────────
def unity_world_to_local(vx,vz,phi):
    return (vx*math.cos(phi)-vz*math.sin(phi), vx*math.sin(phi)+vz*math.cos(phi))

# GO1CoordinateMapper (수정 후: 전부 항등)
def path_local(ulx,ulz):            return (ulx, ulz)      # swap/invert 전부 off
def path_dir_to_yaw_deg(dx,dz):     return ndeg(D(math.atan2(dx,dz)))  # invertYawSign off
def sdk_yaw_to_unity_deg(sdk_rad):  return ndeg(D(sdk_rad))            # invertSdkYaw off
def body_delta_to_unity(dx,dz,sdk_yaw):     # SdkStateWorldDeltaToUnityBodyDelta (부호 수정본)
    F=(math.sin(sdk_yaw), math.cos(sdk_yaw)); Rt=(math.cos(sdk_yaw),-math.sin(sdk_yaw))
    fwd = dx*F[0]+dz*F[1]
    lat = dx*Rt[0]+dz*Rt[1]
    uy  = R(sdk_yaw_to_unity_deg(sdk_yaw))
    uF=(math.sin(uy),math.cos(uy)); uR=(math.cos(uy),-math.sin(uy))
    return (uF[0]*fwd+uR[0]*lat, uF[1]*fwd+uR[1]*lat)

# ── C++ go1_sdk_pc.cpp (수정본) ───────────────────────────────────────────
class Sdk:
    def __init__(self, yaw_rel0):
        self.yaw_rel=yaw_rel0; self.off=0.0
        self.wx=0.0; self.wz=0.0; self.wps=[]; self.anchor=0.0
    @property
    def yaw_unity(self): return wrap(-self.yaw_rel+self.off)
    def on_path(self, start_yaw_deg, pts):
        s=wrap(R(start_yaw_deg))
        self.off=wrap(s+self.yaw_rel)
        self.wx=0.0; self.wz=0.0
        corr=wrap(self.yaw_unity-s)
        self.anchor=wrap(self.yaw_unity-corr)
        A=self.anchor; p0=pts[0]; self.wps=[]
        for (px,pz) in pts[1:]:
            lx,lz=px-p0[0], pz-p0[1]
            self.wps.append((self.wx+lx*math.cos(A)+lz*math.sin(A),
                             self.wz-lx*math.sin(A)+lz*math.cos(A)))
        return corr
    def step(self,vx,vy,wz,dt):
        y=self.yaw_unity
        self.wx+=(vx*math.sin(y)-vy*math.cos(y))*dt
        self.wz+=(vx*math.cos(y)+vy*math.sin(y))*dt
        self.yaw_rel=wrap(self.yaw_rel+wz*dt)   # IMU 는 반시계+

def follow(sdk,dt=0.02,tmax=400.0):
    i=0;t=0.0;traj=[]
    while i<len(sdk.wps) and t<tmax:
        ex,ez=sdk.wps[i]
        dx,dz=ex-sdk.wx, ez-sdk.wz
        if math.hypot(dx,dz)<=0.10: i+=1; continue
        yaw_err=wrap(math.atan2(dx,dz)-sdk.yaw_unity)
        wz=max(-0.45,min(0.45,-2.0*yaw_err))
        vx=0.0 if abs(yaw_err)>0.20 else min(0.07,0.35*math.hypot(dx,dz))
        sdk.step(vx,0.0,wz,dt); t+=dt
        traj.append((sdk.wx,sdk.wz,sdk.yaw_unity))
    return traj, i>=len(sdk.wps)

# ── 시나리오 ──────────────────────────────────────────────────────────────
PATHS={
 "직진 1m"      :[(0,0),(0,0.5),(0,1.0)],
 "우회전 ㄱ자"  :[(0,0),(0,0.8),(0.6,0.8)],
 "좌회전 ㄴ자"  :[(0,0),(0,0.8),(-0.6,0.8)],
 "S자"          :[(0,0),(0,0.5),(0.4,0.9),(0.4,1.4),(0.0,1.8)],
}
print("="*78)
print("계획 경로(Unity 월드)  vs  Unity 화면의 가상 GO1 이 실제로 지나는 점")
print("="*78)
worst=0.0
for name,local_pts in PATHS.items():
    for phi0_deg in (0.0,37.0,90.0,-120.0,180.0):
        phi0=R(phi0_deg); P0=(2.5,-1.0)   # 가상 GO1 의 현재 월드 위치
        # Unity: 계획 경로를 월드에 배치
        world=[(P0[0]+lx*math.cos(phi0)+lz*math.sin(phi0),
                P0[1]-lx*math.sin(phi0)+lz*math.cos(phi0)) for (lx,lz) in local_pts]
        # Unity 송신: 월드 -> 로컬 -> path-local
        pts=[path_local(*unity_world_to_local(w[0]-P0[0],w[1]-P0[1],phi0)) for w in world]
        # C++ 수신
        sdk=Sdk(yaw_rel0=R(17.0))       # 로봇은 아무 방향에나 놓여 있음
        sdk.on_path(phi0_deg,pts)
        # C++ 이 만든 월드 waypoint 를 Unity 표시 좌표로 되돌린다(항등 + 시작점 재기준)
        shown=[(P0[0]+wx,P0[1]+wz) for (wx,wz) in sdk.wps]
        err=max(math.hypot(a[0]-b[0],a[1]-b[1]) for a,b in zip(shown,world[1:]))
        worst=max(worst,err)
        traj,done=follow(sdk)
        end=(P0[0]+traj[-1][0], P0[1]+traj[-1][1]) if traj else P0
        end_err=math.hypot(end[0]-world[-1][0], end[1]-world[-1][1])
        disp_yaw=sdk_yaw_to_unity_deg(traj[-1][2]) if traj else phi0_deg
        print(f"{name:<10} start={phi0_deg:7.1f}deg | waypoint 오차 {err:.4f} m | "
              f"도착 오차 {end_err:.3f} m | 완주={'예' if done else '아니오'} | "
              f"표시 yaw={disp_yaw:7.1f}deg")
print("-"*78)
print(f"waypoint 최대 오차 = {worst:.2e} m  ->  {'PASS' if worst<1e-9 else 'FAIL'}")

print()
print("="*78)
print("각도: 로봇이 실제로 향한 방향과 Unity 표시각이 같은가")
print("="*78)
for phi0_deg in (0.0,37.0,90.0,-120.0):
    sdk=Sdk(yaw_rel0=R(17.0))
    sdk.on_path(phi0_deg,[(0,0),(0,1.0)])
    d0=sdk_yaw_to_unity_deg(sdk.yaw_unity)
    sdk.step(0,0,+0.5,1.0)                    # 실로봇 좌회전 28.6deg
    d1=sdk_yaw_to_unity_deg(sdk.yaw_unity)
    # Unity 는 시계+ 이므로 좌회전이면 -28.6deg 여야 한다
    ok1 = abs(ndeg(d0-phi0_deg))<1e-6
    ok2 = abs(ndeg(d1-d0)+28.65)<0.05
    print(f"  출발 {phi0_deg:7.1f}deg -> 표시 {d0:7.1f}deg [{'OK' if ok1 else 'X'}] | "
          f"좌회전 후 {ndeg(d1-d0):+6.1f}deg [{'OK' if ok2 else 'X'}]")
