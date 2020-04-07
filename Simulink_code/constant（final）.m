m=1000;
J_m=0.25;
J_w=4;
r_w=0.28;
i_g=2;
yita=0.8;
g=9.8;
alpha=0;
C_D=0.35;
rou=1.226;
A=2.2;
C_t=42;
C_e=4.4;
Phi=0.028;
Ra=0.088;
K_p=4;
T_i=1;
f=0.018;
time=round(rand(1,1)*4000);
x=0.1:0.1:time;
y=zeros(1,time*10);
for i=2:1:time*10
    temp=rand(1,1);
    if ((temp>0.35)&&(temp<0.65))
        y(i)=y(i-1);
    else
        if temp<0.35
            y(i)=y(i-1)-temp*5;
        else
            y(i)=y(i-1)+(temp-0.65)*5;
        end
    end
    if y(i)>70
        y(i)=70;
    end
    if y(i)<0
        y(i)=0;
    end
end
%%常数设置
I=10;
P=200;
c=10800;
C=c/137.5;
k=C;
Tao=1;
r0=1;
rp=10000;
cb=1;

%%拟合

%%U_oc
%Uoc0=[3091 3138 3175 3208 3238 3268 3296 3328 3364]/1000;非原始数据
%Uoc0=C*[2981 3116 3167 3201 3238 3266 3298 3336 3364]/1000;原始数据
%Uoc0=C*[2981 3160 3195 3216 3238 3266 3288 3317 3357]/1000;
SOCSOC=[0 0.01 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99];
UUoc0=[0 858 3131 3150 3170 3205 3230 3248 3266 3280 3290 3295 3330 3550]/1000;

%%其他参数
SOC0=[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
tao0=Tao*1/k*[3734 3598 3193 3153 3183 3288 3288 3142 3493]/100;
Ro0=r0*1/k*[113 114 115 112 106 111 97 95 99]./100000;
Rp0=rp*1/k*[40 34 32 27 26 23 24 21 30]./1000000;
Cb0=cb*k*[3788 3818 3848 3857 3861 3869 3894 3904 3906]./10;
SOC=0:0.0001:1;
tao=interp1(SOC0,tao0,SOC, 'spline');
UUoc=interp1(SOCSOC,UUoc0,SOC, 'spline');
Uoc=UUoc*4.2/UUoc(10001);
Ro=interp1(SOC0,Ro0,SOC, 'spline');
Rp=interp1(SOC0,Rp0,SOC, 'spline');
Cb=interp1(SOC0,Cb0,SOC, 'spline');
plot(SOCSOC,UUoc0,'o',SOC,Uoc)
hold on
%plot(SOCSOC,UUoc0,'o',SOC,UUoc)

m=1000;
J_m=0.25;
J_w=4;
r_w=0.28;
i_g=2;
yita=0.8;
g=9.8;
alpha=0;
C_D=0.35;
rou=1.226;
A=2.2;
C_t=42;
C_e=4.4;
Phi=0.028;
Ra=0.088;
K_p=4;
T_i=1;
f=0.018;
time=round(rand(1,1)*4000);
x=0.1:0.1:time;
y=zeros(1,time*10);
for i=2:1:time*10
    temp=rand(1,1);
    if ((temp>0.35)&&(temp<0.65))
        y(i)=y(i-1);
    else
        if temp<0.35
            y(i)=y(i-1)-temp*5;
        else
            y(i)=y(i-1)+(temp-0.65)*5;
        end
    end
    if y(i)>70
        y(i)=70;
    end
    if y(i)<0
        y(i)=0;
    end
end

%%常数设置
I=10;
P=200;
c=10800*0.9;
C=c/137.5;
k=C;
Tao=1;
r0=1;
rp=10000;
cb=1;

%%拟合

%%U_oc
%Uoc0=[3091 3138 3175 3208 3238 3268 3296 3328 3364]/1000;非原始数据
%Uoc0=C*[2981 3116 3167 3201 3238 3266 3298 3336 3364]/1000;原始数据
%Uoc0=C*[2981 3160 3195 3216 3238 3266 3288 3317 3357]/1000;

SOCSOC=[0 0.01 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99];
UUoc0=[0 858 3131 3150 3170 3205 3230 3248 3266 3280 3290 3295 3330 3550]/1000; %平台电压区较大

% SOCSOC=[0 0.01 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99];
% UUoc0=[0 500 3000 3100 3205 3230 3248 3266 3280 3290 3290 3330 3550]/1000;
%%其他参数
SOC0=[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
tao0=Tao*1/k*[3734 3598 3193 3153 3183 3288 3288 3142 3493]/100;
Ro0=r0*1/k*[113 114 115 112 106 111 97 95 99]./100000;
Rp0=rp*1/k*[40 34 32 27 26 23 24 21 30]./1000000;
Cb0=cb*k*[3788 3818 3848 3857 3861 3869 3894 3904 3906]./10;
SOC=0:0.0001:1;
tao=interp1(SOC0,tao0,SOC, 'spline');
UUoc=interp1(SOCSOC,UUoc0,SOC, 'spline');
Uoc=UUoc*4.2/UUoc(10001);
Ro=interp1(SOC0,Ro0,SOC, 'spline');
Rp=interp1(SOC0,Rp0,SOC, 'spline');
Cb=interp1(SOC0,Cb0,SOC, 'spline');
% plot(SOCSOC,UUoc0,'o',SOC,Uoc)
% hold on
plot(SOCSOC,UUoc0,'o',SOC,UUoc)

