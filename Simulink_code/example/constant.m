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