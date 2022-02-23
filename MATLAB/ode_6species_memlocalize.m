 % Concentrations are in units of uM (For A0 and M0)
% Length scale s is in units of um
% time is in s  it is either [T0 Tfinal] or an array of time pts.  
% A0(uM)
% k in units of uM(-1)s(-1)
%kb in s(-1)
%VAratio is units of um
function [timepts, conc] = ode_6species_memlocalize(A0,M0, VAratio, s, kfpp, kbpp, kfpm, kbpm,time)


y0=zeros(6,1);
y0(1)=A0;%P
y0(2)=M0; %M
y0(3)=0;%PM
y0(4)=0; %PP
y0(5)=0; %PPM
y0(6)=0; %MPPM
gamma=VAratio/2/s;% dimensionless!

%opt=odeset('RelTol',1E-4,'AbsTol',1E-7);
[timepts,conc] = ode23s(@(t,y) odes_6species(t,y,kfpp, kbpp, kfpm, kbpm, gamma),time,y0);

Peq=conc(end,1);
Meq=conc(end,2);
PMeq=conc(end,3);
PPeq=conc(end,4);
PPMeq=conc(end,5);
MPPMeq=conc(end,6);
display('Kaeff')
Kaeff=(MPPMeq+PPMeq+PPeq)/(Peq+PMeq)^2
display('Enhancement')
Kaeff/kfpp*kbpp
Ptot=Peq+PPeq+PMeq+2*PPeq+2*PPMeq+2*MPPMeq;
display('Total protein') %Ptot should be equal to A0
Ptot
Mtot=Meq+PMeq+PPMeq+2*MPPMeq;
display('Total Lipids') %Mtot should be equal to M0
Mtot


function dy = odes_6species(t,y,kfpp, kbpp, kfpm, kbpm, gamma)

dy = zeros(6,1);    % a column vector

dy(1) = -2*kfpp*y(1)^2+2*kbpp*y(4)-2*kfpp*y(1)*y(3)+kbpp*y(5)-kfpm*y(1)*y(2)+kbpm*y(3);%P
dy(2) =  -kfpm*y(1)*y(2)+kbpm*y(3)-2*kfpm*y(4)*y(2)+kbpm*y(5)-gamma*kfpm*y(5)*y(2)+2*kbpm*y(6); %M
dy(3) = -2*kfpp*y(3)*y(1)+kbpp*y(5)+kfpm*y(1)*y(2)-kbpm*y(3)-2*gamma*kfpp*y(3)^2+2*kbpp*y(6);%PM
dy(4) = kfpp*y(1)^2-kbpp*y(4)-2*kfpm*y(4)*y(2)+kbpm*y(5);%PP
dy(5) = 2*kfpp*y(3)*y(1)-kbpp*y(5)+2*kfpm*y(4)*y(2)-kbpm*y(5)-gamma*kfpm*y(5)*y(2)+2*kbpm*y(6); %PPM
dy(6) = gamma*kfpp*y(3)^2-kbpp*y(6)+kfpm*gamma*y(5)*y(2)-2*kbpm*y(6); %MPPM


