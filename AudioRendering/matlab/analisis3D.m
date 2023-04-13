t = 0:1/48000:0.5;
t = t(1:end-1);

data3D;
%3D environment
figure('DefaultTextFontName', 'Garamond', 'DefaultAxesFontName', 'Garamond');
plot(t*1000,data3D.^2,'k');
xlabel('Tiempo [ms]','FontSize',14);
ylabel('Cuadrado de la amplitud','FontSize',14);
title ('Respuesta impulsional medida','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'color','w');
print measurement3D.jpg

# a = 0.5 r = 30

figure('DefaultTextFontName', 'Garamond', 'DefaultAxesFontName', 'Garamond');
plot(t*1000,a05r30,'k');
xlabel('Tiempo [ms]','FontSize',14);
ylabel('Intensidad [W/m^2]','FontSize',14);
title ('Respuesta impulsional simulada a=0.5 r=30 ','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'color','w');
print sim3Da05r30.jpg

# a = 0.1 r = 10

figure('DefaultTextFontName', 'Garamond', 'DefaultAxesFontName', 'Garamond');
plot(t*1000,a01r10,'k');
xlabel('Tiempo [ms]','FontSize',14);
ylabel('Intensidad [W/m^2]','FontSize',14);
title ('Respuesta impulsional simulada a=0.1 r=10 ','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'color','w');
print sim3Da01r10.jpg

# a = 0.1 r = 30

figure('DefaultTextFontName', 'Garamond', 'DefaultAxesFontName', 'Garamond');
plot(t*1000,a01r30,'k');
xlabel('Tiempo [ms]','FontSize',14);
ylabel('Intensidad [W/m^2]','FontSize',14);
title ('Respuesta impulsional simulada a=0.1 r=30 ','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'color','w');
print sim3Da01r30.jpg

#a = 0.01 r = 30

figure('DefaultTextFontName', 'Garamond', 'DefaultAxesFontName', 'Garamond');
plot(t*1000,a001r30,'k');
xlabel('Tiempo [ms]','FontSize',14);
ylabel('Intensidad [W/m^2]','FontSize',14);
title ('Respuesta impulsional simulada a=0.01 r=30 ','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'color','w');
print sim3Da001r30.jpg

#a = 0.1 r = 60

figure('DefaultTextFontName', 'Garamond', 'DefaultAxesFontName', 'Garamond');
plot(t*1000,a01r60,'k');
xlabel('Tiempo [ms]','FontSize',14);
ylabel('Intensidad [W/m^2]','FontSize',14);
title ('Respuesta impulsional simulada a=0.1 r=60 ','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'color','w');
print sim3Da01r60.jpg

#a = 0.01 r = 60

figure('DefaultTextFontName', 'Garamond', 'DefaultAxesFontName', 'Garamond');
plot(t*1000,a001r60,'k');
xlabel('Tiempo [ms]','FontSize',14);
ylabel('Intensidad [W/m^2]','FontSize',14);
title ('Respuesta impulsional simulada a=0.01 r=60 ','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'color','w');
print sim3Da001r60.jpg

figure('DefaultTextFontName', 'Garamond', 'DefaultAxesFontName', 'Garamond');
plot(t*1000,polyfit(),'k');
xlabel('Tiempo [ms]','FontSize',14);
ylabel('Intensidad [W/m^2]','FontSize',14);
title ('Respuesta impulsional simulada a=0.01 r=60 ','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'color','w');