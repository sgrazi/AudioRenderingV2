%% 1- Impulse response
t = 0:1/48000:0.5;
t = t(1:end-1);

data1D;
max1 = max(abs(data1D))
%1D environment
figure('DefaultTextFontName', 'Garamond', 'DefaultAxesFontName', 'Garamond');
plot(t*1000,(data1D).^2,'k');
xlabel('Tiempo [ms]','FontSize',14);
ylabel('Cuadrado de la amplitud normalizada','FontSize',14);
title ('Respuesta impulsional medida','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'color','w');

# a = 0.1 r = 30

# rayos 100000
max2 = max(c100000)
figure('DefaultTextFontName', 'Garamond', 'DefaultAxesFontName', 'Garamond');
plot(t*1000,c100000,'k');
xlabel('Tiempo [ms]','FontSize',14);
ylabel('Intensidad [W/m^2]','FontSize',14);
title ('Respuesta impulsional simulada. 100000 rayos ','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'color','w');
print rays100000.jpg

# rayos 1000000
max3 = max(c1000000)
figure('DefaultTextFontName', 'Garamond', 'DefaultAxesFontName', 'Garamond');
plot(t*1000,c1000000,'k');
xlabel('Tiempo [ms]','FontSize',14);
ylabel('Intensidad [W/m^2]','FontSize',14);
title ('Respuesta impulsional simulada. 1000000 de rayos. ','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'color','w');
print rays1000000.jpg

# rayos 10000000
max4 = max(a01r30)
figure('DefaultTextFontName', 'Garamond', 'DefaultAxesFontName', 'Garamond');
plot(t*1000,a01r30,'k');
xlabel('Tiempo [ms]','FontSize',14);
ylabel('Intensidad [W/m^2]','FontSize',14);
title ('Respuesta impulsional simulada. 10000000 de rayos','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'color','w');
print rays10000000.jpg

#Halton

# rayos 100000
figure('DefaultTextFontName', 'Garamond', 'DefaultAxesFontName', 'Garamond');
plot(t*1000,h100000,'k');
xlabel('Tiempo [ms]','FontSize',14);
ylabel('Intensidad [W/m^2]','FontSize',14);
title ('Respuesta impulsional simulada. 100000 rayos ','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'color','w');
print raysH100000.jpg

# rayos 1000000
figure('DefaultTextFontName', 'Garamond', 'DefaultAxesFontName', 'Garamond');
plot(t*1000,h1000000,'k');
xlabel('Tiempo [ms]','FontSize',14);
ylabel('Intensidad [W/m^2]','FontSize',14);
title ('Respuesta impulsional simulada. 1000000 de rayos. ','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'color','w');
print raysH1000000.jpg

# rayos 10000000
figure('DefaultTextFontName', 'Garamond', 'DefaultAxesFontName', 'Garamond');
plot(t*1000,h10000000,'k');
xlabel('Tiempo [ms]','FontSize',14);
ylabel('Intensidad [W/m^2]','FontSize',14);
title ('Respuesta impulsional simulada. 10000000 de rayos','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'color','w');
print raysH10000000.jpg
