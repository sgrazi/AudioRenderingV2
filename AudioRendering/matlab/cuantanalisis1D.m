t = 0:1/48000:0.5;
t = t(1:end-1);
[rows,s] = size(t);
edc  = zeros(s,1);	% Decay curve

data1D;

############################################
# Medición
############################################

m = max(abs(data1D));
data = (data1D/m).^2;
measurement_sound_lvl = 10 * log10(sum(data)/(2e-5)^2)

stop = 0;
tot = sum(data);
i = 1;
while i<=s
    edc(i)= 10*log10(sum(data(i:end))/tot);
    if edc(i)<=-10 && stop==0
        EDT = t(i)*6
        stop=1;
    end
    i = i+1;
end

figure('DefaultTextFontName', 'Garamond', 'DefaultAxesFontName', 'Garamond');
plot(t*1000,edc,'k');
xlabel('Time [ms]','FontSize',14);
ylabel('Disminución del nivel sonoro [dB]','FontSize',14);
title ('Curva de decaimiento ','FontSize',14);
set(gca,'FontSize',14);
set(gcf,'color','w');

############################################
# Uniforme 100k
############################################

m = max(c100000);
data = (c100000/m);
u100k_sound_lvl = 10 * log10(sum(data)/(2e-5)^2)

stop = 0;
tot = sum(data);
i = 1;
while ~stop && i<=s
    edc(i)= 10*log10(sum(data(i:end))/tot);
    if edc(i)<=-10 && stop==0
        EDT = t(i)*6
        stop=1;
    end
    i = i+1;
end

############################################
# Uniforme 1M
############################################

m = max(c1000000);
data = (c1000000/m);
u1m_sound_lvl = 10 * log10(sum(data)/(2e-5)^2)

stop = 0;
tot = sum(data);
i = 1;
while ~stop && i<=s
    edc(i)= 10*log10(sum(data(i:end))/tot);
    if edc(i)<=-10 && stop==0
        EDT = t(i)*6
        stop=1;
    end
    i = i+1;
end

############################################
# Uniforme 10M
############################################

m = max(a01r30);
data = (a01r30/m);
u10m_sound_lvl = 10 * log10(sum(data)/(2e-5)^2)

stop = 0;
tot = sum(data);
i = 1;
while ~stop && i<=s
    edc(i)= 10*log10(sum(data(i:end))/tot);
    if edc(i)<=-10 && stop==0
        EDT = t(i)*6
        stop=1;
    end
    i = i+1;
end

############################################
# Halton 100k
############################################

m = max(h100000);
data = (h100000/m);
h100k_sound_lvl = 10 * log10(sum(data)/(2e-5)^2)

stop = 0;
tot = sum(data);
i = 1;
while ~stop && i<=s
    edc(i)= 10*log10(sum(data(i:end))/tot);
    if edc(i)<=-10 && stop==0
        EDT = t(i)*6
        stop=1;
    end
    i = i+1;
end

############################################
# Halton 1M
############################################

m = max(h1000000);
data = (h1000000/m);
h1m_sound_lvl = 10 * log10(sum(data)/(2e-5)^2)

stop = 0;
tot = sum(data);
i = 1;
while ~stop && i<=s
    edc(i)= 10*log10(sum(data(i:end))/tot);
    if edc(i)<=-10 && stop==0
        EDT = t(i)*6
        stop=1;
    end
    i = i+1;
end

############################################
# Halton 10M
############################################

m = max(h10000000);
data = (h10000000/m);
h10m_sound_lvl = 10 * log10(sum(data)/(2e-5)^2)

stop = 0;
tot = sum(data);
i = 1;
while ~stop && i<=s
    edc(i)= 10*log10(sum(data(i:end))/tot);
    if edc(i)<=-10 && stop==0
        EDT = t(i)*6
        stop=1;
    end
    i = i+1;
end