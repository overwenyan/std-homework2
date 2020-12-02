function [] = main_v4(dirname)
%main_final 声源角度估计

%% address the num of tasks
dirs = dir([dirname,'\*.wav']);
max_round = size(dirs, 1)/4;

%% calculate time delay
L = 32; % length of impulse response
N = L*4;
u = 0.01; % the step of gradient descent
direc = [];
[w_f, H] = initializing(N);
for round = 1:max_round
    % read in .wav file 
    [wave1, fs] = readWavFile(dirname, round);
    % preprocess to locate the start of speech
    [wave1, fs] = preprocess(wave1, fs);
    % initializing
    w_f1 = w_f;
    w_f2 = w_f;
    w_f3 = w_f;
    w_f4 = w_f;
    iter = size(wave1, 1)/L - 1;
%     figure;
    for cnt = 1:iter
        x = wave1(cnt*L-L+1:(cnt+1)*L, :);
        x_f = fft(x);
        % gradient descent
        w_f1 = gd(x_f(:,1), x_f(:,3), H, u, N, w_f1);
        w_f2 = gd(x_f(:,4), x_f(:,2), H, u, N, w_f2);
        w_f3 = gd(x_f(:,2), x_f(:,1), H, u, N, w_f3);
        w_f4 = gd(x_f(:,3), x_f(:,2), H, u, N, w_f4);
        % 调试
%         w2 = ifft(w_f(1:N/2));
%         w1 = ifft(w_f(N/2+1:end));
%         subplot(2,1,1);
%         stem(real(w1));
%         subplot(2,1,2);
%         stem(real(w2));
%         pause(0.3);
    end
%     [dire, theta1, theta2] = getDirec(N, w_f1, w_f2, fs);
%     direc = [direc; dire theta1 theta2];
    direc = [direc; getDirec(N, w_f1, w_f2, w_f3, w_f4, fs)];
end
%% print to file
print_result(direc, dirname);
end

function [wave0, fs0] = readWavFile(dir, round)
[wave10 fs0] = audioread([dir, '\', num2str(round), '_mic1.wav']);
[wave20 fs0] = audioread([dir, '\', num2str(round), '_mic2.wav']);
[wave30 fs0] = audioread([dir, '\', num2str(round), '_mic3.wav']);
[wave40 fs0] = audioread([dir, '\', num2str(round), '_mic4.wav']);
wave0 = [wave10 wave20 wave30 wave40];
end

function [wave1, fs] = preprocess(wave0, fs0)
fs1 = 512*14;
fs = fs0;
wave = resample(wave0, fs1, fs0);
mid = floor(size(wave, 1)/2);
[peak, ind] = max(wave(1:mid,1).^2);
wave1 = resample(wave(ind:ind+mid, :), fs, fs1);
end

function [w_f, H] = initializing(N)
% impulse response
w3 = zeros(N/2, 1);
w3(N/8) = 1;
w3_f = fft(w3);
w_f = [w3_f; zeros(N/2, 1)];
% truncate matrix
h = eye(N/2);
h(1:N/4,1:N/4) = 0;
F = dftmtx(N/2);
H = 2/N*F*h*(F');
end

function [w_f] = gd(x_f1, x_f2, H, u, N, w_f)
temp = [kron(conj(x_f1),x_f1.').*H ...
        kron(conj(x_f1),x_f2.').*H;...
        kron(conj(x_f2),x_f1.').*H ...
        kron(conj(x_f2),x_f2.').*H];
w_f = w_f - 2 * u * temp * w_f;
norm = w_f' * w_f;
w_f = w_f * sqrt(N/norm);
end

function [direc] = getDirec(N, w_f1, w_f2, w_f3, w_f4, fs)
c0 = 343;
len = 0.2;
len1 = len/sqrt(2);
% estimate time delay between 1 and 3
w1 = ifft(w_f1(N/2+1:end));
[peak, ind1] = min(real(w1));
delay1 = N/8 - ind1;
% estimate time delay between 4 and 2
w4 = ifft(w_f2(N/2+1:end));
[peak, ind2] = min(real(w4));
delay2 = ind2 - N/8;
% estimate time delay between 2 and 1
w2 = ifft(w_f3(N/2+1:end));
[peak, ind3] = min(real(w2));
delay3 = ind3 - N/8;
% estimate time delay between 3 and 2
w3 = ifft(w_f4(N/2+1:end));
[peak, ind4] = min(real(w3));
delay4 = ind4 - N/8;
% calculate direc
temp1 = delay1*c0/fs/len;
temp2 = delay2*c0/fs/len;
[theta1, theta2] = getTheta(temp1, temp2);
temp3 = delay3*c0/fs/len1;
temp4 = delay4*c0/fs/len1;
[theta3, theta4] = getTheta(temp3, temp4);
if theta3 < 45
    theta3 = 315 + theta3;
else
    theta3 = theta3 - 45;
end
if theta4 < 45
    theta4 = 315 + theta4;
else
    theta4 = theta4 - 45;
end
theta = [theta1 theta2 theta3 theta4];
mtheta = mean(theta);
[peak, ind] = max(abs(theta - mtheta));
theta(ind) = [];
direc = mean(theta);
end

function [theta1, theta2] = getTheta(temp1, temp2)
if temp2 > 0
    if temp1 > 0
        theta1 = acosd(min([1 temp1]));
        theta2 = asind(min([1 temp2]));
    else
        theta1 = acosd(max([-1 temp1]));
        theta2 = 180 - asind(min([1 temp2]));
    end
else
    if temp1 > 0
        theta1 = 360 - acosd(min([1 temp1]));
        theta2 = 360 + asind(max([-1 temp2]));
    else
        theta1 = 360 - acosd(max([-1 temp1]));
        theta2 = 180 - asind(max([-1 temp2]));
    end
end
end

function [] = print_result(direc, dir)
id = fopen([dir, '\result.txt'], 'w');
fprintf(id, '%.7e\n', direc);
% fprintf(id, '%.7e %f %f\n', direc.');
fclose(id);
end