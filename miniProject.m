clear;clc;clf;
tic
N_subcarrier = 48;
N_zeroSC = 12;
N_pilot = 4;

N_fft = 64;
N_cyclic_prefix = 16;

% one ofdm symbol length
N_ofdm_symbol = N_fft + N_cyclic_prefix;

M = 64;
b = log2(M);
N = 1.44*10^5;
input = randi([0, 1],1,N);

d=2;
E_s8=(8^2-1)*(d.^2)/12; 
Es=2*E_s8;
EsN0 = 1:1:50;
Len = length(EsN0);
N0 = Es./(10.^(EsN0./10));  
EbN0_db = EsN0 - 10*log10(4);
EbN0 = 10.^(EbN0_db./10); 
std_dev =reshape(sqrt(N0./2),1,1,length(EsN0)); %standard deviation
tic
%% convolutional code
trellis = poly2trellis(5,[37 33],37);
code_data = convenc(input, trellis);

%% 64-QAM

% partition- 6*64bits each block
code_block = reshape(code_data, 384, []); % each column represents an ofdm symbol

[x,parallel_y] = size(code_block);
codeblock_de = zeros(64, parallel_y);
% 64 rows - 64 QAM symbols; each column has its own type of modulator
for j=1:parallel_y
for i=1:6:384
    codeblock_de(ceil(i/6),j) = bi2de(code_block(i:i+5,j)','left-msb');
   
end
end

qam_code = zeros(64,parallel_y);
% 64 parallel QAMs

for i=1:64  
qam_code(i,:) = qammod(codeblock_de(i,:), M);
end
[qamx, qamy] = size(qam_code);
% num of a group of ofdm symbol is 64, 
pilot_freq = 2+2*1i;
pilot_interval = 12;

qam_code = reshape(qam_code, 1, []);
[x,parallel_y] = size(qam_code);
qam_code = reshape(qam_code, 48, []);
[sizex,sizey] = size(qam_code);

% 64*n matrix, each column corresponds to a subcarrier
Null = zeros(11, length(qam_code));            % 11 outer carriers from position 27-37
pilot = zeros(1, length(qam_code));            % pilots in position #¨C21, #¨C7, #7 and #21.
zero = zeros(1, length(qam_code));             % outer carriers in position 0

% 
F = [zero; qam_code(25:30,:); pilot; qam_code(31:34,:); pilot;qam_code(35:48,:); Null; qam_code(1:5,:); pilot; qam_code(6:18,:); pilot; qam_code(19:24,:)];


F = ifft(F); % each column represents an ofdm signal

%% Add cyclic prefix --> 80 * some num
F = [F(49:64,:); F];

%%  CHANNEL FILTERRING

% generate channel filter (gaussian, varaince 1/2)
H_rand = sqrt(1/2)*randn(1, 16) + 1i*sqrt(1/2)*randn(1,16);
H1 = [H_rand, zeros(1, 64)]';

%generate diagonal matrix
D = fft(H1);
H_bar = D.*diag(repmat(1,[80,1]));
U = dftmtx(80);
H = inv(U)*H_bar*U;

%% AWGN
noise = std_dev.*(randn(80,sizey,length(EsN0))+1i*randn(80,sizey,length(EsN0)));
received = H*F + noise;



%% demodulation
for i=1:length(EsN0)
receive_temp(:,:,i) = H\received(:,:,i);
end

% 1- discard cyclic prefix
received = receive_temp(17:80,:,:);
% each column is an OFDM symbol

% 2- FFT
ofdm_qam_received = fft(received);

% 3- cut the zero carrires
ofdm_qam_received = [ofdm_qam_received(39:43, :); ofdm_qam_received(45:57, :); ofdm_qam_received(59:64, :); ofdm_qam_received(2:7, :);ofdm_qam_received(9:12, :); ofdm_qam_received(14:27, :)];
received_lines = reshape(ofdm_qam_received, 1,[],length(EsN0));
[rcvx, rcvy, rcvz] = size(received_lines);
for i=1:length(EsN0)
    received_lines_temp(:,:,i) = [received_lines(:,:,i), zeros(1, 64 - mod(rcvy, 64))];
end
received_48parallel = reshape(received_lines_temp, 64, [], length(EsN0));
received_48parallel = reshape(received_lines, 48, [], length(EsN0));
[x,parallel_y,z] = size(received_48parallel);
% 4- serial to parallel
% 64 rows; each row has its own modulation 
for i=1:length(EsN0)
    for j=1:48
        demodulated(j,:,i) = qamdemod(received_48parallel(j,:,i), M);
    end
end
demodulated_lines = reshape(demodulated, 1, [], length(EsN0));
[demx,demy,demz] = size(demodulated_lines);

% 5- convert the demodulated symbols to bit sequence
for i=1:length(EsN0)
demodulated_bits(:,:,i) = de2bi(demodulated_lines(:,:,i),6,'left-msb');
end

% calculate the bit error rate of uncoded system
for i=1:length(EsN0)
demodulated_bit_line(:,:,i) = reshape(demodulated_bits(:,:,i)', 1, []);
error1(1,i) = 1-(sum(demodulated_bit_line(:,:,i)==code_data)/(2*N));
end
demodulated_bit_line = squeeze(demodulated_bit_line);

% 6- decode using viterbi algorithm
for snr=1:length(EsN0)
decoded(:,snr) = vitdec(demodulated_bit_line(1:length(code_data),snr),trellis,30,'trunc','hard');
error(1,snr) = 1-(sum(decoded(:,snr)==input')/N);
end

toc
 
figure(1)
semilogy(EbN0_db,error);
hold on
semilogy(EbN0_db,error1);
hold on

%% Single 64 QAM - for comparison
input_6bit = reshape(input, 6, []);
input_de = bi2de(input_6bit','left-msb')';
qam_modulated = qammod(input_de, M);
qam_noise =std_dev.*(randn(1,length(input_de),length(EsN0)) + 1i*randn(1,length(input_de),length(EsN0)));
received_qam = qam_modulated + qam_noise;

for i=1:length(EsN0)
    demodulated_qam_de(:,i) = qamdemod(received_qam(:,:,i), M);
end

for i=1:length(EsN0)
     dem_temp(:,:,i) = de2bi(demodulated_qam_de(:,i),6,'left-msb');
end
a = dem_temp(:,:,20);
for pages=1:length(EsN0)
bits(:,:,pages) = reshape(dem_temp(:,:,pages)', 1, []);
end
bit1 = bits(:,:,20);

for i=1:length(EsN0)
     error_norm_qam(1,i) = 1-(sum(bits(:,:,i)==input)/N);   
end

semilogy(EbN0_db,error_norm_qam);

legend('OFDM-1/2 rate convolutional coding','OFDM no ecc','single 64QAM');
grid on
xlabel('Eb/N0 (db)');
ylabel('BER');
title('BER');

