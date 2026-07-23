%% HỆ THỐNG HUẤN LUYỆN VÀ ĐÁNH GIÁ MẠNG CNN PHÂN LOẠI HẠT ĐIỀU (TOÀN DIỆN)
clear; clc; close all;

%% 1. CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG VÀ KHỞI TẠO
exportFolder = 'D:\Hoc_Tap\3.Nam_ba\HK3\Mang_Noron\KetQua\BoSung(Robo)2';
dataRoot = 'D:\Hoc_Tap\3.Nam_ba\HK3\Mang_Noron\Dataset\RoboFlow\ChiaBoSung';

% Trỏ vào thư mục train và valid
trainDir = fullfile(dataRoot, 'train');
validDir = fullfile(dataRoot, 'valid');

if ~exist(exportFolder, 'dir'), mkdir(exportFolder); end
if ~exist(trainDir, 'dir') || ~exist(validDir, 'dir')
    error('Lỗi: Không tìm thấy thư mục dữ liệu Train/Valid tại đường dẫn đã cấu hình!');
end

%% 2. ĐỌC VÀ THỐNG KÊ SỐ LƯỢNG MẪU HẠT ĐIỀU
fprintf('>>> Đang quét thư mục dữ liệu hạt điều từ Roboflow...\n');
imdsTrain = imageDatastore(trainDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsValidation = imageDatastore(validDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Thống kê số lượng mẫu của từng tập dữ liệu
trainCount = countEachLabel(imdsTrain);
valCount = countEachLabel(imdsValidation);

% Lập bảng thống kê phân bố dữ liệu 
Lop_Hat_Dieu = string(trainCount.Label);
So_Luong_Train = trainCount.Count;
So_Luong_Validation = valCount.Count;
Tong_So_Mau = So_Luong_Train + So_Luong_Validation;
Ty_Le_Phan_Tram = round((Tong_So_Mau / sum(Tong_So_Mau)) * 100, 1);

SummaryTable = table(Lop_Hat_Dieu, So_Luong_Train, So_Luong_Validation, Tong_So_Mau, Ty_Le_Phan_Tram, ...
    'VariableNames', {'Lop_Hat_Dieu', 'So_Luong_Train', 'So_Luong_Validation', 'Tong_Cong', 'Ty_Le_Phan_Tram_Percent'});
TotalRow = table("Tổng cộng", sum(So_Luong_Train), sum(So_Luong_Validation), sum(Tong_So_Mau), 100.0, ...
    'VariableNames', {'Lop_Hat_Dieu', 'So_Luong_Train', 'So_Luong_Validation', 'Tong_Cong', 'Ty_Le_Phan_Tram_Percent'});
SummaryTable = [SummaryTable; TotalRow];

% Xuất file Excel phục vụ viết thuyết minh đồ án
excelSamplePath = fullfile(exportFolder, '1_Thong_Ke_Mau_TC2.xlsx');
writetable(SummaryTable, excelSamplePath);
disp('--- BẢNG PHÂN BỐ DỮ LIỆU HẠT ĐIỀU ĐẦU VÀO (ROBOFLOW) ---');
disp(SummaryTable);

%% 3. CẤU HÌNH KÍCH THƯỚC VÀ CHUẨN BỊ DỮ LIỆU
imageSize = [224 224 3]; % Sử dụng size 224x224 để thấy rõ đặc trưng hạt

% Ép kích thước về [224 224] tránh lỗi kênh màu
augimdsTrain = augmentedImageDatastore(imageSize(1:2), imdsTrain);
augimdsValidation = augmentedImageDatastore(imageSize(1:2), imdsValidation);
numClasses = numel(categories(imdsTrain.Labels));

%% 4. THIẾT KẾ KIẾN TRÚC MẠNG CNN 4 KHỐI ĐẦU RA
layers = [
    imageInputLayer(imageSize, 'Normalization', 'zerocenter', 'Name', 'Input_Layer')
    
    % Khối tích chập 1
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'Conv_1') 
    batchNormalizationLayer('Name', 'BatchNorm_1')
    reluLayer('Name', 'ReLU_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool_1') 
    
    % Khối tích chập 2
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'Conv_2')
    batchNormalizationLayer('Name', 'BatchNorm_2')
    reluLayer('Name', 'ReLU_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool_2') 
    
    % Khối tích chập 3
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'Conv_3')
    batchNormalizationLayer('Name', 'BatchNorm_3')
    reluLayer('Name', 'ReLU_3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool_3')
    
    % Khối tích chập 4
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'Conv_4')
    batchNormalizationLayer('Name', 'BatchNorm_4')
    reluLayer('Name', 'ReLU_4')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool_4')
    
    % Khối Dropout hạn chế Overfitting
    dropoutLayer(0.5, 'Name', 'Dropout_Layer') 
    
    % Tầng kết nối đầy đủ đầu ra tự động theo 4 lớp hạt điều
    fullyConnectedLayer(numClasses, 'Name', 'FullyConnected_Out') 
    softmaxLayer('Name', 'Softmax_Probability')          
    classificationLayer('Name', 'Output_Classification')  
];

%% 5. THIẾT LẬP THAM SỐ VÀ TIẾN HÀNH HUẤN LUYỆN TRÊN GPU
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...       
    'MaxEpochs', 20, ...         
    'MiniBatchSize', 32, ...            
    'Shuffle', 'every-epoch', ...       
    'ValidationData', augimdsValidation, ...  
    'ExecutionEnvironment', 'gpu', ...  
    'Plots', 'training-progress', ...   
    'Verbose', false);

fprintf('\n>>> Đang khởi chạy quy trình huấn luyện CNN trên GPU NVIDIA GTX 1650...\n');
[net, info] = trainNetwork(augimdsTrain, layers, options);

%% 6. DỰ ĐOÁN, VẼ MA TRẬN NHẦM LẪN VÀ LƯU MÔ HÌNH
predictedLabels = classify(net, augimdsValidation);
trueLabels = imdsValidation.Labels;
accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
fprintf('>>> Độ chính xác phân loại toàn cục trên tập Valid: %.2f%%\n', accuracy * 100);

% --- VẼ VÀ XUẤT MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ---
fprintf('>>> Đang xuất Ma trận nhầm lẫn (Confusion Matrix)...\n');
figConfMat = figure('Name', 'Confusion Matrix', 'Position', [150, 150, 800, 600]);

cm = confusionchart(trueLabels, predictedLabels);
cm.Title = 'Ma trận nhầm lẫn phân loại chất lượng Hạt Điều (4 Phân lớp)';
cm.ColumnSummary = 'column-normalized'; % Hiển thị % Precision (cột)
cm.RowSummary = 'row-normalized';       % Hiển thị % Recall (hàng)

% Màu sắc Ma trận nhầm lẫn
cm.DiagonalColor = [0.1 0.5 0.8];    % Xanh dương cho ô đoán đúng
cm.OffDiagonalColor = [0.9 0.6 0.5]; % Cam nhạt cho ô đoán sai

% Lưu file ảnh Ma trận nhầm lẫn
saveas(figConfMat, fullfile(exportFolder, '2_Confusion_Matrix.png'));
disp('>>> Đã lưu ảnh Ma trận nhầm lẫn thành công: 2_Confusion_Matrix.png');

% Lưu trữ mô hình hạt điều xuống ổ cứng
modelPath = fullfile(exportFolder, 'model_hatdieu_gtx1650.mat');
save(modelPath, 'net');

%% 7. TRÍCH XUẤT LỊCH SỬ HUẤN LUYỆN VÀ XUẤT EXCEL
valIdx = ~isnan(info.ValidationAccuracy);
allTrainAcc = info.TrainingAccuracy(valIdx)';
allTrainLoss = info.TrainingLoss(valIdx)';
allValAcc = info.ValidationAccuracy(valIdx)';
allValLoss = info.ValidationLoss(valIdx)';
allIters = find(valIdx)';

stepsPerEpoch = ceil(numel(imdsTrain.Files) / 32);
allEpochs = ceil(allIters / stepsPerEpoch);

% Lọc lấy chỉ số của bước cuối cùng trong mỗi Epoch
[~, uniqueEpochIdx] = unique(allEpochs, 'last');
Epochs = allEpochs(uniqueEpochIdx);
Iterations = allIters(uniqueEpochIdx);
Training_Accuracy = round(allTrainAcc(uniqueEpochIdx), 1);
Training_Loss = round(allTrainLoss(uniqueEpochIdx), 2);
Validation_Accuracy = round(allValAcc(uniqueEpochIdx), 2);
Validation_Loss = round(allValLoss(uniqueEpochIdx), 2);

HistoryTable = table(Epochs, Iterations, Training_Accuracy, Training_Loss, Validation_Accuracy, Validation_Loss);
excelHistoryPath = fullfile(exportFolder, 'Bang_Lich_Su_Huon_Luyen.xlsx');
writetable(HistoryTable, excelHistoryPath);

%% 8. TRÍCH XUẤT ĐẶC TRƯNG VÀ PHÂN TÍCH KHÔNG GIAN PCA
fprintf('>>> Đang trích xuất đặc trưng tầng Fully Connected và tính toán không gian PCA...\n');
featuresTest = activations(net, augimdsValidation, 'FullyConnected_Out', 'OutputAs', 'rows');

% Thực hiện thuật toán giảm chiều dữ liệu PCA
[~, score] = pca(featuresTest);
PC1 = score(:, 1); 
PC2 = score(:, 2); 
if size(score, 2) >= 3
    PC3 = score(:, 3);
else
    PC3 = zeros(size(score, 1), 1);
end

classNames = categories(trueLabels);
colors = [0.1, 0.5, 0.8; 0.9, 0.6, 0.2; 0.2, 0.7, 0.3; 0.8, 0.2, 0.2];

figPCA = figure('Name', 'PCA Analysis', 'Position', [100, 100, 1000, 800], 'Visible', 'off');

% Đồ thị 3D không gian đặc trưng
subplot(2,2,1); hold on;
for idx = 1:numel(classNames)
    classIdx = (trueLabels == classNames{idx});
    scatter3(PC1(classIdx), PC2(classIdx), PC3(classIdx), 30, colors(idx, :), 'filled', 'MarkerEdgeColor', 'k');
end
grid on; view(3); xlabel('PC1'); ylabel('PC2'); zlabel('PC3'); 
title('(a) Không gian đặc trưng hình khối 3D'); legend(classNames, 'Location', 'best');

% Đồ thị đối chứng 2D: PC1 vs PC2
subplot(2,2,2); hold on;
for idx = 1:numel(classNames)
    classIdx = (trueLabels == classNames{idx});
    scatter(PC1(classIdx), PC2(classIdx), 30, colors(idx, :), 'filled', 'MarkerEdgeColor', 'k');
end
grid on; xlabel('PC1'); ylabel('PC2'); title('(b) Đối chứng mặt phẳng PC1 vs PC2');

% Đồ thị đối chứng 2D: PC2 vs PC3
subplot(2,2,3); hold on;
for idx = 1:numel(classNames)
    classIdx = (trueLabels == classNames{idx});
    scatter(PC2(classIdx), PC3(classIdx), 30, colors(idx, :), 'filled', 'MarkerEdgeColor', 'k');
end
grid on; xlabel('PC2'); ylabel('PC3'); title('(c) Đối chứng mặt phẳng PC2 vs PC3');

% Đồ thị đối chứng 2D: PC1 vs PC3
subplot(2,2,4); hold on;
for idx = 1:numel(classNames)
    classIdx = (trueLabels == classNames{idx});
    scatter(PC1(classIdx), PC3(classIdx), 30, colors(idx, :), 'filled', 'MarkerEdgeColor', 'k');
end
grid on; xlabel('PC1'); ylabel('PC3'); title('(d) Đối chứng mặt phẳng PC1 vs PC3');

saveas(figPCA, fullfile(exportFolder, '3_PCA_Analysis.png')); 
close(figPCA);

%% 9. TRỰC QUAN HÓA DỰ ĐOÁN THỰC TẾ TRÊN MẪU THỬ NGHIỆM
figPredict = figure('Name', 'Predict Result', 'Visible', 'off');
numImagesToDisplay = 4; 
randIndices = randperm(numel(imdsValidation.Files), numImagesToDisplay);

for i = 1:numImagesToDisplay
    idx = randIndices(i); 
    [img, ~] = readimage(imdsValidation, idx); 
    predLabel = predictedLabels(idx); 
    actualLabel = trueLabels(idx);
    
    subplot(2, 2, i); 
    imshow(img); 
    
    if predLabel == actualLabel
        textColor = 'green'; 
    else
        textColor = 'red'; 
    end
    title({['Thực tế: ' strrep(char(actualLabel), '_', ' ')], ...
           ['CNN đoán: ' strrep(char(predLabel), '_', ' ')]}, 'Color', textColor, 'FontWeight', 'bold');
end

saveas(figPredict, fullfile(exportFolder, '4_Du_Doan_Thuc_Te.png')); 
close(figPredict);

fprintf('\n>>> QUÁ TRÌNH HUẤN LUYỆN, ĐÁNH GIÁ VÀ ĐÓNG GÓI SỐ LIỆU ĐÃ HOÀN TẤT THÀNH CÔNG! <<<\n');