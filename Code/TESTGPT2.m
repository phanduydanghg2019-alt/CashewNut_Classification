clear; clc; close all;

%% 1. CẤU HÌNH ĐƯỜNG DẪN
% Đường dẫn tới file model bác đã lưu
modelPath = 'D:\Hoc_Tap\3.Nam_ba\HK3\Mang_Noron\KetQua\BoSung(Robo)2\model_hatdieu_gtx1650.mat'; 

% Trỏ đúng vào thư mục test có chứa 4 thư mục con (Butts, Pieces, Split, Wholes)
testFolder = 'D:\Hoc_Tap\3.Nam_ba\HK3\Mang_Noron\Dataset\RoboFlow\ChiaBoSung\test'; 

if ~exist(modelPath, 'file')
    error('Lỗi: Không tìm thấy file model! Bác kiểm tra lại đường dẫn nhé.');
end
if ~exist(testFolder, 'dir')
    error('Lỗi: Không tìm thấy thư mục test!');
end

%% 2. LOAD MÔ HÌNH VÀ TẬP DỮ LIỆU
fprintf('>>> Đang tải mô hình AI từ ổ cứng...\n');
load(modelPath, 'net');

imdsTest = imageDatastore(testFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
numImages = numel(imdsTest.Files);

if numImages == 0
    error('Thư mục test đang trống! Bác kiểm tra lại đường dẫn nhé.');
end

fprintf('>>> Tìm thấy %d ảnh trong tập Test. Đang tiến hành dự đoán...\n', numImages);

%% 3. TIẾN HÀNH DỰ ĐOÁN TOÀN BỘ TẬP TEST
augimdsTest = augmentedImageDatastore([224 224], imdsTest);
[allPredLabels, allScores] = classify(net, augimdsTest);
allTrueLabels = imdsTest.Labels;

%% 4. TẠO GIAO DIỆN 1 CỬA SỔ CHỨA NHIỀU TAB
imagesPerTab = 16; % Tối đa 16 ảnh trên 1 Tab (lưới 4x4)
numTabs = ceil(numImages / imagesPerTab);

% Mở 1 Cửa sổ
mainFig = figure('Name', 'Phần mềm Phân loại Hạt Điều - Kết Quả Test', ...
                 'Units', 'normalized', 'OuterPosition', [0 0 1 1], 'NumberTitle', 'off');

% Tạo nhóm Tab quản lý các trang
tgroup = uitabgroup(mainFig);

for tabIdx = 1:numTabs
    % Xác định phạm vi ảnh cho Tab hiện tại
    startIdx = (tabIdx - 1) * imagesPerTab + 1;
    endIdx = min(tabIdx * imagesPerTab, numImages);
    currentNumImages = endIdx - startIdx + 1;
    
    % Tạo một Tab mới
    tabName = sprintf('Trang %d (Ảnh %d - %d)', tabIdx, startIdx, endIdx);
    currentTab = uitab(tgroup, 'Title', tabName);
    
    % Tính toán số hàng và cột cho lưới trong Tab này
    cols = 4;
    rows = ceil(currentNumImages / cols);
    
    localCount = 1;
    for i = startIdx:endIdx
        img = readimage(imdsTest, i);
        
        predLabel = char(allPredLabels(i));
        actualLabel = char(allTrueLabels(i));
        confidence = max(allScores(i, :)) * 100;
        
        % Đánh màu chữ: Đoán ĐÚNG = Màu Xanh lá, Đoán SAI = Màu Đỏ
        if strcmp(predLabel, actualLabel)
            textColor = [0, 0.6, 0];
        else
            textColor = [0.8, 0, 0];
        end
        
        % Tạo subplot gắn vào tab hiện tại
        ax = subplot(rows, cols, localCount, 'Parent', currentTab);
        imshow(img, 'Parent', ax);
        
        % CHỮ TRÊN: Kết quả Dự đoán
        title(ax, sprintf('Dự đoán: %s (%.1f%%)', predLabel, confidence), ...
              'Color', textColor, 'FontWeight', 'bold', 'FontSize', 10, 'Interpreter', 'none');
              
        % CHỮ DƯỚI: Nhãn Thực tế
        xlabel(ax, sprintf('Thực tế: %s', actualLabel), ...
               'Color', 'k', 'FontWeight', 'bold', 'FontSize', 10, 'Interpreter', 'none');
        
        localCount = localCount + 1;
    end
end

%% 5. IN BÁO CÁO TỔNG KẾT XUỐNG COMMAND WINDOW
testAccuracy = sum(allPredLabels == allTrueLabels) / numImages;
fprintf('\n====================================================\n');
fprintf('>>> ĐÃ TẠO GIAO DIỆN VỚI %d TAB (TRANG) HIỂN THỊ!\n', numTabs);
fprintf('>>> ĐỘ CHÍNH XÁC TỔNG THỂ TRÊN TẬP TEST: %.2f%%\n', testAccuracy * 100);
fprintf('====================================================\n');