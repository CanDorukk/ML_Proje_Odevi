%% Iris veri seti
load fisheriris

%% Eğitim ve Test Verisini Sabitleme
cv = cvpartition(species, 'HoldOut', 0.3); % %70 eğitim, %30 test

X_train = meas(training(cv), :);
y_train = species(training(cv)); 
X_test = meas(test(cv), :);
y_test = species(test(cv));

%% Normalization
X_train = normalize(X_train);
X_test = normalize(X_test);

%% 1. Adım: Eğitim ve Test Verisi Scatter Plot (Sepal Length vs Sepal Width)

figure;
h1 = gscatter(X_train(:,1), X_train(:,2), y_train); % Eğitim verisi
hold on;
h2 = gscatter(X_test(:,1), X_test(:,2), y_test, 'rgb', 'o'); % Test verisi

title('Eğitim ve Test Verisi (Sepal Length vs Sepal Width)');
xlabel('Sepal Length');
ylabel('Sepal Width');
text(min(X_train(:,1))-0.5, min(X_train(:,2))-0.5, ...
     'Bu grafik, eğitim ve test verilerinin Sepal Length ve Sepal Width özelliklerine göre nasıl ayrıldığını göstermektedir.', ...
     'FontSize', 10, 'Color', 'black');

legend([h1(1), h2(1)], {'Eğitim Verisi', 'Test Verisi'}, 'Location', 'best');
hold off;

%% 2. Adım: SVM Karar Sınırları ve Sınıflandırma

svm_template = templateSVM('KernelFunction', 'linear', 'BoxConstraint', 1);
svm_model = fitcecoc(X_train, y_train, 'Learners', svm_template);

% Karar sınırı için meshgrid
x1_min = min(X_train(:,1)) - 1;
x1_max = max(X_train(:,1)) + 1;
x2_min = min(X_train(:,2)) - 1;
x2_max = max(X_train(:,2)) + 1;

[x1, x2] = meshgrid(linspace(x1_min, x1_max, 100), linspace(x2_min, x2_max, 100));
X_grid = [x1(:), x2(:), zeros(length(x1(:)), 2)];

y_grid = predict(svm_model, X_grid);

figure;
gscatter(x1(:), x2(:), y_grid, 'rgb', 'x');
hold on;
gscatter(X_train(:,1), X_train(:,2), y_train);
gscatter(X_test(:,1), X_test(:,2), y_test, 'rgb', 'o');
title('SVM Karar Sınırları ve Sınıflandırma');
xlabel('Sepal Length');
ylabel('Sepal Width');
text(min(X_train(:,1))-1, min(X_train(:,2))-1, ...
     'Bu grafik, SVM modelinin karar sınırlarını ve sınıflandırma sonuçlarını göstermektedir.', ...
     'FontSize', 10, 'Color', 'black');
hold off;

%% 3. Adım: AOA ile Parametre Optimizasyonu

C_values = [0.1, 1, 5, 10];
kernel_types = {'linear', 'rbf'};

best_accuracy = 0;
best_f1_score = 0;
best_C = 0;
best_kernel = '';

for i = 1:length(C_values)
    for j = 1:length(kernel_types)
        C = C_values(i);
        kernel = kernel_types{j};
        
        svm_template = templateSVM('KernelFunction', kernel, 'BoxConstraint', C);
        svm_model = fitcecoc(X_train, y_train, 'Learners', svm_template);
        
        y_pred = predict(svm_model, X_test);
        accuracy = sum(strcmp(y_pred, y_test)) / length(y_test) * 100;
        
        cm = confusionmat(y_test, y_pred);
        % Önlem: Confusion matrisinin boyutu 3x3 olabilir, burada sadece class 2 kullanılıyor (örnek amaçlı)
        if size(cm,1) >= 2
            precision = cm(2,2) / (sum(cm(:,2)) + eps);
            recall = cm(2,2) / (sum(cm(2,:)) + eps);
            f1_score = 2 * (precision * recall) / (precision + recall + eps);
        else
            f1_score = 0;
        end
        
        if f1_score > best_f1_score
            best_accuracy = accuracy;
            best_f1_score = f1_score;
            best_C = C;
            best_kernel = kernel;
        end
    end
end

fprintf('En iyi C parametresi: %.1f\n', best_C);
fprintf('En iyi Kernel: %s\n', best_kernel);
fprintf('En iyi Doğruluk: %.2f%%\n', best_accuracy);
fprintf('En iyi F1 Skoru: %.2f\n', best_f1_score);

figure;
bar_data = [best_C, best_accuracy, best_f1_score];
b = bar(bar_data);
title('En İyi Parametrelerin Sonuçları');
xlabel('Parametre');
ylabel('Değer');
xticklabels({'C', 'Doğruluk (%)', 'F1 Skoru'});
ylim([0 max(bar_data) + 10]);
grid on;

% Değerleri çubukların üstüne yaz
for i = 1:length(bar_data)
    text(b.XData(i), bar_data(i) + 1, sprintf('%.2f', bar_data(i)), ...
         'HorizontalAlignment','center', 'FontSize', 10);
end



%% 4. Adım: Doğruluk ve F1 Skoru Karşılaştırması

f1_scores = zeros(1, length(C_values));
accuracies = zeros(1, length(C_values));

for i = 1:length(C_values)
    C = C_values(i);
    svm_template = templateSVM('KernelFunction', 'linear', 'BoxConstraint', C);
    svm_model = fitcecoc(X_train, y_train, 'Learners', svm_template);
    
    y_pred = predict(svm_model, X_test);
    accuracies(i) = sum(strcmp(y_pred, y_test)) / length(y_test) * 100;
    
    cm = confusionmat(y_test, y_pred);
    if size(cm,1) >= 2
        precision = cm(2,2) / (sum(cm(:,2)) + eps);
        recall = cm(2,2) / (sum(cm(2,:)) + eps);
        f1_scores(i) = 2 * (precision * recall) / (precision + recall + eps);
    end

    fprintf('C = %.1f için Model Doğruluğu: %.2f%%\n', C, accuracies(i));
    fprintf('C = %.1f için F1 Skoru: %.2f\n', C, f1_scores(i));
end

% Sonuç grafikleri
figure;
subplot(1,2,1);
bar(C_values, accuracies);
xlabel('C Parametresi');
ylabel('Doğruluk (%)');
title('C Değerine Göre Doğruluk');
grid on;

subplot(1,2,2);
bar(C_values, f1_scores);
xlabel('C Parametresi');
ylabel('F1 Skoru');
title('C Değerine Göre F1 Skoru');
grid on;
 