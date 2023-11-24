%% parameters of X
muX = 2;                               % μ
sigmaX = 6;                            % σ
N = 1000;                              % sample size
bins = [5, 10, 15, 20];                % no. of bins

%% 
Xsamples = normrnd(muX, sigmaX, [N, 1]);

%% figure1
figure('Name', 'Histogram and PDF');
histogram(Xsamples, 'Normalization', 'pdf', 'EdgeColor', 'b', 'LineWidth', 1.5);
hold on;

    % theoretical pdf
    x_values = linspace(muX - 4*sigmaX, muX + 4*sigmaX, 1000);
    pdf_theoretical = normpdf(x_values, muX, sigmaX);
    plot(x_values, pdf_theoretical, 'k', 'LineWidth', 1.5);

    title('Histogram and PDF');
    xlabel('Value');
    ylabel('Probability Density');
    legend('Empirical PDF', 'Theoretical PDF');

%% figure2 - pdf
figure('Name', 'Empirical vs Theoretical PDFs');

for i = 1:length(bins)
    subplot(2, 2, i);
    histogram(Xsamples, bins(i), 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'EdgeColor', 'b', 'LineWidth', 1.5);
    hold on;
    
    % theoretical
    x_values = linspace(muX - 4*sigmaX, muX + 4*sigmaX, 1000);
    pdf_theoretical = normpdf(x_values, muX, sigmaX);
    plot(x_values, pdf_theoretical, 'k', 'LineWidth', 1.5);
    title(sprintf('PDF with %d bins', bins(i)));
    xlabel('Value');
    ylabel('Probability Density');
    legend('Empirical PDF', 'Theoretical PDF');
    
    text(muX + 1, max(pdf_theoretical)*0.5, sprintf('N(%d, %d)', muX, sigmaX), 'FontSize', 7);
    
    hold off;
end

%% figure3 - cdf
figure('Name', 'Empirical vs Theoretical CDFs');

for i = 1:length(bins)
    subplot(2, 2, i);
    
    % empirical
    [f, x_values] = ecdf(Xsamples);
    stairs(x_values, f, 'b','LineWidth', 1.5);
    hold on;
    
    % theoretical
    cdf_theoretical = normcdf(x_values, muX, sigmaX);
    plot(x_values, cdf_theoretical, 'k', 'LineWidth', 2);
    title(sprintf('CDF with %d bins', bins(i)));
    xlabel('Value');
    ylabel('Cumulative Probability');
    legend('Empirical CDF', 'Theoretical CDF');
   
    text(muX + 1, 0.8, sprintf('N(%d, %d)', muX, sigmaX), 'FontSize', 10);
    
    hold off;
end
