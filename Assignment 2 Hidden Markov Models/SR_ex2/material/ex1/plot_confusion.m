%PLOT_CONFUSION - Plots a phoneme confusion matrix
%
%     PLOT_CONFUSION(M, phonemes)
%
%       M        - Confusion matrix
%       phonemes - Phoneme labels

function plot_confusion(M, phonemes)

  figure;
  imagesc(M);
  colorbar;
  H = gca;
  set(H, 'XTick', 1:length(phonemes));
  set(H, 'YTick', 1:length(phonemes));
  set(H, 'XTickLabel', phonemes');
  set(H, 'YTickLabel', phonemes');
    
  xlabel('Recognized')
  ylabel('Correct')
