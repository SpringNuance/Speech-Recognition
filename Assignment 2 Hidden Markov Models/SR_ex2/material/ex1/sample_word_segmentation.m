%SAMPLE_WORD_SEGMENTATION
%
%    Adds the segmentation of the sample word (ex1) to the current
%    plot. Assumes the frame rate of 10ms.

function sample_word_segmentation

pho_times = [20 26 34 43 50 59 70 77 85 94 102 109 115 123 131];
pho_labels={'p','y','ö','rr','e','m','y','r','s','k','y','i','s','t','ä'};
H=gca; set(H, 'XTick', pho_times);
set(H, 'XTickLabel', pho_labels); set(H, 'FontSize', 20);
