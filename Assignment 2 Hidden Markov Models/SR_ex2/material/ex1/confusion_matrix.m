%CONFUSION_MATRIX - Construct a confusion matrix
%
%     M = CONFUSION_MATRIX(hyp, ref)
%
%     Generates a confusion matrix from hypotheses labels given the
%     reference labels. Matrix element M(i,j) will be the proportion
%     of the samples whose correct label is i which were recognized as
%     label j. That is, each row i of the matrix represents the
%     results of the samples with correct label i, and each column j
%     the samples recognized as label j.

function M = confusion_matrix(hyp, ref)
  max_index = max(max(hyp), max(ref));
  
  for i=1:max_index
    cur_ref_index = find(ref==i);
    num_ref = length(cur_ref_index);
    for j=1:max_index
      M(i,j) = length(find(hyp(cur_ref_index)==j))/num_ref;
    end
  end
  