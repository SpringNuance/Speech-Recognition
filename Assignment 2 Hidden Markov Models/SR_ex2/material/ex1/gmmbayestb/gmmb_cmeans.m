% GMMB_CMEANS  simple c-means clustering
%
% T = CMEANS(data, nclust, count)
% [T, CLUST] = CMEANS(...)
%
% data    input data, N x D matrix
% nclust  number of clusters
% count   number of iterations
%
% T       output, data labels, 1 x N vector
% CL      output, cluster centers, nclust x D matrix
%
% Author: Jarmo Ilonen
% Editor: Pekka Paalanen
%
%   Modified for T-61.5150: Retry in case of empty classes
%
% $Name:  $ $Id: gmmb_cmeans.m,v 1.1 2004/11/02 08:32:22 paalanen Exp $

function [pclass, clust]=gmmb_cmeans(pdata,nclust,count);

  
maxretries = 20;
retrycount = 1;
failed = 1;

while failed && retrycount < maxretries

  rp = randperm(size(pdata,1));
  clust = pdata(rp(1:nclust),:);
  
  failed = 0;

  for kierros=1:count,
	% compute squared distance from every point to every cluster center.
	for i=1:nclust,
		vd = pdata - repmat(clust(i,:),size(pdata,1),1);
		cet(:,i) = sum(abs(vd).^2, 2);
	end;

	% compute new cluster centers
	[a, pclass]=min(cet');

	for i=1:nclust,
                cli = find(pclass==i);
                if length(cli) > 0 
		   clust(i,:) = mean( pdata(cli, :) );
                else
                   failed = 1;
                   break;
                end
	end;
        if failed > 0
          break;
        end
  end;
  
  retrycount = retrycount + 1;
  
end;
