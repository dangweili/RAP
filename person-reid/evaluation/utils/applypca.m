function [ux,u,m,v] = applypca(X)
%%% X is D*N
%%% ux is D*N
%%% u is D*D, each coloumn is a eig vector
%%% m is D*1
%%% v is eig values
[d,N]  = size(X);

m = mean(X,2);
X = X - m*ones(1,N); % remove mean from data

cc = cov(X',1); % compute covariance 
[cvv,cdd] = eig(cc); % compute eignvectors
[v,ii] = sort(diag(cdd), 'descend'); % sort according to eigenvalues
u = cvv(:,ii); % pick leading eigenvectors
ux=u'*X;

end

