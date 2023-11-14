n = 400; k = 7; d = 2; c = 3; e = 10;
[X,T] = gmmbvl_mixgen(n,n,k,d,c,e);
[W,M,R,Tlogl] = gmmbvl_em(X,12,0,1,0,0);
