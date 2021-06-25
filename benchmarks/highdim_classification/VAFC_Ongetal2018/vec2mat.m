% Convert a 1-d vector into a matrix (written by Libi Hertzberg) 
function mat = vec2mat(vec, m,n)

n = (length(vec))/m;

mat = (reshape(vec, m, n));